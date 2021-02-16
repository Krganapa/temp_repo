from mesa_geo import GeoAgent, AgentCreator, GeoSpace
from mesa.time import BaseScheduler
from mesa import datacollection
from mesa import Model
from scipy import stats 
from inspect import signature
import math
import os
import sys
import warnings


#Shapely Imports
from shapely.geometry import Polygon, Point, LineString
import shapely


#Data Analysis 
import geopandas as gpd
import pandas as pd
import numpy as np
import random
import csv

# Configuration Data and Files
import configparser

#Plot
import matplotlib.pyplot as plt

# Aerosol Transmission
import aerosol_new

# Prefix for config data
#os.chdir(os.path.dirname(sys.path[0]))
config_file_path_prefix = './config/'


# parser viz config data
viz_ini_file = 'vizparams.ini'

parser_viz = configparser.ConfigParser()
parser_viz.read(config_file_path_prefix + viz_ini_file)

default_section = parser_viz['DEFAULT_PARAMS']


# parser disease config data

disease_params_ini = 'diseaseparams.ini'
parser_dis = configparser.ConfigParser()
parser_dis.read(config_file_path_prefix + disease_params_ini)
incubation = parser_dis['INCUBATION']


# NPI config data


npi_params_ini = 'NPI.ini'
parser_npi = configparser.ConfigParser()
parser_npi.read(config_file_path_prefix + npi_params_ini)

# School Params 
parser_school = configparser.ConfigParser()
parser_school.read(config_file_path_prefix + 'schoolparams.ini')
intervention_section = parser_school['INTERVENTION']
ventilation_type = intervention_section['ventilation_type']
teacher_testing_frequency = eval(intervention_section['teacher_testing_freq'])
teacher_vaccination_proportion = float(intervention_section['teacher_vaccine_prob']) 
student_testing_frequency = eval(intervention_section['student_testing_freq'])

def generate_random(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            return pnt
        
        
        
        
        
        
def find_room_type(room_agents, room_type):
    """
    room_agents: a list containing room agents
    room_type: a valid string of room type: [None, 'restroom_grade_boys', 'lunch_room', 'classroom_grade',
       'restroom_all', 'restroom_grade_girls', 'restroom_KG',
       'classroom_KG', 'community_room', 'library',
       'restroom_special_education', 'restroom_faculty',
       'classroom_special_education', 'health_room', 'faculty_lounge',
       'classroom_preschool', 'restroom_preschool']
    """
    out = []
    for agent in room_agents:
        if agent.room_type == room_type:
            out.append(agent)
    return out





def load_map(file_path):
    '''
    This is specific to the current school layout at this point, should be modified later in the future
    assume the input school map file is sufficient
    '''
    school_geometry = gpd.read_file(file_path)

    # move second floor to map bottom
    p = Polygon([(900,-800), (900,-1100), (1650,-1100), ( 1650,-800)])
    sf = school_geometry.geometry.intersects(p)
    new_sf = school_geometry[sf].translate(yoff=-800)
    school_geometry.loc[sf,["geometry"]] = new_sf
    school_gdf = school_geometry.rename(columns={"object": "room_type"})


    # generate recess area
    grd_recess = Polygon([(750, -1000), (750, -710), (1615, -710), (1615, -1480), (1450, -1480), (1450, -1200), 
                                                               (900, -1200), (900, -1000)])
    pkg_recess = Polygon([(430, -1400), (430, -1150), (660, -1150), (660, -1400)])


    school_gdf = school_gdf[['Id', 'room_type', 'geometry']]
    school_gdf.loc[len(school_gdf)] = [90000, 'recess_yard', grd_recess]
    school_gdf.loc[len(school_gdf)] = [90001, 'recess_yard', pkg_recess]

    return school_gdf

def return_breathing_flow_rate(breathing_flow_rate):
    # Converting from activity type to physical parameters for breathing_flow_rate
    if breathing_flow_rate == 'resting':
        breathing_flow_rate = 0.4927131 #0.29 ft3/min
    elif breathing_flow_rate == 'moderate_exercise':
        breathing_flow_rate = 2.3446349 #1.38 ft3/min
    elif breathing_flow_rate == 'light_exercise':
        breathing_flow_rate = 1.3761987 #0.8 ft3/min

    return breathing_flow_rate



    
def return_air_exchange_rate(air_exchange_rate):
    # Converting from activity type to physical parameters for air_exchange_rate
    if air_exchange_rate == 'open_windows':
        air_exchange_rate = 2  #hr-1
    elif air_exchange_rate == 'closed_windows':
        air_exchange_rate = 0.3 #hr-1
    elif air_exchange_rate == 'mechanical_ventilation':
        air_exchange_rate = 3  #hr-1
    elif air_exchange_rate == 'fans':
        air_exchange_rate = 6  #hr-1
    elif air_exchange_rate == 'advanced_mechanical_ventilation':
        air_exchange_rate = 8  #hr-1

    return air_exchange_rate


def return_mask_passage_prob(mask_passage_prob):
    # Converting from activity type to physical parameters for mask_passage_prob

    if mask_passage_prob == None:
        mask_passage_prob = 1
    elif mask_passage_prob == 'Cotton':
        mask_passage_prob = 0.5
    elif mask_passage_prob == 'Multilayer':
        mask_passage_prob = 0.7
    elif mask_passage_prob == 'Surgical':
        mask_passage_prob = 0.9
    elif mask_passage_prob == 'N95':
        mask_passage_prob = 0.95

    return mask_passage_prob

def return_exhaled_air_inf(exhaled_air_inf):
    # Converting from activity type to physical parameters for Exhaled air infectivity
    if exhaled_air_inf == 'talking_whisper':
        exhaled_air_inf = 28.9580743587 #q/m3
    
    elif exhaled_air_inf == 'talking_loud':
        exhaled_air_inf = 141.965193807 #q/m3

    elif exhaled_air_inf == 'breathing_heavy':
        exhaled_air_inf = 8.82868120692 #q/m3

    elif exhaled_air_inf == 'talking_normal':
        exhaled_air_inf = 72.0420386485 #q/m3

    return exhaled_air_inf

class Human(GeoAgent):
            
    # plot config

    marker = default_section['marker']
    colordict = {"healthy": default_section['healthy'], 'exposed': default_section['exposed'], 'infectious': default_section['infectious']}
    edgedict = {"healthy": default_section['healthy_edge'], 'exposed': default_section['exposed_edge'], 'infectious': default_section['infectious_edge']}
    sizedict = {"healthy": default_section['healthy_size'], 'exposed': default_section['exposed_size'], 'infectious': default_section['infectious_size']}
    
    
    
    
    # dummy config for data collection
    viral_load = None
    
    
    
    # UPDATE 10/16: move stats to class level
     

    
    
    
    
    # infectious curve config
    ###################################### 
    # based on gamma fit of 10000 R code points

    shape, loc, scale = (float(incubation['shape']), float(incubation['loc']), float(incubation['scale']))

    # infectious curve
    range_data = list(range(int(incubation['lower_bound']), int(incubation['upper_bound']) + 1))
    infective_df = pd.DataFrame(
        {'x': range_data,
         'gamma': list(stats.gamma.pdf(range_data, a=shape, loc=loc, scale=scale))
        }
    )
    #########################################

    
    def __init__(self, unique_id, model, shape, room, health_status = 'healthy'):
        super().__init__(unique_id, model, shape)
        

        
        # disease config

        self.health_status = health_status
        prevalence = float(parser_dis['ASYMPTOMATIC_PREVALENCE']['prevalence'])
        self.asymptomatic = np.random.choice([True, False], p = [prevalence, 1-prevalence])
        self.symptoms = False

        self.tested = False
        self.vaccinated = False
        
        # UPDATE 10/17: delay infection by 1 day to avoid infection explosion
        self.infective = False
        
        # symptom onset countdown config
        ##########################################
        # From 10000 lognorm values in R
        countdown = parser_dis['COUNTDOWN']
        shape, loc, scale =  (float(countdown['shape']), float(countdown['loc']), float(countdown['scale']))

        lognormal_dist = stats.lognorm.rvs(shape, loc, scale, size=1)

        num_days = min(np.round(lognormal_dist, 0)[0], int(countdown['upper_bound'])) # failsafe to avoid index overflow
        self.symptom_countdown = int(num_days)
        #######################################
        
        

        
        self.room = room
        self.x = self.shape.x
        self.y = self.shape.y


        
        

 
    def update_shape(self, new_shape):
        self.shape = new_shape
        self.x = self.shape.x
        self.y = self.shape.y
        
    
    def __update(self):
        # UPDATE 10/16: reorganized things from Bailey's update
        # TODO: currently mask has no functionality other than reducing transmission distance, is this faithful?

        # mask wearing reduces droplet transmission max range
        # infection above max range is considered as aerosal transmission
        if self.mask and not (self.model.activity[self.room.schedule_id] == 'lunch'):
            neighbors = self.model.grid.get_neighbors_within_distance(self, int(parser_npi['MASKS']['infection_distance']))
        else:
            neighbors = self.model.grid.get_neighbors_within_distance(self, int(parser_npi['NO_NPI']['infection_distance']))

        
        # UPDATE 10/16: infectious has made obsolete due to infectious curve covering after symptom onset fit
        # credit Bailey Man
        '''
        if self.health_status == 'infectious':
            #TODO: sliding Distance calculation for neighbor infection
            # a scale for health status


            # Loop through infected agent's neighbors that are within 3
        
            for neighbor in neighbors:
                
                ### 
                #temp value, replace this with distance -> viral load -> infection_prob calculation eventually
                temp_prob = .857 # previous value used

                # Check class is Human                            
                if issubclass(type(neighbor), Human):

                    infective_prob = np.random.choice ([True, False], p = [temp_prob, 1 - temp_prob])
                    if infective_prob and self.__check_same_room(neighbor):
                        neighbor.health_status = 'exposed'
        '''    

                        
                        
                        
        if self.health_status == 'exposed' and self.infective:

            # normalize symptom countdown value to infectious distribution value
            # 0 being most infectious
            # either -10 or 8 is proven to be too small of a chance to infect others, thus covering asympotmatic case


            countdown_norm = min(int(incubation['upper_bound']), max(int(incubation['lower_bound']), 0 - self.symptom_countdown))
            temp_prob = self.infective_df[self.infective_df['x'] == countdown_norm]['gamma'].iloc[0]

            
            for neighbor in neighbors:

                # Check class is Human                            
                if issubclass(type(neighbor), Human):
                    if neighbor.unique_id != self.unique_id:                   
                        # TODO: update this to a more realistic scale
                        agent_distance = self.shape.distance(neighbor.shape)

                        try:
                            dist_bias = np.sqrt(min(1, 1/agent_distance))
                        except:
                            dist_bias = 1


                        # row a dice of temp_prob*dist_bias chance to expose other agent
                        infective_prob = np.random.choice ([True, False], p = [temp_prob*dist_bias, 1 - (temp_prob*dist_bias)])

                        if infective_prob and self.__check_same_room(neighbor) and not self.vaccinated:
                            neighbor.health_status = 'exposed'

                        
    def __check_same_room(self, other_agent):
        '''
        check if current agent and other agent is in the same room
        
        the purpose of this function is to make sure to eliminate edge cases that one agent near the wall of its room
        infects another agent in the neighboring room
        
        this is at this iteration of code only implemented for class purpose, as unique id check is way more efficient
        
        later implementation should add attribute to human agent for current room
        
            other_agent: other agent to check
            returns: boolean value for if the two agents are in the same room
        '''
        same_room = True
        if self.model.activity[self.room.schedule_id] == 'class':
            same_room = (self.room.unique_id == other_agent.room.unique_id)
        return same_room
    
    
    def __move(self, move_spread = 12, location = None):
        '''
        Checks the current location and the surrounding environment to generate a feasbile range of destination (the area
        of a circle) for agent to move to.
        The radius of the circle is determined by agent's move_factor.
        Assigns new point to override current point.
        '''   
        
        if not location:
            location = self.room
        move_spread = location.shape.intersection(self.shape.buffer(move_spread))

        if hasattr(location, 'seating_pattern'):
            if location.seating_pattern == 'circular':
                if hasattr(self, 'desk'):
                    try:
                        move_spread = move_spread.difference(self.desk)
                    except ValueError:
                        move_spread = None

        if move_spread is not None:
            minx, miny, maxx, maxy = move_spread.bounds
            while True:
                pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))            
                # check if point lies in true area of polygon
                if move_spread.contains(pnt):
                    self.update_shape(pnt)
                    break
    
    
    def plot(self):
        plt.plot(
            self.shape.x, self.shape.y, 
            marker=self.marker, 
            mec = self.edgedict[self.health_status],
            color = self.colordict[self.health_status],
            markersize = self.sizedict[self.health_status]
                )
        

                        
        
        
        





class Student(Human):
    def __init__(self, unique_id, model, shape, room, health_status = 'healthy', mask_on=False):
        super().__init__(unique_id, model, shape, room, health_status)

        viz_ini_file = 'vizparams.ini'
        parser = configparser.ConfigParser()
        parser.read(config_file_path_prefix + viz_ini_file)
        student_viz_params = parser['STUDENT']

        self.grade = self.room.room_type.replace('classroom_', '')
        self.mask = mask_on       
        self.mask_type = None 
        self.seat = Point(self.shape.x, self.shape.y)
        self.marker = student_viz_params['marker']
        
        
        

        self.out_of_place = False        
        self.prev_activity = None
        self.lunch_count = 0
        self.desk = None
        self.breathing_rate = None
        self.breathing_activity = None
        
        
        
        
        
                

                

                
                
    def step(self):
        self._Human__update() 
        
        activity = self.model.activity[self.room.schedule_id]
        # case 1: student in class
        if activity == 'class': 
            if self.prev_activity != activity:
                self.prev_activity = activity
                self.update_shape(self.seat)
                self.breathing_rate = 'resting'
                self.breathing_activity = np.random.choice(['talking_whisper', 'talking_loud', 'breathing_heavy'], p=[0.2, 0.05, 0.75])

                
            if self.room.prob_move:
                self.out_of_place = True
                self._Human__move()
            else:
                if self.out_of_place:
                    self.update_shape(self.seat)
                    self.out_of_place = False
                    
                
                
        # case 2: student in recess            
        elif activity == 'recess':
            ### TODO: recess yard assignment is hard coded ###
            self.breathing_rate = 'moderate_exercise'
            self.breathing_activity = np.random.choice(['talking_loud', 'talking_normal', 'breathing_heavy'], p=[0.4, 0.1, 0.5])
            location = self.model.recess_yards[0]
            if self.grade != 'grade':
                location = self.model.recess_yards[1]
                
            if self.prev_activity != activity:
                self.update_shape(generate_random(location.shape))
                self.prev_activity = activity
            
            self._Human__move(move_spread=15, location = location)
        
        
        
        # case 3: student having lunch
        elif activity == 'lunch':
            self.breathing_rate = 'resting'
            self.breathing_activity = 'breathing_heavy'
            #in class lunch case
            if self.model.inclass_lunch or self.grade != 'grade':
                if self.prev_activity != activity:
                    self.update_shape(self.seat)
                    self.prev_activity = activity
                    self.out_of_place = True
                    self._Human__move()
                else: 
                    if self.out_of_place:
                        self.update_shape(self.seat)
                        self.out_of_place = False

                    
            #in cafeteria lunch case
            else:
                if self.prev_activity != activity:
                    self.update_shape(generate_random(self.model.lunchroom.shape))
                    self.prev_activity = activity

                # enter lunch cafeteria, move free for 2 iteration
                if self.lunch_count < 2:
                    self._Human__move(move_spread=40, location = self.model.lunchroom)

                # finds seat, settle in seat until finish lunch
                elif self.lunch_count == 2:
                    self.update_shape(self.model.lunchroom.seats[0])
                    # remove seat from seat list if the student occupies it
                    self.model.lunchroom.seats = self.model.lunchroom.seats[1:]

                # release seat back to seat list
                elif self.lunch_count == 7:
                    self.model.lunchroom.seats.append(self.shape)
                    self.lunch_count = -1

            
                self.lunch_count += 1
    
                        
                            
                            
                            




                
                
    
class Teacher(Human):
    def __init__(self, unique_id, model, shape, room, health_status = 'healthy',mask_on=True):
        super().__init__(unique_id, model, shape, room, health_status)
        self.mask = mask_on
        self.mask_type = None
        self.classroom = self.room # TODO: for future development enabling Teachers to move to other room during non-class time
        self.breathing_rate = None
        self.breathing_activity = None

        viz_ini_file = 'vizparams.ini'
        parser = configparser.ConfigParser()
        parser.read(config_file_path_prefix + viz_ini_file)

        self.vaccinated = np.random.choice([True, False], p=[teacher_vaccination_proportion, 1-teacher_vaccination_proportion])

        teacher_viz_params = parser['TEACHER']
        
        self.marker = teacher_viz_params['marker']
        self.edgedict = {"healthy": teacher_viz_params['healthy_edge'], 'exposed': teacher_viz_params['exposed_edge'], 'infectious': teacher_viz_params['infectious_edge']}
        self.sizedict = {"healthy": teacher_viz_params['healthy_size'], 'exposed': teacher_viz_params['exposed_size'], 'infectious': teacher_viz_params['infectious_size']}
    
    
    def step(self):
        self.breathing_rate = 'light_exercise'
        self.breathing_activity = np.random.choice(['talking_loud', 'talking_normal', 'breathing_heavy'], p=[0.5, 0.25, 0.25])
        self._Human__update()
        self._Human__move()
                
        





class Classroom(GeoAgent):
    
    
    # dummy config for data collection
    health_status = None
    symptoms = None
    x = None
    y = None
    
    
    def __init__(self, unique_id, model, shape, room_type):
        super().__init__(unique_id, model, shape)
        self.occupants = []
        self.aerosol_transmission_rate = []
        #self.occupants = occupants #List of all occupants in a classroom
        #self.barrier = barrier_type
        self.room_type = room_type
        self.seating_pattern = None
        self.viral_load = 0
        self.prob_move = False
        self.schedule_id = None
        self.environment = ventilation_type
        self.floor_area = shapely.affinity.scale(shape, xfact=0.37, yfact=0.37, origin=(0,0)).area
        self.height = 12

    
    def step(self):
        # roll weighted probability for current step having in-class activity
        self.prob_move = np.random.choice([True, False], p = [0.2, 0.8])
        occupants = [a for a in list(self.model.grid.get_intersecting_agents(self)) if issubclass(type(a), Human)]
        exposed = [a for a in occupants if a.health_status != "healthy"]
        exposed = [a for a in exposed if a.infective]

        num_occupants = len(occupants)
        num_exposed = len(exposed)

        exposure_time = 5/60

        mean_breathing_rate = np.mean([return_breathing_flow_rate(a.breathing_rate) for a in occupants])
        mean_infectivity = np.mean([return_exhaled_air_inf(a.breathing_activity) for a in occupants])
        ACH = return_air_exchange_rate(self.environment)
        floor_area = self.floor_area
        mask_passage_prob = 1
        height = self.height


        transmission_rate = aerosol_new.return_aerosol_transmission_rate(floor_area=floor_area, room_height=height,
        air_exchange_rate=ACH, aerosol_filtration_eff=0, relative_humidity=0.69, breathing_flow_rate=mean_breathing_rate, exhaled_air_inf=mean_infectivity,
        mask_passage_prob=mask_passage_prob)

        transmission_rate *= exposure_time
        transmission_rate *= num_exposed #To be changed to some proportion to get infectious
        transmission_rate *= (num_occupants - num_exposed)

        transmission_rate = transmission_rate/num_occupants

        self.aerosol_transmission_rate.append(transmission_rate)
    
        # this upates room viral_load to an arbitrary that's scaled with num of covid patients and area of room
        # should change to volume of room later
        self.viral_load += len([a for a in occupants if a.health_status != "healthy"])/self.shape.area
        
        self.viral_load = max(self.viral_load-0.002, 0)
        
        if self.schedule_id is not None:
            if "classroom" in self.room_type:
                if self.model.activity[self.schedule_id] != 'class':
                    self.ventilate()
        elif (self.room_type == 'lunch_room') and (not ('lunch' in self.model.activity.to_numpy())):
            self.ventilate()
            
        
    def ventilate(self):
        ### TODO: check literature for aerosal infection ###
        self.viral_load = max(self.viral_load-0.05, 0)
        
    def generate_desks(self, shape, radius_desk=12, spacing_desk=20):
        
        desks = []
        minx, miny, maxx, maxy = shape.bounds

        # Starting from left to right start assigning centers of desks
        x_start, y_start = minx + radius_desk, maxy - radius_desk
        
        # Small offsets to account for imperfect corners in digitization.
        x_start += 0.2
        y_start -= 0.2

        # Create the desks using the right buffers.
        desk = Point(x_start, y_start).buffer(radius_desk)
        net_space = radius_desk + spacing_desk
        
        x_coordinate = x_start
        y_coordinate = y_start
        
        # Loop to create the desks
        while True:
            desks.append(desk)        
            
            x_coordinate += net_space
            
            if (y_coordinate < miny) and (x_coordinate > maxx):         
                break
                        
            if x_coordinate > maxx:
                x_coordinate -= net_space
                x_coordinate = x_start 
                y_coordinate = y_coordinate - net_space
                
            
            desk = Point(x_coordinate, y_coordinate).buffer(radius_desk)    
        
        desks = gpd.GeoSeries(desks)

        # Figure out the desks which intersect the classroom and reject those. 
        # Note that we do not use contains operation. 

        desks  = desks[desks.apply(lambda desk: shape.intersects(desk))]
        
        return desks


    def generate_seats_circular(self, shape, desks, N, num_children_per_desk=5):

        # Helper function to help generate seat positions given a set of desks.
        def return_coords(desk, num_children_per_desk):
            boundary = list(desk.boundary.coords)
            step = len(boundary) // num_children_per_desk
            boundary = pd.Series(boundary[::step])
            boundary = boundary[:num_children_per_desk]
            return boundary.apply(Point).values.tolist() 

        # Return results using efficient vectorized apply functions.
        result = desks.apply(return_coords, args=(num_children_per_desk,))
        dataframe = pd.DataFrame((desks, result)).T
        dataframe = dataframe.rename({0:'desk', 1:'seats'}, axis=1)
        desk_series = dataframe.apply(lambda row: [row['desk'] for i in range(len(row['seats']))], axis=1)
        
        desk_series = desk_series.sum()
        result = result.sum()
        
        final_df = pd.DataFrame()
        final_df['desk'] = desk_series
        final_df['seat'] = result
        
        final_df['desk_id'] = final_df['desk'].apply(str)
        
        #Check those seats which are extremely close to the boundary and retain them. This is to compensate for digitization errors.
        to_drop = final_df[(gpd.GeoSeries(final_df['seat']).distance(shape) >= 0.1) & (~final_df['seat'].apply(lambda seat: shape.contains(seat)))].drop_duplicates(subset=['desk_id'])
        to_drop = to_drop.desk_id.values
        
        self.num_children_per_desk = num_children_per_desk

        # Only return N seating positions.
        return final_df[~final_df.desk_id.isin(to_drop)][:N]


    def generate_seats(self, N, width, style='individual', num_children_per_desk=None):
        
        self.seats = []
        shape = self.shape
        
        if style == 'individual':         
            center = shape.centroid
            md = math.ceil(N**(1/2))
            pnt = Point(center.x - width*md//2, center.y - width*md//2)
            for i in range(md):
                for j in range(md+1):
                    self.seats.append(Point(pnt.x + i*width, pnt.y + j*width))
        
        elif style == 'circular':
            
            if num_children_per_desk is not None:
                # Use this if a custom number of children per desk is input.
                dataframe_seats_desks = self.generate_seats_circular(shape, self.generate_desks(shape), N, num_children_per_desk=num_children_per_desk)
            else:
                dataframe_seats_desks = self.generate_seats_circular(shape, self.generate_desks(shape), N)
            
            self.seats = dataframe_seats_desks.seat.tolist()

            self.desks = dict(dataframe_seats_desks[['desk_id', 'desk']].drop_duplicates(['desk_id']).set_index('desk_id')['desk'])
                                
    def generate_seats_lunch(self, xwidth, ywidth):
        
        self.seats = []
        xmin, ymin, xmax, ymax = self.shape.bounds
        xcoords = xmin + xwidth
        ycoords = ymin + ywidth
        
        y_pointer = ycoords
        x_pointer = xcoords
        
        while (xcoords < xmax):
            
            while (ycoords < ymax):
                self.seats.append(Point(xcoords, ycoords))
                ycoords += ywidth
                
            xcoords += xwidth
            ycoords = y_pointer
            
        np.random.shuffle(self.seats)
        
        


class School(Model):

    
    def __init__(self, map_path, schedule_path, grade_N, KG_N, preschool_N, special_education_N, 
                 faculty_N, seat_dist, init_patient=3, attend_rate=1, mask_prob=0.516, inclass_lunch=False, username="jleiucsd"):
        # zipcode etc, for access of more realistic population from KG perhaps
        
        
        
        # model param init
        self.__mask_prob = mask_prob
        self.inclass_lunch = inclass_lunch
        self.seat_dist = math.ceil(seat_dist/(attend_rate**(1/2)))
        self.idle_teachers = [] # teachers to be assigned without a classroom
        self.init_patient = init_patient

        
        


        # mesa model init
        self.running = True
        self.grid = GeoSpace()
        self.schedule = BaseScheduler(self)
        
        
        
        #data collect init
        model_reporters = {"day": "day_count",
                           "cov_positive": "infected_count"}
        agent_reporters = {"unique_id": "unique_id",
                           "health_status": "health_status",
                           "symptoms": "symptoms",
                           "x": "x",
                           "y": "y",
                           "viral_load": "viral_load"}
        self.datacollector = datacollection.DataCollector(model_reporters=model_reporters, agent_reporters=agent_reporters)
        
        

        
        
        
        school_gdf = load_map(map_path)



        
        
        # room agent init
        self.room_agents = school_gdf.apply(lambda x: Classroom(
        unique_id = x["Id"], 
        model = self,
        shape = x["geometry"], 
        room_type = x["room_type"]),
                     axis=1
                    ).tolist()

        self.grid.add_agents(self.room_agents)
        
        
        
        # stats tracking init
        self.infected_count = 0
        self.step_count = 0
        self.day_count = 0
        self.num_exposed = 0
        
        
        
        # student activity init
        self.schoolday_schedule = pd.read_csv(schedule_path)
        self.activity = None
        
        
        # id tracking init
        self.__teacher_id = 0
        self.__student_id = 0
        self.__faculty_N = faculty_N
        self.schedule_ids = self.schoolday_schedule.columns
        
        
        
        self.recess_yards = find_room_type(self.room_agents, 'recess_yard')
        
        
        def init_agents(room_type, N, partition=False):
            '''
            batch initialize human agents into input room type rooms with equal partition size
            
            room_type: a valid string of room type: [None, 'restroom_grade_boys', 'lunch_room', 'classroom_grade',
               'restroom_all', 'restroom_grade_girls', 'restroom_KG',
               'classroom_KG', 'community_room', 'library',
               'restroom_special_education', 'restroom_faculty',
               'classroom_special_education', 'health_room', 'faculty_lounge',
               'classroom_preschool', 'restroom_preschool']
            '''

                
            rooms = find_room_type(self.room_agents, room_type)
            
            
            # if student group should be seperated to different day schedules
            # assigning schedule_id to equally partitioned rooms
            # currently only grade 1-5 "grade" students need to be partitioned, 
            partition_size = len(rooms)
            if partition:
                partition_size = math.ceil(partition_size/len(self.schedule_ids))
                
            class_size = N//len(rooms)
            remaining_size = N%len(rooms)

            for i, classroom in zip(range(len(rooms)), rooms):
                #Assigning a probability for circular rooms.
                prob_circular = np.random.choice([True, False], p=[0.5, 0.5])
                if prob_circular:
                    classroom.generate_seats(class_size, self.seat_dist, style='circular')
                    classroom.seating_pattern = 'circular'
                else:
                    classroom.generate_seats(class_size, self.seat_dist)
                    classroom.seating_pattern = 'individual'

                classroom.schedule_id = self.schedule_ids[i//partition_size]
                
                # Assigning index to base value of 0 for loop.
                idx = 0
                while (idx >= 0) and (idx <= class_size-1):
                    try:
                        pnt = classroom.seats[idx]
                    except IndexError:

                        # This case is when the num_children_per_desk is not enough
                        # to support the total number of kids.
                        
                        if prob_circular:
                            # Increment num_children until we find that all fits into the classroom.
                            num_children = classroom.num_children_per_desk
                            num_children += 1
                            print('Reassigning num_children now', num_children)
                            classroom.generate_seats(class_size, self.seat_dist, style='circular', num_children_per_desk=num_children)
                        
                        # Restart seat assignments
                        idx = 0
                        
                    
                    mask_on = np.random.choice([True, False], p=[mask_prob, 1-mask_prob])
                    agent_point = Student(model=self, shape=pnt, unique_id="S"+str(self.__student_id), room=classroom, mask_on=mask_on)
                    
                    if classroom.seating_pattern == 'circular':
                        desks = gpd.GeoSeries(classroom.desks)
                        agent_point.desk = desks[desks.distance(agent_point.shape).sort_values().index[0]]
                

                    self.grid.add_agents(agent_point)
                    self.schedule.add(agent_point)
                    self.__student_id += 1

                    idx += 1

                # spread remaining student into all classrooms
                if remaining_size > 0:
                        
                    pnt = classroom.seats[class_size-1]
                
                    mask_on = np.random.choice([True, False], p=[mask_prob, 1-mask_prob])
                    agent_point = Student(model=self, shape=pnt, unique_id="S"+str(self.__student_id), room=classroom, mask_on=mask_on)
                    
                    self.grid.add_agents(agent_point)
                    self.schedule.add(agent_point)
                    self.__student_id += 1
                    remaining_size -= 1


                #add teacher to class
                pnt = generate_random(classroom.shape)
                agent_point = Teacher(model=self, shape=pnt, unique_id="T"+str(self.__teacher_id), room=classroom)
                
              

                self.grid.add_agents(agent_point)
                self.schedule.add(agent_point)
                self.idle_teachers.append(agent_point)
                self.__teacher_id += 1
                self.__faculty_N -= 1


        
        # initialize all students and teachers in classrooms
        init_agents("classroom_grade", int(grade_N*attend_rate), partition=True)        
        # keep track of student types
        self.grade_students = [a for a in list(self.schedule.agents) if isinstance(a, Student)]        
        init_agents("classroom_KG", int(KG_N*attend_rate))
        init_agents("classroom_preschool", int(preschool_N*attend_rate))        
        #self.pkg_students = [a for a in list(set(self.schedule.agents).difference(self.grade_students)) if isinstance(a, Student)]
        init_agents("classroom_special_education", int(special_education_N*attend_rate))
  
            

        # dump remaining teacher to faculty lounge
        for f_lounge in find_room_type(self.room_agents, "faculty_lounge"):
            f_lounge.schedule_id = self.schedule_ids[0]
            
            for i in range(self.__faculty_N):

                pnt = generate_random(f_lounge.shape)
                agent_point = Teacher(model=self, shape=pnt, unique_id="T" + str(self.__teacher_id), room=f_lounge)
                self.grid.add_agents(agent_point)
                self.schedule.add(agent_point)
                self.__teacher_id += 1
        
        #self.people = list(self.schedule.agents)
        
        # add rooms to scheduler at last 
        for room in self.room_agents:
            self.schedule.add(room)
            
            
            
            
        self.lunchroom = find_room_type(self.room_agents, 'lunch_room')[0]
        self.lunchroom.generate_seats_lunch(3, 12)
            
            
            
     
    
    
    def small_step(self):
        self.schedule.step()
        self.grid._recreate_rtree() 
        
        
        
    def add_N_patient(self, N): 
        patients = random.sample([a for a in self.schedule.agents if isinstance(a, Student)], N)
        for p in patients:
            p.health_status = "exposed"
            p.asymptomatic = True
            p.infective = True
    
    
    def show(self):
        '''
        plot current step visualization
        deprecated since end of model visualization update
        '''
        
        # UPDATE 10/16: add deprecation warning
        message  = "this function is no longer used for performance issues, check output_image.py for end of model visualization"
        warnings.warn(message, DeprecationWarning)


        school_geometry = gpd.GeoSeries([a.shape for a in self.room_agents])
        school_map = gpd.GeoDataFrame({"viral_load" : [min(a.viral_load, 5) for a in self.room_agents]})
        school_map.geometry = school_geometry
        basemap = school_map.plot(column = "viral_load", cmap="Reds", alpha = 0.5, vmin = 0, vmax = 5)
        school_map.boundary.plot(ax = basemap, color='k', linewidth=0.2)

        list(map(lambda a: a.plot(), [a for a in self.schedule.agents if issubclass(type(a), Human)]))
        
        hour = 9 + self.step_count*5//60 # assume plot start at 9am
        minute = self.step_count*5%60
        plt.title("Iteration: Day {}, ".format(self.day_count + 1) + "%d:%02d" % (hour, minute), fontsize=30)

        
    
    def __update_day(self):

        '''
        update incubation time, reset viral_load, remove symptomatic agents, aerosol transmission etc for end of day
        '''

        for room in self.schedule.agents[:]:
            if issubclass(type(room), Classroom):
                mean_aerosol_transmissions = sum(room.aerosol_transmission_rate)
                if np.isnan(mean_aerosol_transmissions):
                    mean_aerosol_transmissions = 0

                occupants = [a for a in list(self.grid.get_intersecting_agents(room)) if issubclass(type(a), Human)]
                healthy_occupants = [a for a in occupants if a.health_status == 'healthy']
                mean_aerosol_transmissions = math.ceil(mean_aerosol_transmissions)

                to_expose = np.random.choice(healthy_occupants, size=mean_aerosol_transmissions)

                for student in to_expose:
                    student.health_status = 'exposed'
            
        for a in self.schedule.agents[:]:
            if issubclass(type(a), Human):
                    
                if a.symptoms:
                    # remove agent if symptom onset
                    if isinstance(a, Teacher):
                        # assign a new teacher to position
                        new_teacher = self.idle_teachers.pop()
                        new_teacher.shape = a.shape
                        new_teacher.room = a.room
                        new_teacher.classroom = a.classroom
                    self.schedule.remove(a)
                    self.grid.remove_agent(a)

                elif isinstance(a, Teacher):
                    if teacher_testing_frequency is not None:
                        if (self.day_count%teacher_testing_frequency == 0):
                            # assign a new teacher to position
                            if len(self.idle_teachers) > 0:
                                new_teacher = self.idle_teachers.pop()
                                new_teacher.shape = a.shape
                                new_teacher.room = a.room
                                new_teacher.classroom = a.classroom
                            self.schedule.remove(a)
                            self.grid.remove_agent(a)   
                
                elif isinstance(a, Student):
                    if student_testing_frequency is not None:
                        if (self.day_count %  student_testing_frequency== 0):
                            self.schedule.remove(a)
                            self.grid.remove_agent(a)            
                    
                # UPDATE 10/16: infectious made obsolete, end of day update rework
                elif a.health_status == "exposed":
                    # UPDATE 10/17: update infective delay if agent is not infective by end of day
                    a.infective = True
                    a.symptom_countdown -= 1
                    # calculate when symptoms begin to show using 0-15 density
                    if a.symptom_countdown <= 0:
                        if a.symptom_countdown == 0: 
                            self.infected_count += 1
                        # update model stat for total infected
                        # negative countdown means this agent is asymptomatic
                        
                        if not a.asymptomatic:
                            # this is a really small chance, however possible
                            # set symtoms to true
                            # next day this agent will be removed from the model
                            a.symptoms = True

            else:
                # reset viral_load of room agents
                a.viral_load = 0

        self.day_count += 1
        
        
    def step(self):
        '''
        simulate a day with school day schedule
        '''
        if not self.schedule.steps:
            self.add_N_patient(self.init_patient)
        
        
        
        for i, row in self.schoolday_schedule.iterrows():
            self.activity = row
            self.datacollector.collect(self)
            self.schedule.step()
            self.grid._recreate_rtree() 
            self.step_count += 1
            
            
        self.__update_day()  
        self.grid._recreate_rtree() 
        self.step_count = 0
