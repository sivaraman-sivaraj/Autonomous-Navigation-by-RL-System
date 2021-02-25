"""
Created on Fri Jan  1 18:25:51 2021

@author: Sivaraman Sivaraj, Suresh Rajendran
"""

import numpy as np
from gym import Env
from gym.utils import seeding
import nomoto_dof_1

colors ={"water" : [0, 0, 0.54],
         "land" : [0.05, 0.02, 0],
         "green_water":[0, 0.39, 0.39],
         "ship1":[0.7, 0.7, 0.7],
         "ship2": [0,0.48,0],
         "ship3": [0.89, 0.82, 0.28],
         "ship_engine":[0.8,0,0],
         "trajectory" : [0.74,0.74,0.74],
         "required_path":[1,1,0.2],
         "circle" : [0,0,0.16]}


######################################
#### Mariner's Ship polygan point ####
######################################
# import marinerp #for rendring ship # if we need to import

def import_polygon(sr):
    """
    Mariner's ship dimension
    
    Length between perpendiculr         : 320  m (scaled by ratio)
    Maximum Beam                        : 58  m # 5.7925 m(scaled)
    Returns
    -------
    polygon points

    """
    pp = list() #Polygon points
    ########
    ########
    pp.append((29,0))
    pp.append((29,100))
    ########
    ########
    pp.append((25,115))#curve points
    pp.append((20,125))
    pp.append((15,135))
    pp.append((10,145))
    pp.append((3,158))
    ########
    ########
    pp.append((0,160))
    ########
    ########
    pp.append((-3,158))#curve points
    pp.append((-10,145))
    pp.append((-15,135))
    pp.append((-20,125))
    pp.append((-25,115))
    ########
    ########
    pp.append((-29,100))
    pp.append((-29,0))
    ########
    ########
    pp.append((-29,-80))
    pp.append((-35,-80))
    pp.append((-35,-100))
    ########
    ########
    pp.append((-25,-100))
    pp.append((-25,-110))
    pp.append((-15,-110))
    pp.append((-15,-120))
    ########
    ########
    pp.append((15,-120))
    pp.append((15,-110))
    pp.append((25,-110))
    pp.append((25,-100))
    ########
    ########
    pp.append((35,-100))
    pp.append((35,-80))
    pp.append((29,-80))
    ########
    ########
    pp.append((29,0))
    pp.append((29,0))
    
    pp_temp = np.array(pp)
    pp_rt = pp_temp/sr
    pp_rt.tolist()
    return pp_rt



def import_polygon2(sr):
    pp = list() #Polygon points
    ########
    ########
    pp.append((25,-110))
    pp.append((25,-50))
    pp.append((-25,-50))
    pp.append((-25,-110))
    #######
    pp_temp = np.array(pp)
    pp_rt = pp_temp/sr
    pp_rt.tolist()
    return pp_rt
    
def import_polygon3(sr):
    pp = list() #Polygon points
    ########
    ########
    pp.append((25,-50))
    pp.append((25,100))
    pp.append((-25,100))
    pp.append((-25,-50))
    #######
    pp_temp = np.array(pp)
    pp_rt = pp_temp/sr
    pp_rt.tolist()
    return pp_rt

def import_polygon4(sr):
    pp = list() #Polygon points
    ########
    ########
    pp.append((25,100))
    pp.append((15,120))
    pp.append((5,130))
    pp.append((0,135))
    pp.append((-5,130))
    pp.append((-15,120))
    pp.append((-25,100))
    #######
    pp_temp = np.array(pp)
    pp_rt = pp_temp/sr
    pp_rt.tolist()
    return pp_rt
    
#########################################
###### to check outer plot ##############
#########################################
##print(import_polygon(1))
# sr = 1
# A = import_polygon(sr)
# B = import_polygon2(sr)
# C = import_polygon3(sr)
# D = import_polygon4(sr)
# x,y = list(),list()
# x1,y1 = list(),list()
# x2,y2 = list(),list()
# x3,y3 = list(),list()
# for i in range(len(A)):
#     x.append(A[i][0])
#     y.append(A[i][1])
# for i in range(len(B)):
#     x1.append(B[i][0])
#     y1.append(B[i][1])
# for i in range(len(C)):
#     x2.append(C[i][0])
#     y2.append(C[i][1])
# for i in range(len(D)):
#     x3.append(D[i][0])
#     y3.append(D[i][1])
    
# import matplotlib.pyplot as plt
# # import matplotlib.pyplot.FilledPolygon as ply
# plt.figure(figsize=(12,9))
# plt.plot(x,y,color='k',label="Container ship outer bound")
# plt.plot(x1,y1,color='k')
# plt.plot(x2,y2,color='k')
# plt.plot(x3,y3,color='k')
# plt.fill(x,y)
# plt.fill(x1,y1)
# plt.fill(x2,y2)
# plt.fill(x3,y3)
# plt.grid()
# plt.xlim(-200,200)
# plt.ylim(-200,200)
# plt.legend(loc="best")
# plt.title("Ship's Top View in( "+str(sr)+" ) sclaed ratio")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# plt.show()
######################################
###### Environment Construction ######
######################################



class Surrounding(Env): #basinGrid-s0
     metadata = {'render.modes': ['human', 'rgb_array'],
                 'video.frames_per_second': 180 }
     
     def __init__(self,R,prp,grid_size,land,green_water,u,t_i,wave= False):
         
        '''
        The initialization of the grid-world parameters.

        Parameters
        ----------
        R                   : Reward Matrix
        prp                 : path reward points
        grid_size           : totol grid size
        land, green_water   : width of land and green water respectively
        u                   : intial velocity of ship
        t_i                 : time interval
        Returns
        -------
        environment 

        '''
        self.prp = prp
        self.R = R
        self.land = land
        self.green_water = green_water
        self.grid_size = grid_size
        self.u = u
        self.t_i = t_i
        self.rpm = int((60*self.u)/(np.pi*9.58)) # propeller diameter as 9.58
        self.rewards = self.R
        self.t = self.prp[-1] # last element in prp is terminal point
        self.end = self.t
        self.goals = np.array(self.t) # end point of ship
        self.start = np.array([(int(grid_size/2),land+green_water)]) # starting point of ship 
        self.done = False #see here
        self.viewer = None
        self.st_x = int(grid_size/2)
        self.st_y = land+green_water
        self.actions_set = {'-7':-35,'-6':-30,'-5':-25,'-4':-20,'-3':-15,'-2':-10,'-1':-5,'0':0,
                       '7':35,'6':30,'5':25,'4':20,'3':15,'2':10, '1':5}
        
        ###############################
        ############ Seed #############
        ###############################
        t = prp[-1] # last element in prp is terminal point
        self.end = t
        self.seed()
        
        
     def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
     def reset(self):
        self.current_state = [self.u,0,self.st_x,self.st_y,0,0,self.rpm]
        self.done = False
        return self.current_state
    
     def step(self, action):
        self.next_state = np.copy(self.current_state)
        self.done = False#see here
        self._ = 'in water'
        
        self.next_state = nomoto_dof_1.activate(self.current_state,self.t_i,
                                                nomoto_dof_1.degree_to_rad(self.actions_set[str(action)]))
        self.int_x = int(np.ceil(self.next_state[2]))
        self.int_y = int(np.ceil(self.next_state[3]))
        self.reward_a = self.rewards[self.int_x][self.int_y]
        
        self.current_state = np.copy(self.next_state)
        '''
        done argument
        '''
        if self.int_x == self.goals[0] and self.int_y == self.goals[1] :
            self.done = True
        if self.reward_a == (-100):
            self._ = 'Not in water'
        if self.reward_a == 100:
            self._ = 'In correct path and water'
        return self.current_state, self.reward_a, self.done,self._

     def action_space_sample(self):
        n = np.random.randint(-7,8)
        return n
    
     def action_space(self):
        return np.arange(-7,8,1)
     
     def render(self,tp,mode='human'):
         """
         Note : We take grid size of (4000,4000). which should be scaled to {1:4} as (800,800)
         Parameters
         ----------
         mode : The default is 'human' : .
         H_A : Heading_Angle : which is for ship rendering
         tp      : Trajectory Points
         
         ----------------
         description
         ----------------
         Scale : (grid_size/600) = scaling ratio
         
         """
         screen_width = 600
         screen_height = 600
         self.sr = round((self.grid_size/600),2)
         self.tp = tp
         self.rtn = tp[-1][2]# to set the heading angle in ship
         if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
                
            ###################################################
            ############## Water Surface ######################
            ###################################################
            #water surface
            # water_surface = rendering.FilledPolygon([(land+green_water,land+green_water), (2850,350), (2850,2850), (350,2850)]) #actual size
            water_surface = rendering.FilledPolygon([((self.land+self.green_water)/self.sr,(self.land+self.green_water)/self.sr),
                                                     ((self.grid_size-self.land-self.green_water)/self.sr,(self.land+self.green_water)/self.sr),
                                                     ((self.grid_size-self.land-self.green_water)/self.sr,(self.grid_size-self.land-self.green_water)/self.sr),
                                                     ((self.land+self.green_water)/self.sr,(self.grid_size-self.land-self.green_water)/self.sr)])
            self.wstrans = rendering.Transform() 
            water_surface.set_color(colors["water"][0],colors["water"][1],colors["water"][2])
            water_surface.add_attr(self.wstrans)
            self.viewer.add_geom(water_surface)
            
            
                
            #Green water surface_bottom
            # green_water_surface_bottom = rendering.FilledPolygon([(100,100), (3100,100), (3100,350), (100,350)]) #actual size
            green_water_surface_bottom = rendering.FilledPolygon([(self.land/self.sr,self.land/self.sr),
                                                                  ((self.grid_size - self.land)/self.sr,self.land/self.sr),
                                                                  ((self.grid_size - self.land)/self.sr,(self.land+self.green_water)/self.sr),
                                                                  (self.land/self.sr,(self.land+self.green_water)/self.sr)])
            self.gwstrans_b = rendering.Transform() 
            green_water_surface_bottom.set_color(colors["green_water"][0],colors["green_water"][1],colors["green_water"][2])
            green_water_surface_bottom.add_attr(self.gwstrans_b)
            self.viewer.add_geom(green_water_surface_bottom)
                
            #Green water surface_top
            # green_water_surface_top = rendering.FilledPolygon([(100,2850), (3100,2850), (3100,3100), (100,3100)]) #actual size
            green_water_surface_top = rendering.FilledPolygon([(self.land/self.sr,(self.grid_size-self.land-self.green_water)/self.sr),
                                                               ((self.grid_size - self.land)/self.sr,(self.grid_size-self.land-self.green_water)/self.sr),
                                                               ((self.grid_size - self.land)/self.sr,(self.grid_size - self.land)/self.sr),
                                                               (self.land/self.sr,(self.grid_size - self.land)/self.sr)])
            self.gwstrans_t = rendering.Transform() 
            green_water_surface_top.set_color(colors["green_water"][0],colors["green_water"][1],colors["green_water"][2])
            green_water_surface_top.add_attr(self.gwstrans_t)
            self.viewer.add_geom(green_water_surface_top)
        
            
            ###################################################
            ############### Green Water  ######################
            ###################################################
            #Green water surface_left
            # green_water_surface_left = rendering.FilledPolygon([(100,350), (350,350), (350,2850), (100,2850)]) # actual size
            green_water_surface_left = rendering.FilledPolygon([(self.land/self.sr,(self.land+self.green_water)/self.sr),
                                                                ((self.land+self.green_water)/self.sr,(self.land+self.green_water)/self.sr),
                                                                ((self.land+self.green_water)/self.sr,(self.grid_size-self.land-self.green_water)/self.sr),
                                                                (self.land/self.sr,(self.grid_size-self.land-self.green_water)/self.sr)])
            self.gwstrans_l = rendering.Transform() 
            green_water_surface_left.set_color(colors["green_water"][0],colors["green_water"][1],colors["green_water"][2])
            green_water_surface_left.add_attr(self.gwstrans_l)
            self.viewer.add_geom(green_water_surface_left)
            
            
            
            #Green water surface_right
            # green_water_surface_right = rendering.FilledPolygon([(2850,350), (3100,350), (3100,2850), (2850,2850)]) # actual size
            green_water_surface_right = rendering.FilledPolygon([((self.grid_size-self.land-self.green_water)/self.sr,(self.land+self.green_water)/self.sr),
                                                                 ((self.grid_size - self.land)/self.sr,(self.land+self.green_water)/self.sr),
                                                                 ((self.grid_size - self.land)/self.sr,(self.grid_size-self.land-self.green_water)/self.sr),
                                                                 ((self.grid_size-self.land-self.green_water)/self.sr,(self.grid_size-self.land-self.green_water)/self.sr)])
            self.gwstrans_r = rendering.Transform() 
            green_water_surface_right.set_color(colors["green_water"][0],colors["green_water"][1],colors["green_water"][2])
            green_water_surface_right.add_attr(self.gwstrans_r)
            self.viewer.add_geom(green_water_surface_right)
                
            ###################################################
            ############### Land Surface ######################
            ###################################################
            #left land rendering
            # left_land = rendering.FilledPolygon([(0,0), (100,0), (100,3200), (0,3200)]) #actual size
            left_land = rendering.FilledPolygon([(0,0),
                                                 (self.land/self.sr,0),
                                                 (self.land/self.sr,600),
                                                 (0,600)])
            self.lltrans = rendering.Transform()#left land transform
            left_land.set_color(colors["land"][0],colors["land"][1],colors["land"][2])
            left_land.add_attr(self.lltrans)
            self.viewer.add_geom(left_land)
                
            #right land rendering
            # right_land = rendering.FilledPolygon([(3100,0), (3200,0), (3200,3200), (3100,3200)]) #actual size
            right_land = rendering.FilledPolygon([((self.grid_size - self.land)/self.sr,0),
                                                  (600,0),
                                                  (600,600),
                                                  ((self.grid_size - self.land)/self.sr,600)])
            self.rltrans = rendering.Transform()#right land transform
            right_land.set_color(colors["land"][0],colors["land"][1],colors["land"][2])
            right_land.add_attr(self.rltrans)
            self.viewer.add_geom(right_land)
            
            #top land rendering
            # top_land = rendering.FilledPolygon([(100,3100), (3100,3100), (3100,3200), (100,3200)]) #actual size
            top_land = rendering.FilledPolygon([(self.land/self.sr,(self.grid_size - self.land)/self.sr),
                                                ((self.grid_size - self.land)/self.sr,(self.grid_size - self.land)/self.sr),
                                                ((self.grid_size - self.land)/self.sr,600),
                                                (self.land/self.sr,600)])
            self.tptrans = rendering.Transform()#top land transform
            top_land.set_color(colors["land"][0],colors["land"][1],colors["land"][2])
            top_land.add_attr(self.tptrans)
            self.viewer.add_geom(top_land)
            
            #bottom land rendering
            # bottom_land = rendering.FilledPolygon([(100,0), (3100,0), (3100,100), (100,100)])#actual size
            bottom_land = rendering.FilledPolygon([(self.land/self.sr,0),
                                                   ((self.grid_size - self.land)/self.sr,0),
                                                   ((self.grid_size - self.land)/self.sr,self.land/self.sr),
                                                   (self.land/self.sr,self.land/self.sr)])
            self.bttrans = rendering.Transform()#right land transform
            bottom_land.set_color(colors["land"][0],colors["land"][1],colors["land"][2])
            bottom_land.add_attr(self.bttrans)
            self.viewer.add_geom(bottom_land)
            
            ###################################################
            ############### Ship Rendering ####################
            ###################################################            
            ship = rendering.FilledPolygon(import_polygon(self.sr*3))
            ship.set_color(colors["ship1"][0],colors["ship1"][1],colors["ship1"][2])
            self.shiptrans = rendering.Transform(translation=(0,0),rotation= self.rtn)#
            ship.add_attr(self.shiptrans)
            
            ship1 = rendering.FilledPolygon(import_polygon2(self.sr*3))
            ship1.set_color(colors["ship2"][0],colors["ship2"][1],colors["ship2"][2])
            ship1.add_attr(self.shiptrans)
            
            ship2 = rendering.FilledPolygon(import_polygon3(self.sr*3))
            ship2.set_color(colors["ship_engine"][0],colors["ship_engine"][1],colors["ship_engine"][2])
            ship2.add_attr(self.shiptrans)
            
            ship3 = rendering.FilledPolygon(import_polygon4(self.sr*3))
            ship3.set_color(colors["ship3"][0],colors["ship3"][1],colors["ship3"][2])
            ship3.add_attr(self.shiptrans)
            
            self.viewer.add_geom(ship)
            self.viewer.add_geom(ship1)
            self.viewer.add_geom(ship2)
            self.viewer.add_geom(ship3)
            self.axle = rendering.make_circle(3) #will chnage to 3.8
            self.axle.add_attr(self.shiptrans)
            self.axle.set_color(colors["circle"][0],colors["circle"][1],colors["circle"][2])
            self.viewer.add_geom(self.axle)
            
            self._ship_geom = ship
            
            ###################################################
            ############### Path Rendering ####################
            ###################################################
            end_point = self.prp[-1]
            x_e = end_point[0]/self.sr #divided by 4 for scaling
            y_e = end_point[1]/self.sr
            self.path = rendering.Line((self.st_x/self.sr,self.st_y/self.sr), (x_e,y_e))
            self.path.set_color(colors["required_path"][0],colors["required_path"][1],colors["required_path"][2])
            self.viewer.add_geom(self.path)
            ###################################################
            ############### Trajectory Rendering ##############
            ###################################################
            self.traject = rendering.make_polyline(self.tp) # for every time step, position should be updated
            self.traject.set_color(colors["trajectory"][0],colors["trajectory"][1],colors["trajectory"][2])
            self.viewer.add_geom(self.traject)
            
            
         shipx = int(self.current_state[2]/self.sr)
         shipy = int(self.current_state[3]/self.sr)
         ship = self._ship_geom
                   
         self.shiptrans.set_translation(shipx,shipy)
         self.shiptrans.set_rotation(self.rtn) #ship rotation
        
         return self.viewer.render(return_rgb_array=mode == 'rgb_array')
     
        
     def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            
            
         
        
        
        
        
        
     

