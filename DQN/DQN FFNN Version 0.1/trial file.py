import gym
import numpy as np,time,sys,matplotlib.pyplot as plt
from gym.envs.registration import register
from matplotlib import cm
import torch
import Q_network,waypoints,wp_anaysis

###############################################
########## Parameters Declaration #############
###############################################
GFL                = 300 # global frame limit
land               = 10
green_water        = 10
theta              = 30
initial_velocity   = 0.175
###############################################
########## Q learnng Parameter ################
###############################################
gamma = 0.99
alpha = 0.15
###############################################
########## Calling Functions ##################
###############################################
# prpt,x_path,y_path,L   = waypoints.straight_line(GFL,theta)
prpt,x_path,y_path,L   = waypoints.spiral(50)
wpA                    = wp_anaysis.activate(prpt)
print("The required starting and end point of the ship are ", prpt[0],"&",prpt[-1])
##################################################
##### Registering the Environment in Gym #########
##################################################
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
      if 'maneuvere-v0' in env:
          print('Removed {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]

print("Renewal of Environment id : {} has done".format("maneuvere-v0"))
register(id ='maneuvere-v0', entry_point='Environment.Basin:Surrounding',
         max_episode_steps=10000,kwargs={'wpA' : wpA,
                                         'grid_size' : GFL,
                                         'land' : land,
                                         'green_water' : green_water,
                                         'u' : initial_velocity,
                                         'prp': prpt})

###################################################
###### Initializing the Environment ##########
###################################################
env = gym.make('maneuvere-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ship_current_state = torch.tensor([initial_velocity,0,0,0,0,0,0]) #[u,v,r,x,y,psi,t] 
###################################################
###### Algorithm  for Heading Action ##############
###################################################
end = False
env.reset()
trained_net = Q_network.RNN()

count           = 0
plot_points     = [] 
##################################
######### Result #################
##################################
sr = GFL/600
H  = [0,0] 
##########################################
######## Own Command #####################
##########################################
while not end:
    env.render(plot_points)
    if count == 0:
        time.sleep(5)
    count += 1
    print(count)
    
    if count < 50:
        action = 22
    elif 50 <= count < 184 :
        action = 11
    else:
        action = 5
    C = [action,H]
    observation, reward,done,_ = env.step(C)
    # print(observation, reward,done,_)
    ########## update ############
    x_temp = float((observation[4]+(GFL/2))/sr)
    y_temp = float((observation[3]+(GFL/2))/sr)
    plot_points.append((x_temp, y_temp))
    ship_current_state = observation
    if count == 650:
        end = True

time.sleep(1)
env.close()


