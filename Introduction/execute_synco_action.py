"""
Reinforcement learning for Ship Manoeuvering model

@author: Sivaraman Sivaraj, Suresh Rajendran
"""
print(__doc__)

import gym
import numpy as np,time,sys
from gym import wrappers
from gym.envs.registration import register
import heading
import nomoto_dof_1
import matplotlib.pyplot as plt
import body_frame
###############################################
########## Parameters Declaration #############
###############################################
grid_size = 3200
land = 200 
green_water = 200
theta = 0
rendering_window_size = 600 #if you need different window size, please change in environment file as well
time_interval = 0.1
ws = grid_size - (2*land) - (2*green_water) # water surface length
###############################################
########## Calling Functions ##################
###############################################

M,prp,x,y = heading.activate(grid_size,green_water,land,theta)

print("The required starting and end point of the ship are ", prp[0],"&",prp[-1])

###################################################
###### Registering the Environment in Gym #########
###################################################
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
      if 'basinGrid-v0' in env:
          print('Removed {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]


print("Renewal of Environment id : {} has done".format("basinGrid-v0"))
register(id ='basinGrid-v0',
    entry_point='Environment.Basin:Surrounding',
    max_episode_steps=10000,
    kwargs={'R' : M, 'prp' : prp, 'grid_size' : grid_size, 'land' : land, 'green_water' : green_water},
    )

# plt.matshow(R)

###################################################
###### Initializing the Grid Environment ##########
###################################################
env = gym.make('basinGrid-v0')
count = 0
sr = grid_size/rendering_window_size
plot_points = [(int(grid_size/2)/sr,(land+ green_water)/sr,0)]
trajectory_points = [(int(grid_size/2),(land+ green_water),0)]
###################################################
###### Initializing the Ship's state ##############
###################################################

ship_current_state =[7.75,0,int(grid_size/2),int(land+ green_water),0,0,120] #[u,r,x,y,psi,delta,n] 

###################################################
###### Algorithm  for Heading Action ##############
###################################################
data = [ship_current_state]
end = False
env.reset()

while not end:
    env.render(plot_points)
    if count == 0:
        time.sleep(1)
    count += 1
    r_a_c = 0
    obs = (0,0,0)
    ########################
    ## selecting obs #######
    ########################
    x_r, y_r = data[-1][2],data[-1][3]
    # print(x_r,y_r)
    y_n = int(np.ceil(y_r)) + 1
    next_row_rewards = []
    for nr in range(ws):
        temp = M[land+green_water+nr][y_n]
        next_row_rewards.append(temp)
    x_max_arg = np.argmax(next_row_rewards)
    x_n = int(land+green_water+x_max_arg)
    obs = (x_n,y_n,0)
    ##############################
    ### updating current state ###
    ##############################
    r_a_c = body_frame.get(data[-1],obs)
    temp_state = nomoto_dof_1.activate(data[-1],0.1,r_a_c)#
    if (temp_state[3] - np.floor(y_r)) > 1:
        if temp_state[2]-x_r > 1:
            action = 0
            observation, reward,done,_ = env.step(action)
        elif -1< (temp_state[2]-x_r) < 1:
            action = 1
            observation, reward,done,_ = env.step(action)
        elif (temp_state[2]-x_r) < -1:
            action = 2
            observation, reward,done,_ = env.step(action)
    data.append(temp_state)
    print(temp_state)
    x_temp = round((temp_state[2])/sr,2)
    y_temp = round((temp_state[3])/sr,2)
    plot_points.append((x_temp, y_temp,0))
    trajectory_points.append((temp_state[2],temp_state[3],0))
    if np.ceil(data[-1][3]) == np.ceil(prp[-1][1]):
        end = True
    # #just to terminate
    if count == 100:
        end = True

time.sleep(2)
env.close()

# data finally in CSV file

