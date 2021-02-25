"""
Reinforcement learning for Ship Manoeuvering model

@author: Sivaraman Sivaraj, Suresh Rajendran
"""
print(__doc__)

import gym
import numpy as np,time
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

ship_current_state =[10,0,int(grid_size/2),int(land+ green_water),0,0,120] #[u,r,x,y,psi,delta,n] 

# to check
ship_next_state = nomoto_dof_1.activate(ship_current_state,0.1,0.05) 
print(ship_next_state) #[u,r,x,y,psi,delta,n]
###################################################
###### Algorithm  for Heading Action ##############
###################################################
end = False
env.reset()

while not end:
    env.render(plot_points)
    if count == 0:
        time.sleep(2)
    count += 1
    print(count)
    x_tm,y_tm = trajectory_points[-1][0],trajectory_points[-1][1]
    r = M[x_tm+1][y_tm]
    u = M[x_tm][y_tm+1]
    l = M[x_tm-1][y_tm]
    action = np.argmax([r,u,l]) #choosing the best possible action 
    observation, reward,done,_ = env.step(action)
    print(observation,reward,done,_)
    x_temp = round((observation[0])/sr,2)
    y_temp = round((observation[1])/sr,2)
    plot_points.append((x_temp, y_temp,0))
    trajectory_points.append((observation[0],observation[1],0))
    if observation[1] == prp[-1][1]:
        end = True

time.sleep(2)
env.close()
