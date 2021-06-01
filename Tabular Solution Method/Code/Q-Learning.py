"""
##############################################################
###### Reinforcement learning for Ship Manoeuvering model ####
##############################################################
######### author: Sivaraman Sivaraj, Suresh Rajendran ########
##############################################################
"""
print(__doc__)

import gym,heading,nomoto_dof_1
import numpy as np,time,sys,matplotlib.pyplot as plt
from gym.envs.registration import register
from matplotlib import cm
###############################################
########## Parameters Declaration #############
###############################################
grid_size = 100
land = 5
green_water = 5
theta = 10
rendering_window_size = 600 #if you need different window size, please change in environment file as well
time_interval = 0.5
ws = grid_size - (2*land) - (2*green_water) # water surface length
initial_velocity = 7.75
###############################################
########## Q learnng Parameter ################
###############################################
max_states = ws*ws
gamma = 0.9
alpha = 0.1
###############################################
########## Calling Functions ##################
###############################################
M,prp,x,y = heading.activate(grid_size,green_water,land,theta)
print("The required starting and end point of the ship are ", prp[0],"&",prp[-1])
##################################################
##### Registering the Environment in Gym #########
##################################################
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
      if 'basinGrid-v0' in env:
          print('Removed {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]

print("Renewal of Environment id : {} has done".format("basinGrid-v0"))
register(id ='basinGrid-v0',
    entry_point='Environment.Basin:Surrounding',
    max_episode_steps=10000,
    kwargs={'R' : M, 'prp' : prp, 'grid_size' : grid_size, 'land' : land, 'green_water' : green_water,
            'u' : initial_velocity, 't_i' : time_interval},
    )
# plt.matshow(M)
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

ship_current_state =[5,0,int(grid_size/2),int(land+ green_water),0,0,120] #[u,r,x,y,psi,delta,n] 

###################################################
###### Algortham's Implementing Preparation #######
###################################################
def optimal_action(Qs_action):#maximum action value from dictionary at particular cell
    max_value = float('-inf')
    for act, val in Qs_action.items():
        if val > max_value:
            max_value = val
            opt_act = act
    return opt_act, max_value

def create_bins(grid_size): #to get ship's cell in grid
    bins = np.zeros((2,grid_size))
    bins[0] = np.arange(0,grid_size,1)
    bins[1] = np.arange(0,grid_size,1)
    return bins

def assign_bins(observation, bins): # cell position for finding reward value
    cell_x = np.digitize(observation[2],bins[0])
    cell_y = np.digitize(observation[3],bins[1])
    return [int(cell_x),int(cell_y)]

def digit_count(grid_size):# to declare the string length in dictionary
    count = 0
    while grid_size != 0:
        grid_size //= 10
        count += 1
    return count

def get_static_states(grid_size,land,green_water):
    ws = grid_size - (2*land) - (2*green_water) # water surface length
    ss = land + green_water #static surfce width
    states_ss = []
    for i1 in range(ss): # left land
        for j1 in range(grid_size):
            states_ss.append(str(i1).zfill(digit_count(grid_size))
                          +str(j1).zfill(digit_count(grid_size)))
    for i2 in range(ss):#right land
        for j2 in range(grid_size):
            states_ss.append(str(i2+ws+ss).zfill(digit_count(grid_size))
                          +str(j2).zfill(digit_count(grid_size)))
    for i3 in range(ws):#bottom land
        for j3 in range(ss):
            states_ss.append(str(i3+ss).zfill(digit_count(grid_size))
                          +str(j3).zfill(digit_count(grid_size)))
    for i4 in range(ws):#top land
        for j4 in range(ss):
            states_ss.append(str(i4+ss).zfill(digit_count(grid_size))
                          +str(j4+ss+ws).zfill(digit_count(grid_size)))
    return states_ss

def get_state_as_strings(state): # to store in the dictionary
    string_state = str(state[0]).zfill(digit_count(grid_size))+str(state[1]).zfill(digit_count(grid_size))
    return string_state

def get_all_states_strings(grid_size,land,green_water): #Dictionary containg all the state's actions
    ws = grid_size - (2*land) - (2*green_water)
    states =[]
    for i in range(ws):
        for j in range(ws):
            states.append(str(i+land+green_water).zfill(digit_count(grid_size))+str(j+land+green_water).zfill(digit_count(grid_size)))
    return states

def Q_table():
    Q ={}
    
    all_states = get_all_states_strings(grid_size,land,green_water)
    for state in all_states:
        Q[state] = {}
        for action in env.action_space():
            Q[state][action] = 0
    return Q
###################################################
############# One Episode & Q-Learning ############
###################################################

def play_one_epdisode(Q,Q_s,bins,eps = 0.4):
    observation = env.reset()
    count = 0
    done = False
    state = get_state_as_strings(assign_bins(observation, bins))
    total_reward = 0
    while not done:
        # env.render()
        count +=1
        if np.random.uniform() < eps:
            action = env.action_space_sample()#episilon greedy
        else:
            action = optimal_action(Q[state])[0]
            # print(action)
        observation, reward,done,_ = env.step(action)
        total_reward += reward
        
        if total_reward/count < 95:
            done = True
            
        state_next = get_state_as_strings(assign_bins(observation, bins))
        if state_next not in Q_s:
            opt_act, act_val = optimal_action(Q[state_next])
            Q[state][action] += alpha*(reward + gamma*act_val - Q[state][action])
            state, action = state_next, opt_act
        if state_next in Q_s:
            done = True
    return total_reward, count,Q

###################################################
################## Agent's Training ###############
###################################################
def training(bins,eps = 0.3, N = 100):
    Q = Q_table()
    Q_s = get_static_states(grid_size,land,green_water)
    Q_return = Q.copy()
    length,reward = [],[]
    for n in range(N):
        # if n == N-1:
        #     env.render()
        episode_reward, episode_length,Q_t = play_one_epdisode(Q,Q_s,bins,eps)
        Q_return = Q_t
        # env.close()
        length.append(episode_length)
        reward.append(episode_reward)
        # if n%100 ==0 :
            # print(n, '%4f', eps, episode_reward)
        if n%1000 == 0:
            print(n)#to know, which episode is running
    # env.close()
    return length, reward, Q_return

###################################################
############### Q-table Documentation  ############
###################################################
def Documentation(Q,grid_size):
    Q_output = np.zeros((grid_size,grid_size))
    value_matrix = np.zeros((grid_size,grid_size))
    for state in  Q:
        x,y = int(state[0:int(len(state)/2)]),int(state[int(len(state)/2):])
        state_action,state_val = optimal_action(Q[state])
        # print(state_action[0])
        Q_output[x][y] = int(state_action)
        value_matrix[x][y]= int(state_val)
    return Q_output,value_matrix

###################################################
################### Execution #####################
###################################################
def execute(eps):
    bins = create_bins(grid_size)
    episode_length, episode_reward, Q = training(bins,eps,N = 50000)
    # plot_running_avg(episode_reward)
    action_map,value_matrix = Documentation(Q, grid_size)
    return Q,action_map,value_matrix,episode_reward


Q1,action_map1,value_matrix1,eps_r1 = execute(0.05)
Q2,action_map2,value_matrix2,eps_r2 = execute(0.2)
Q3,action_map3,value_matrix3,eps_r3 = execute(0.35)
Q4,action_map4,value_matrix4,eps_r4 = execute(0.5)
