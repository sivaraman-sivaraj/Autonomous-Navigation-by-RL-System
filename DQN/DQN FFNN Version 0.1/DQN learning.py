import torch,torch.nn as nn#library importing
import torch.optim as optim,torch.nn.functional as F,torchvision.transforms as T
import gym,numpy as np,random,time,sys,matplotlib.pyplot as plt, math 
import  Q_network,waypoints,wp_anaysis # file importing
from gym.envs.registration import register
from collections import namedtuple, deque
from itertools import count


######################################
############## Memory ################
######################################
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    def all_sample(self):
        return self.memory
    
memory = ReplayMemory(10000)

###########################################
############ action selection #############
###########################################

def select_action(state,eps):
    
    sample = random.random()
    
    if sample > eps:
        with torch.no_grad():
            temp  = state.detach().tolist()
            op    = policy_net(torch.tensor(temp))
            return np.argmax(op.detach().numpy())
    else:
        return env.action_space_sample()

#########################################
############# optimizer #################
#########################################

def optimize_model():
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch       = Transition(*zip(*transitions))
    # print(batch)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)),
                                  device=device, dtype=torch.bool)
    
    non_final_next_states = torch.tensor(batch.next_state)
    state_batch  = torch.tensor(batch.state)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    action_batch = action_batch.unsqueeze(1)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values   = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)
    # Compute Huber loss
    # print(state_action_values.shape)
    # print(expected_state_action_values.shape)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
########################################
########## eps calculation #############
########################################

def eps_calculation(i_episode,num_episodes):
    """
    here, we are using decaying epision value for first half of total number of episode
    
    """
    steps = num_episodes*0.05
    start = 1
    end = 0.1
    if i_episode < steps :
        step_size= (start - end)/steps
        eps = 1 - (step_size*i_episode)
    else : 
        eps = end
    return eps


###############################################
########## Parameters Declaration #############
###############################################
####### Froude Scaling #########
scaled_u              = np.sqrt((7/320)*7.75*7.75)

GFL                   = 260
theta                 = 190
initial_velocity      = scaled_u # you can fix it as zero for convienient
###############################################
############ waypoints maker ##################
###############################################
prp,x_path,y_path,L = waypoints.straight_line(GFL,theta)
# prp,x_path,y_path,L = waypoints.spiral(40)
wpA                 = wp_anaysis.activate(prp)
print("The required starting and end point of the ship are ", prp[0],"&",prp[-1])
print("The Length of trajectory is :", L)
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
                                         'grid_size'   : int(GFL*2),
                                         'u'   : initial_velocity,
                                         'prp' : prp})
###################################################
###### Initializing the Grid Environment ##########
###################################################
env                     = gym.make('maneuvere-v0')
device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ship_current_state      = torch.tensor([initial_velocity,0,0,0,0,0,0])

H                       = [0,0,False,0] #[Preceeding Quadrant, Preceeding WP,Preceeding action taken]    
###################################################
###### Initialing the network #####################
###################################################
batch_size     = 108
gamma          = 0.99
target_update  = 10
done           = False
policy_net     = Q_network.MLFFNN()
target_net     = Q_network.MLFFNN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer  = optim.Adam(policy_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#optim.RMSprop()


##############################################
################ Training DQN ################
##############################################
start = time.time()
plot_points = [[0,0]]
num_episodes = 2
NoEpi = []
Cumulative_reward = []
for i_episode in range(num_episodes):
    ss = []
    eps = eps_calculation(i_episode,num_episodes)
    ##############################################
    #### Initialize the environment and state ####
    ##############################################
    ship_current_state,H = env.reset()
    
    state = ship_current_state
    if i_episode%1 ==0:
        print(i_episode)
    total_reward = 0
    for it in count():
        # Select and perform an action
        print(it)
        env.render(plot_points)
        C = [select_action(state,eps),H]
        observation, reward, done, H = env.step(C)
        
        print(H)
        
        if it > 450:
            done = True
        
        if done:
            # print("restart")
            NoEpi.append(it+1)
            break
        # Observe new state
    
        next_state = observation
        #######################################
        #### Store the transition in memory ###
        #######################################
        st_m   = state.tolist()
        n_st_m = next_state.tolist()
        r_m    = reward.tolist()
        memory.push(st_m, H[-1], n_st_m, r_m)
        #################################
        ###### Move to the next state ###
        #################################
        state = observation.clone().detach()
        #################################
        ######### optimization ##########
        #################################
        optimize_model()
        total_reward += reward
    env.close()
   
    if it == 0 :
        Cumulative_reward.append(0)
    else:
        Cumulative_reward.append(total_reward/it)
    ##############################################################################
    ####### Update the target network, copying all weights and biases in DQN #####
    ###############################################################################
    if i_episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    
# plt.figure(figsize=(9,6))
# N = len(Cumulative_reward)
# running_avg1 = np.empty(N)
# for t in range(N):
#         running_avg1[t] = np.mean(Cumulative_reward[max(0, t-100):(t+1)])
# plt.plot(running_avg1,color='g')
# plt.title("Cumulative Reward Plot for Training the Model")
# plt.xlabel("No of Episode")
# plt.ylabel("Reward Unit")
# # plt.grid()
# print('Complete')

##########################################
######### Saving the weight ##############
#########################################
# L1 = policy_net.ipL.weight.data
# LL1 = L1.numpy()
# np.save("layer1 weight",LL1)
# L2 = policy_net.HL1.weight.data
# LL2 = L2.numpy()
# np.save("layer2 weight",LL2)
# L3 = policy_net.HL2.weight.data
# LL3 = L3.numpy()
# np.save("layer3 weight",LL3)
# L4 = policy_net.opL.weight.data
# LL4 = L4.numpy()
# np.save("layer4 weight",LL4)
# ##### bias saving ################
# Lb1 = policy_net.ipL.bias.data
# LLb1 = Lb1.numpy()
# np.save("layer1 bias",LLb1)
# Lb2 = policy_net.HL1.bias.data
# LLb2 = Lb2.numpy()
# np.save("layer2 bias",LLb2)
# Lb3 = policy_net.HL2.bias.data
# LLb3 = Lb3.numpy()
# np.save("layer3 bias",LLb3)
# Lb4 = policy_net.opL.bias.data
# LLb4 = Lb4.numpy()
# np.save("layer4 bias",LLb4)
# end = time.time()



end = time.time()
print("Total time taken for 2500 episode", (end - start),'seconds')

