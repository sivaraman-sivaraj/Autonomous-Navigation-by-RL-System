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
    
memory = ReplayMemory(2016)

########################################
########## eps calculation #############
########################################
def eps_calculation(i_episode):
    """
    decaying epision value is an exploration parameter
    
    """
    
    start = 0.9
    end   = 0.1
    eps = end + (start- end) * math.exp(-1 * i_episode / 100)
    return eps

##########################################
########### action selection #############
##########################################

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
        return 0
    
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
    # Compute F1 smoothLoss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss
    

########################################
############ DQN Training ##############
########################################
def Train_DQN(N):
    """
    Parameters
    ----------
    N           : Number of Episodes

    Returns
    -------
    NoEpi                : Episode Duration for N episodes
    CUmulative_reward    : Cumulative reward for N episodes
    HEs                  : Average Heading Error for N episodes
    MSEs                 : Mean Square Error for N episodes
    """
    NoEpi             = [] # episode duration
    Cumulative_reward = [] # cumulative reward
    HEs               = [] # Average Heading error for an episode
    MSEs              = [] # Mean Square error for N episode
    for i_episode in range(N):
        total_reward = 0
        total_he     = 0 
        total_mse    = 0
        #plot_points = [[0,0]]
        eps   = eps_calculation(i_episode)
        if i_episode % 50 == 0:
            print("Episode: ",i_episode,"Running....")
        ##############################################
        #### Initialize the environment and state ####
        ##############################################
        ship_current_state,H = env.reset()
        state = ship_current_state
        
        for it in count():
            env.render()
            C = [select_action(state,eps),H]
            observation, reward, done, H = env.step(C) # Select and perform an action
            
            if it > 350:
                done = True
            
            if done == True:
                NoEpi.append(it+1)
                break
            
            next_state = observation                   # Observe new state
   
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
            loss = optimize_model()
            total_reward += reward
            total_he     += (abs(np.rad2deg(observation[5])) - theta)
            total_mse    += float(loss)
        env.close()
        HEs.append(total_he/it) # theta is global declaration
        Cumulative_reward.append(total_reward/it)
        MSEs.append(total_mse/it)
        ##############################################################################
        ####### Update the target network, copying all weights and biases in DQN #####
        ###############################################################################
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
    L1 = policy_net.ipL.weight.data
    LL1 = L1.numpy()
    np.save("layer1 weight",LL1)
    L2 = policy_net.HL1.weight.data
    LL2 = L2.numpy()
    np.save("layer2 weight",LL2)
    L3 = policy_net.HL2.weight.data
    LL3 = L3.numpy()
    np.save("layer3 weight",LL3)
    L4 = policy_net.opL.weight.data
    LL4 = L4.numpy()
    np.save("layer4 weight",LL4)
    ##### bias saving ################
    Lb1 = policy_net.ipL.bias.data
    LLb1 = Lb1.numpy()
    np.save("layer1 bias",LLb1)
    Lb2 = policy_net.HL1.bias.data
    LLb2 = Lb2.numpy()
    np.save("layer2 bias",LLb2)
    Lb3 = policy_net.HL2.bias.data
    LLb3 = Lb3.numpy()
    np.save("layer3 bias",LLb3)
    Lb4 = policy_net.opL.bias.data
    LLb4 = Lb4.numpy()
    np.save("layer4 bias",LLb4)
    
            
    return NoEpi, Cumulative_reward, HEs, MSEs
            
##########################################
########### Image Plotting ###############
##########################################

def plot_result1(NoEpi, Cumulative_reward):
    plt.figure(figsize=(9,12))
    #############################
    plt.subplot(2,1,1)
    N = len(Cumulative_reward)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(Cumulative_reward[max(0, t-100):(t+1)])
    plt.plot(running_avg1,color='r',label="Cumulative Reward")
    plt.title("DQN Training Resuts : Episode Durations & CUmulative Rewards ")
    plt.xlabel("No of Episode")
    plt.ylabel("Reward Unit")
    # plt.ylim(0,110)
    plt.legend(loc="best")
    plt.grid()
    ##############################
    plt.subplot(2,1,2)
    N = len(NoEpi)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(NoEpi[max(0, t-100):(t+1)])
    plt.plot(running_avg2,color='g', label = "Episode Durations")
    plt.xlabel("No of Episode")
    plt.ylabel("Length of Episodes")
    # plt.ylim(0,300)
    plt.legend(loc="best")
    plt.grid()
    plt.legend(loc="best")
    plt.show()
    
    
def plot_result2(HEs,MSE):
    plt.figure(figsize=(9,12))
    ##############################
    plt.subplot(2,1,1)
    N = len(MSE)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(MSE[max(0, t-100):(t+1)])
    plt.plot(running_avg1,color='m',label="Mean Square Error")
    plt.title("DQN Training Resuts : SmoothL1loss , Heading Error ")
    plt.xlabel("No of Episode")
    plt.ylabel("SmoothL1loss")
    # plt.ylim(0,110)
    plt.legend(loc="best")
    plt.grid()
    
    
    plt.subplot(2,1,2)
    N = len(HEs)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(HEs[max(0, t-10):(t+1)])
    plt.plot(running_avg2,color='b', label = "Cumulative Heading Error")
    plt.xlabel("No of Episode")
    plt.ylabel("Heading Error in degree")
    plt.legend(loc="best")
    plt.grid()
    # plt.ylim(0,120)
    plt.show()
    
###############################################
###############################################
########## Parameters Declaration #############
###############################################
####### Froude Scaling #########
scaled_u              = np.sqrt((7/320)*7.75*7.75)
GFL                   = 300 #Grid Full Length 
theta                 = 45
initial_velocity      = scaled_u # you can fix it as zero for convienient
###############################################
############ waypoints maker ##################
###############################################
prp,x_path,y_path,L = waypoints.straight_line(GFL,theta)
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
###### Initializing the Environment in Gym ########
###################################################
env                     = gym.make('maneuvere-v0')
device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ship_current_state      = torch.tensor([initial_velocity,0,0,0,0,0])

H                       = [0,0,False,0] #[Preceeding Quadrant, Preceeding WP,Preceeding action taken]    
###################################################
######## Initialing the Q - network ###############
###################################################
batch_size     = 128
gamma          = 0.99
target_update  = 5
done           = False
policy_net     = Q_network.MLFFNN()
target_net     = Q_network.MLFFNN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer      = optim.Adam(policy_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#optim.RMSprop()
##############################################
################ Training DQN ################
##############################################
N   = 5

start = time.time()
NoEpi, Cumulative_reward, HEs,MSEs = Train_DQN(N)
end = time.time()

print("Total time taken for training the DQN by " +str(N) +" episodes :    ", (end - start),'seconds')
###################################################
############# Plotting the result #################
###################################################
plot_result1(NoEpi, Cumulative_reward)
plot_result2(HEs,MSEs)
############################################
######### End of DQN Training ##############
############################################
