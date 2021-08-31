import torch,torch.nn as nn#library importing
import torch.optim as optim
import gym,numpy as np,random,time,math,os
import  Q_network,waypoints,wp_analysis # file importing
from gym.envs.registration import register
import graph
from collections import namedtuple, deque
from itertools import count
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
start = time.time()
###############################################
########## Parameters Declaration #############
###############################################
scaled_u              = np.sqrt((7/320)*7.75*7.75)   # Froude Scaling
theta                 = 90
initial_velocity      = scaled_u                     # you can fix it as zero for convienient
No_Episodes           = 5000
###############################################
############ waypoints maker ##################
###############################################
# wp,x_path,y_path,L       = waypoints.straight_line(250,theta)
# wp,x_path,y_path,L       = waypoints.spiral(50)
# wp,x_path,y_path,L       = waypoints.Fibbanaci_Trajectory(15)
# wp,x_path,y_path,L       = waypoints.cardioid(40) 
# wp,x_path,y_path,L       = waypoints.parametric(30)
wp,x_path,y_path,L         = waypoints.Arc_spline()

wpA                        = wp_analysis.activate(wp,No_Episodes)
print("The goal points             :",wpA[2][0])
##################################################
##### Registering the Environment in Gym #########
##################################################
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
      if 'NavigationRL-v0' in env:
          del gym.envs.registration.registry.env_specs[env]

register(id ='NavigationRL-v0', entry_point='Environment:load',kwargs={'wpA' : wpA, 'u': initial_velocity, 'wp': wp})

###################################################
################## DQN Training ###################
###################################################

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
    
memory = ReplayMemory(3000) # we can alter it for some different

########################################
########## eps calculation #############
########################################
def eps_calculation(i_episode):
    """
    decaying epision value is an exploration parameter
    
    """
    
    start = 0.999
    end   = 0.05
    eps = end + (start- end) * math.exp(-0.5 * i_episode / 100)
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
    
def reinforce_action(ip):
    temp = ip.detach().tolist()
    ip   = torch.tensor(temp)
    with torch.no_grad():
        op   = reinforce_net(ip)
    return np.argmax(op.detach().numpy()) 



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
    # Compute F1 smoothLoss or MSE loss()
    criterion = nn.MSELoss()#SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss
#########################################################
###### Initializing the Environment in Gym & DQN ########
#########################################################
env                     = gym.make('NavigationRL-v0')
device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
goals,steps,NoEpis      = wpA[2]
batch_size              = 128
gamma                   = 0.99
target_update           = 10
policy_net              = Q_network.FFNN()
target_net              = Q_network.FFNN()
reinforce_net           = Q_network.FFNN()
policy_net     = policy_net.to(device=device)
target_net     = target_net.to(device=device)
reinforce_net  = reinforce_net.to(device=device)

optimizer               = optim.Adam(policy_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#optim.RMSprop()
 
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
torch.save(policy_net.state_dict(),"W.pt")
reinforce_net.load_state_dict(torch.load("W.pt"))
reinforce_net.eval()
############################################
############ Main function #################
############################################   
R_steps                       = [0]
R_rewards                     = [0]
for ii in range(1,len(goals)):
    
    goal_N,steps_N,epi_N = goals[ii],steps[ii],NoEpis[ii] # monte carlo steps
    ##### Pre-phase training ######
    reinforce_index                   = R_steps[ii-1]
    ##### phase training ######
    NoEpi              = [] # episode duration
    Cumulative_reward  = [] # cumulative reward
    HEs                = [] # Average Heading error for an episode
    MSEs               = [] # Mean Square error for N episode
    
    print("Phase for goal-step-episode : (",goal_N,steps_N,epi_N,") training has started...!")
    for i_episode in range(epi_N):
        total_reward = 0
        total_HE     = 0 
        total_MSE    = 0
        #plot_points = [[0,0]]
        eps   = eps_calculation(i_episode)
        if i_episode % 200 == 0:
            print("Episode: ",i_episode,"Running....")
        ##############################################
        #### Initialize the environment and state ####
        ##############################################
        ship_current_state,H = env.reset()
        state                = ship_current_state
        LoE                  = 1                     # length of episode
        for it in count():
            # env.render()
            if it >= reinforce_index:
                action                       = select_action(state,eps)
                C                            = [action,H,LoE,goal_N]
                observation, [reward,HE], done, H     = env.step(C) # Select and perform an action
            
            if it <reinforce_index :
                action                       = reinforce_action(state)
                C                            = [action,H,LoE,goal_N]
                observation, [reward,HE], done, H     = env.step(C) # Select and perform an action
                reward,HE                             = R_rewards[-1],0.0
            
            LoE                                   += 1
            # print(reward)
            if it >= steps_N:
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
            r_m    = reward.item()
            
            memory.push(st_m, action, n_st_m, r_m)
            #################################
            ###### Move to the next state ###
            #################################
            state = observation.clone().detach()
            #################################
            ######### optimization ##########
            #################################
            loss = optimize_model()
            total_reward += reward.item()
            total_HE     += np.rad2deg(HE)
            total_MSE    += float(loss)
        
        # env.close()
        HEs.append(total_HE/LoE) # theta is global declaration
        Cumulative_reward.append(total_reward/LoE)
        MSEs.append(total_MSE/LoE)
        ##############################################################################
        ####### Update the target network, copying all weights and biases in DQN #####
        ###############################################################################
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if i_episode % 1000 == 0:
            torch.save(policy_net.state_dict(),"Phase_"+str(ii)+"_W_"+str(i_episode)+".pt") 
            
    torch.save(policy_net.state_dict(),"W.pt")  
    R_steps.append(np.mean(NoEpi[-2000:])) # averaging the last 2000 episodes for reinfore=ce steps
    R_rewards.append(np.mean(Cumulative_reward[-2000:]))
    ############################
    ######### plotting #########
    ############################
    
    graph.plot_result1(NoEpi, Cumulative_reward,ii)
    graph.plot_result2(HEs,MSEs,ii)




print("complete...!")
end = time.time()
print("Total time has taken for Training the process : ",(round((end-start)/60,1)), " minutes" )
##########################################
################## End ###################
##########################################





