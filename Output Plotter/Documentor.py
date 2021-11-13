###############################################
############# Libraries Import ################
###############################################
import torch,torch.nn as nn, torch.optim as optim
import gym,numpy as np,random,time,math,os,sys,pathlib
from gym.envs.registration import register
from collections import namedtuple, deque
from itertools import count
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt 
from matplotlib import animation
###############################################
############### GIF Function ##################
###############################################
def Create_GIF(frames,filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save( os.path.join(os.getcwd(),filename), writer='imagemagick', fps=60)

###############################################
########### Environment Import ################
###############################################
Environmet_Folder = pathlib.Path("Environment")
sys.path.insert(1,os.path.join(os.getcwd(),Environmet_Folder))
import  Q_network,waypoints,wp_analysis

###############################################
########## Parameters Declaration #############
###############################################
start                 = time.time()
scaled_u              = np.sqrt((7/320)*7.75*7.75)   # Froude Scaling
rps                   = 12.01                        # revolution per second
initial_velocity      = scaled_u                     # you can fix it as zero for convienient
W_Flag                = True
Head_Wave_Angles      = [-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4, np.pi/4,np.pi/2,3*np.pi/4,np.pi]
Lbl                   = ['-$\pi$','-3$\pi$/4','-$\pi$/2','-$\pi$/4','$\pi$/4','$\pi$/2','3$\pi$/4','$\pi$']

###############################################
############ waypoints maker ##################
###############################################
wp,x_path,y_path,L       = waypoints.straight_line(220,180)
# wp   = wp_[0:len(wp_)-2]
# wp,x_path,y_path,L       = waypoints.spiral(50)
# wp,x_path,y_path,L       = waypoints.Fibbanaci_Trajectory(15)
# wp,x_path,y_path,L       = waypoints.cardioid(25)
# wp.append([-25,-10])
# wp,x_path,y_path,L       = waypoints.parametric(30)
# wp,x_path,y_path,L       = waypoints.Arc_spline()
# print(wp[-1])
wpA                        = wp_analysis.activate(wp)
steps                      = wpA[2]
print("Approximate Event Horizon Steps     : <<",steps,">>")
##################################################
##### Registering the Environment in Gym #########
##################################################
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
      if 'NavigationRL-v0' in env:
          del gym.envs.registration.registry.env_specs[env]

register(id ='NavigationRL-v0', entry_point='Environment.Env:load',kwargs={'wpA' : wpA,
                                                                           'u': initial_velocity,
                                                                           'wp': wp,'rps' : rps,
                                                                           'W_Flag': W_Flag})


###################################################
################## Evaluation #####################
###################################################
Sts = [steps+30,steps+30,steps+30,steps+35,
       steps+35,steps+40,steps+30,steps+30] #sigmoid steps-239,steps-130,steps-230,steps-250,steps-240,steps-130,steps-130

# Sts = np.array([steps]*8)-70
# Sts   = [steps]*8 
# Sts[-1] += 130
# Sts = [290,195,192,180,205,295,250,285] --- -45
# Sts   = [290,195,192,180,190,195,210,285]--- 45
# Sts    = [150,155,145,220,140,150,150,150] 
Sts    = [270,270,290,370,
          390,290,280,260]
env                     = gym.make('NavigationRL-v0')
ship_current_state,H    = env.reset()
D                       = {'0':-35,'1':-20,'2':0,'3':20,'4':35}
net                     = Q_network.FFNN()
net.load_state_dict(torch.load("W1000.pt"))

DPs       = []
HError    = []
RD        = []
YR        = []
for i in range(len(Head_Wave_Angles)):
    WHA                  = Head_Wave_Angles[i]
    ship_current_state,H = env.reset()
    state                = ship_current_state
    Data                 = [ship_current_state]
    X,Y                  = [state[3]],[state[4]]
    for i_episode in range(Sts[i]):
        ip                                    = torch.tensor(Data[-1].tolist())
        action                                = np.argmax(net(ip).detach().numpy())
        C                                     = [action,H,WHA]
        observation, [reward,HE], done, H     = env.step(C) # Select and perform an action
        # if observation[3] > -250 and observation[4]> -245:
        X.append(observation[3])
        Y.append(observation[4])
        Data.append(observation)
        if i == len(Head_Wave_Angles)-1:
            YR.append(observation[2])
            HError.append(HE)
            RD.append(D[str(action)])
    DPs.append([X,Y])
#### radian to deg #########
HE_  = np.array(HError)
HError   = np.rad2deg(HE_)

YR_  = np.array(YR)
YR   = np.rad2deg(YR_)
####################################
###### Result Plotting #############
####################################
sctr_x,sctr_y = [],[]
for i in range(len(wp)):
    sctr_x.append(wp[i][0])
    sctr_y.append(wp[i][1])

####################################
crs = ['springgreen','olive','crimson','teal','orchid','burlywood','deeppink','royalblue','orangered']

plt.figure(figsize=(9,6))
plt.plot(x_path,y_path,'k',alpha=0.6,linestyle='dashdot',label="Target Path")
plt.scatter(sctr_x[0:len(sctr_x)-1],sctr_y[0:len(sctr_x)-1],marker = '8',color="purple",label = "waypoints")

for i in range(len(DPs)):
    plt.plot(DPs[i][0],DPs[i][1],color=crs[i],label= Lbl[i])


plt.title("180$^{\circ}$ Heading with Different 3.2-meter Head Waves") #  Navigation by DQN $\longrightarrow$ 

plt.xlabel("Advance  ($meters$)")
plt.ylabel("Transfer  ($meters$)")
plt.ylim(-50,50)
# plt.xlim(-100,100)
plt.axhline(y=0,color="grey",alpha=0.6)
plt.axvline(x=0,color="grey",alpha=0.6)
plt.legend(loc="best")
plt.grid()
plt.savefig("pic 1.jpg",dpi=480)
plt.show()
####################################
####################################

###################################
##### Result Plotting4 ############
###################################
plt.figure(figsize=(8,6))
plt.subplot(311)
plt.plot(HError,"darkcyan",label="Heading Error")
plt.title( "DQN : Heading Error, Rudder Deflection and Yaw Rate")
plt.xlabel("Time $(seconds)$")
plt.legend(loc="best")
plt.ylabel("Degree")
# plt.ylim(-50,50)
plt.grid()

plt.subplot(312)   
plt.plot(RD,"crimson",label="Rudder Deflection")
plt.xlabel("Time  $(seconds)$")
plt.ylabel("Degree")
plt.ylim(-40,40)
plt.legend(loc="best")
plt.grid()

plt.subplot(313)   
plt.plot(YR,"seagreen",label="Yaw Rate")
plt.xlabel("Time  $(seconds)$")
plt.ylabel("Deg / sec")
plt.ylim(-8,8)
plt.legend(loc="best")
plt.grid()

plt.savefig("pic 4.jpg",dpi=480)
####################################
############## End #################
####################################
 