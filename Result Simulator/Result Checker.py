###############################################
############# Libraries Import ################
###############################################
import torch,torch.nn as nn, torch.optim as optim,time
import gym,numpy as np,random,time,math,os,sys,pathlib
from gym.envs.registration import register
from collections import namedtuple, deque
from itertools import count
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt 
from matplotlib import animation
###############################################
########### Environment Import ################
###############################################
Environmet_Folder = pathlib.Path("Environment")
sys.path.insert(1,os.path.join(os.getcwd(),Environmet_Folder))
import  Q_network,waypoints,wp_analysis,graph,MMG,LOS
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
########## Parameters Declaration #############
###############################################
start                 = time.time()
scaled_u              = np.sqrt((7/320)*7.75*7.75)   # Froude Scaling
initial_velocity      = scaled_u                     # you can fix it as zero for convienient
###############################################
############ waypoints maker ##################
###############################################
# wp,x_path,y_path,L       = waypoints.straight_line(250,-135)
# wp,x_path,y_path,L       = waypoints.spiral(35)
wp,x_path,y_path,L       = waypoints.Fibbanaci_Trajectory(15)
# wp,x_path,y_path,L       = waypoints.cardioid(40) 
# wp,x_path,y_path,L       = waypoints.parametric(30)
# wp,x_path,y_path,L       = waypoints.Arc_spline()
# wp = wp[0:len(wp)-70]
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

register(id ='NavigationRL-v0', entry_point='Environment.Env:load',kwargs={'wpA' : wpA, 'u': initial_velocity, 'wp': wp})
###################################################
################## Evaluation #####################
###################################################
env                     = gym.make('NavigationRL-v0')
ship_current_state,H    = env.reset()
print(H)
D                       = {'0':-35,'1':-20,'2':0,'3':20,'4':35}
net                     = Q_network.FFNN()
net.load_state_dict(torch.load("W.pt"))


Data  = [ship_current_state]
X,Y   = [],[]
YR    = []
RD    = []
frames = []
for i in range(steps-50):
    frames.append(env.render(mode="rgb_array"))
    ip                                    = torch.tensor(Data[-1].tolist())
    action                                = np.argmax(net(ip).detach().numpy())
    C                                     = [action,H]
    observation, [reward,HE], done, H     = env.step(C)
    # print(H)
    Data.append(observation)
    X.append(observation[3])
    Y.append(observation[4])
    YR.append(observation[2])
    RD.append(D[str(action)])
env.close()
#### radian to deg #########
YR_  = np.array(YR)
YR   = np.rad2deg(YR_)

# Create_GIF(frames,filename='gym_animation.gif')
####################################
###### Result Plotting #############
####################################
plt.figure(figsize=(9,6))
plt.plot(X,Y,color='green',label="DQN Trained Path")
plt.plot(x_path,y_path,'y',alpha=0.8,linestyle='dashdot',label="Target Path")
plt.scatter(x_path[::14],y_path[::14],color="purple",)
plt.xlabel("Advance  ($meters$)")
plt.ylabel("Transfer  ($meters$)")
plt.grid()
# plt.ylim(-50,50)
# plt.xlim(-100,100)
plt.axhline(y=0,color="grey",alpha=0.6)
plt.axvline(x=0,color="grey",alpha=0.6)
plt.title("Navigation by DQN Algorithm $\longrightarrow$ Zero Degree Heading")
plt.legend(loc="best")
plt.savefig("pic 1.jpg",dpi=720)
plt.show()
###### second image #######
plt.figure(figsize=(9,6))
plt.subplot(211)
plt.plot(YR,"g")
plt.title("Trained Path's Yaw Rate and Rudder Deflection")
plt.xlabel("Time $(seconds)$")
plt.ylabel("Yaw Rate (deg/sec)")
plt.grid()
plt.subplot(212)   
plt.plot(RD,"r")
plt.xlabel("Time  $(seconds)$")
plt.ylabel("Rudder Deflection (in degrees)")
plt.grid()
plt.savefig("pic 4.jpg",dpi=720)
####################################
############## End #################
####################################

