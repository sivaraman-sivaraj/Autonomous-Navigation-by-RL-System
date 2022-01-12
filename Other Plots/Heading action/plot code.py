import numpy as np
import matplotlib

matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import waypoints

# s0x,s0y     = np.load("S0.npy")
s45x,s45y   = np.load("S45.npy")
s_45x,s_45y = np.load("S-45.npy")
s90x,s90y   = np.load("S90.npy")
s_90x,s_90y = np.load("S_90.npy")
s135x,s135y   = np.load("S135.npy")
s_135x,s_135y = np.load("S_135.npy")
s180x,s180y   = np.load("S180.npy")


print(len(s180x),len(s45x))

wp,x_0,y_0,L         = waypoints.straight_line(250,0)
wp,x_45,y_45,L       = waypoints.straight_line(250,45)
wp,x_45m,y_45m,L     = waypoints.straight_line(250,-45)
wp,x_90,y_90,L       = waypoints.straight_line(250,90)
wp,x_90m,y_90m,L         = waypoints.straight_line(250,-90)
wp,x_135,y_135,L       = waypoints.straight_line(250,135)
wp,x_135m,y_135m,L       = waypoints.straight_line(250,-135)
wp,x_180,y_180,L       = waypoints.straight_line(250,180)


plt.figure(figsize=(9,6))
###################################### Target Path
# plt.plot(x_0[::14],y_0[::14],color="cyan",marker="8",alpha=0.7,linestyle='dashed',label="0 \N{DEGREE SIGN} Heading")
plt.plot(x_45m[::14],y_45m[::14],alpha=0.7,linestyle='dashed',color="red",marker="8",label="-45 \N{DEGREE SIGN} Heading")
plt.plot(x_90m[::14],y_90m[::14],alpha=0.7,linestyle='dashed',color="b",marker="8",label="-90 \N{DEGREE SIGN} Heading")
plt.plot(x_135m[::14],y_135m[::14],alpha=0.7,linestyle='dashed',color="lime",marker="8",label="-135 \N{DEGREE SIGN} Heading")
plt.plot(x_180[::14],y_180[::14],alpha=0.7,linestyle='dashed',color="purple",marker="8",label="-180 \N{DEGREE SIGN} Heading")


# ###################################### DQN Path
# plt.plot(s45x[:320],s45y[:320],color="green",label="DQN Trained Path")
plt.plot(s_45x[:320],s_45y[:320],color="green",label="DQN Trained Path")
plt.plot(s_135x[:370],s_135y[:370],color="green")
# plt.plot(s90x,s90y,color="green")
plt.plot(s_90x[:280],s_90y[:280],color="green")

# # plt.plot(s135x,s135y,color="green")
# # plt.plot(s_135x,s_135y,color="green")
plt.plot(s180x[:320],s180y[:320],color="green")
# plt.plot(s0x,s0y,color="green")
###############################################
plt.legend(loc="best")
plt.title("Heading Action in Calm Water")
plt.xlabel("Advance (in meters)")
plt.ylabel("Transfer (in meters)")
plt.grid()
plt.savefig("HCCW.jpg",dpi=480)
plt.show()





