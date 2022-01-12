import numpy as np
import matplotlib

matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
font = {'family' : 'normal',
        'size'   : 11}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import waypoints


wpo,x_o,y_o,L         = waypoints.Arc_spline()
wp,x_c,y_c,L         = waypoints.Ellipse(110,55)

STPx,STPy =[],[]
for i in range(len(wp)):
    STPx.append(wp[i][0])
    STPy.append(wp[i][1])
    
STPxc,STPyc =[],[]
for i in range(len(wpo)):
    STPxc.append(wpo[i][0])
    STPyc.append(wpo[i][1])


Tox,Toy = np.load("openloop.npy")
Tcx,Tcy = np.load("closedloop.npy")


plt.figure(figsize=(9,6))
plt.plot(STPx,STPy,color="red",marker="8",label= "Closed Loop Waypoints",linestyle='dashed')
plt.plot(Tcx,Tcy,"b",label="Closed Loop Trained Path")

plt.plot(STPxc,STPyc,color="y",alpha=0.9,marker="8",linestyle='dashed',label="Open Loop Waypoints")


plt.plot(Tox,Toy,"g",label="Open Loop Trained Path")

plt.legend(loc="best")
plt.title("Path Following in Calm Water")
plt.xlabel("Advance (in meters)")
plt.ylabel("Transfer (in meters)")
plt.grid()
plt.savefig("PFCW.jpg",dpi=480)
plt.show()




