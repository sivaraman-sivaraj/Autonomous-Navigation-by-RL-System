import numpy as np
import matplotlib

matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import waypoints

wp,x_55,y_55,L       = waypoints.straight_line(250,-55)
wp,x_105,y_105,L     = waypoints.straight_line(200,-105)
wp,x_90,y_90,L       = waypoints.straight_line(200,90)

##################################################
X55,Y55   = np.load("H55.npy")
X105,Y105 = np.load("H105.npy")
Y105 *= -1
X90_1,Y90_1 = np.load("H90 0.5 to 0.25.npy")
X90_2,Y90_2 = np.load("H90 0.6 to 0.25.npy")


plt.figure(figsize=(9,6))
plt.plot([0,200],[0,-200],color="lightseagreen",marker="s",linestyle='dashed',alpha=0.7)
plt.plot(x_55[::14],y_55[::14],"lightseagreen",marker="*")
plt.plot(X55,Y55,"g",label="-45$^\circ$ to -55$^\circ$ learning($\epsilon$ = 0.35 $\longrightarrow$ 0.2)")


plt.plot([0,0],[0,-200],color="gold",marker="s",linestyle='dashed',alpha=0.7)
plt.plot(x_105[::14],y_105[::14],"gold",marker="*")
plt.plot(X105,Y105,"y",label="-90$^\circ$ to -105$^\circ$ learning ($\epsilon$ = 0.4 $\longrightarrow$ 0.2)")


plt.plot([0,200],[0,200],color="orchid",marker="s",linestyle='dashed',alpha=0.7)
plt.plot(x_90[::14],y_90[::14],"orchid",marker="*")

plt.plot(X90_1,Y90_1,"crimson",label="45$^\circ$ to 90$^\circ$ learning($\epsilon$ = 0.6 $\longrightarrow$ 0.2)")
# plt.axhline(y=0,color="grey")
# plt.axvline(x=0,color="grey")
plt.legend(loc="best")
plt.xlabel("Advance(in meters)")
plt.ylabel("Transfer(in meters)")
plt.title("Effect of Pre-Trained weights in New Heading Learning")
plt.grid()
plt.savefig("EPW.jpg",dpi = 480)

