import numpy as np
import matplotlib.pyplot as plt
import LOS,waypoints, wp_anaysis
import MMG
##############################################
##############################################

# def getIP(file):
#     data = open(file,'r')
#     data.readline()
#     x,y = [],[]
#     for line in data:
#         dd = line.split(',')
#         x.append(round(float(dd[4]),2))
#         y.append(round(float(dd[5]),2))
#     return x[::],y[::]

# wp_x,wp_y = getIP('P.csv')

# plt.plot(wp_x,wp_y)
# plt.show()
        

# wp_x = [0,438,1104,1704,2148,2262,2048,1591,928,283,-230,-492,-435,-43]

# wp_y = [0,0.2,76,357,918,1535,2208,2652,2846,2712,2289,1654,1020,423]


# prp = []
# for jj in range(len(wp_x)):
#     temp = [wp_x[jj]/15,wp_y[jj]/15]
#     prp.append(temp)



#############################################
######## Parameters Initialization ##########
#############################################
prp,x,y,L = waypoints.straight_line(300,135)
# prp,x,y,L = waypoints.spiral(90)
# prp,x,y,L = waypoints.Fibbanaci_Trajectory(30)
# prp,x,y = waypoints.cardioid(50)
# prp,x,y = waypoints.f2(110)
# L = 0
wpA       = wp_anaysis.activate(prp)

A,B         = wpA
print("#############################")
print("The Starting Quadrant        : ", A[0])
print("The Major Quadrant           : ", A[1])
print("Quadrant 1 points    :",A[2])
print("Quadrant 2 points    :",A[3])
print("Quadrant 3 points    :",A[4])
print("Quadrant 4 points    :",A[5])
print("#############################")
print("X axis Points        : ",A[6]+A[7])
print("Y axis Points        : ",A[8]+A[9])
print("#############################")
print("Quadrant Sequence    : ", B[0])
print("#############################")

sum_error = 0
delta     = 0
H         = [0,0,False,0] # history of waypoints selection
#############################################
#############################################

###########################################
######### First Quadrant ##################
###########################################

def PID_error(HE,sum_error,r):
    Kp = 1.38        # proportional gain
    Kd = 1.366       # differential gain
    Ki = 0.00363     # integral gain
    t  = 0.1
    
    delta = (Kp*HE) + (Ki*sum_error*t) + (Kd*r)
    if delta > 0.61:
        delta = 0.61
    elif delta < -0.61:
        delta = -0.61
    
    # if HE >= np.pi :
    #     delta = -0
    # elif HE < -np.pi:
    #     delta = 0
    
    return delta
    


state   = [1.179,0,0,0,0,0,0]
x_p,y_p = [],[]

aa = []
bb = []
cc = []
# D = np.load('ddd.npy')



for i in range(670):
    # delta = D[i]
    # if 369 < i < 426 :
    #     delta = 0.61
    
    op          = MMG.activate(state,delta)
    y_e, HE, H,psid  = LOS.activate(op,wpA,H)
    aa.append(psid)
    bb.append(HE)
    cc.append(op[5])
    
    delta       = PID_error(HE, sum_error, op[2])
    
    # print(H)
    sum_error   += HE
    x_p.append(op[3])
    y_p.append(op[4])
    state = op

# np.save('ddd.npy',D)

plt.figure(figsize=(9,12))
plt.subplot(2,1,2)
plt.plot(bb,'g--',label = "Heading Error")
plt.plot(aa,'m--',label = "Desired heading angle")
plt.plot(cc,label="Actual heading angle")
plt.axhline(y=0,color='grey')
plt.axvline(x=0,color='grey')
plt.axhline(y= np.pi,color='k',alpha=0.5)
plt.axhline(y= -np.pi,color ='k',alpha=0.5)
plt.text(0,2.7,"HE = $\pi$")
plt.text(0,-2.9,"HE = -$\pi$")
# plt.title("Random curve Trajectory Error")
plt.legend()
# plt.show()

# plt.figure(figsize=(9,9))
plt.subplot(2,1,1)
plt.plot(x_p,y_p,'crimson',label = "MMG Trajectory")
# plt.plot([0,-150],[0,-150])
# plt.plot(np.array(wp_x)/15,np.array(wp_y)/15,'g--',label = "Required Trajectory")
plt.plot(x,y,'g--',label= "Desired Trajectory")
# plt.xlim(-150,150)
# plt.ylim(-150,150)
plt.xlabel("Advance")
plt.ylabel("Transfer")
plt.axvline(x=0,color='y',alpha=0.8)
plt.axhline(y=0,color='y',alpha=0.8)
plt.title("LOS Evaluation by PID controller for Custom Points")
# plt.grid()
plt.legend(loc="best")
plt.show()

###########################################
######### Second Quadrant #################
###########################################



