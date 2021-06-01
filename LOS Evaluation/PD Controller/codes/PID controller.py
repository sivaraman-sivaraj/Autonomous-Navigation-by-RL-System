import numpy as np
import matplotlib.pyplot as plt
import LOS,waypoints, wp_anaysis
import MMG
##############################################
##############################################
wp_x = [0,
438.867743137049,
1104.88040949612,
1704.06292876645,
2148.37505543028,
2262.37367591823,
2048.18256809444,
1591.19147306297,
928.039907270854,
283.951840742774,
0,
-230.49423512264,
-492.30366659399,
-435.37598517378,
-43.4865869493611]

wp_y = [0,
0.204934271251896,
76.9470995024240,
357.937130050299,
918.071223031780,
1535.14478057369,
2208.76269137567,
2652.19382880053,
2846.80610881057,
2740,
2450,
2289.46285634797,
1654.91712057620,
1020.68286801990,
423.382537017940]

prp = []
for jj in range(len(wp_x)):
    temp = [wp_x[jj]/45,wp_y[jj]/45]
    prp.append(temp)



#############################################
######## Parameters Initialization ##########
#############################################
# prp,x,y,L = waypoints.straight_line(300,-45)
prp,x,y,L = waypoints.spiral(70)
# prp,x,y,L = waypoints.Fibbanaci_Trajectory(40)
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
    Ki = 0#0.00363  # integral gain
    t  = 0.1
    
    delta = (Kp*HE) + (Ki*sum_error*t) + (Kd*r)
    if delta > 0.61:
        delta = 0.61
    if delta < -0.61:
        delta = -0.61
        
    return delta
    


state   = [1.179,0,0,0,0,0,0]
x_p,y_p = [],[]

aa = []
bb = []
for i in range(2100):
    
    op          = MMG.activate(state,delta)
    y_e, HE, H,psid  = LOS.activate(op,wpA,H)
    aa.append(psid)
    bb.append(op[5])
    delta       = PID_error(HE, sum_error, op[2])
    # print(H)
    sum_error   += HE
    x_p.append(op[3])
    y_p.append(op[4])
    state = op

plt.figure(figsize=(18,12))
plt.plot(aa,'g--')
plt.plot(bb)
plt.axhline(y=0)
plt.axvline(x=0)
plt.axhline(y= np.pi,color='k')
plt.axhline(y= -np.pi,color ='k')
plt.show()

plt.figure(figsize=(9,6))
plt.plot(x_p,y_p,'g')
# plt.plot(np.array(wp_x)/45,np.array(wp_y)/45,'r--')
plt.plot(x,y,'r--')
plt.xlim(-300,300)
plt.ylim(-300,300)
plt.xlabel("Advance")
plt.ylabel("Transfer")
plt.axvline(x=0,color='yellow',alpha=0.5)
plt.axhline(y=0,color='yellow',alpha=0.5)
plt.title("LOS Evaluation by PID controller for Fibbonacci Trajectory")
plt.grid()
plt.show()

###########################################
######### Second Quadrant #################
###########################################



