import numpy as np
import matplotlib.pyplot as plt


def spiral(pivot):
    prp = list()
    X,Y = list(),list()
    ############################   
    ######### first arc ########
    ############################
    r1 = pivot #(pivot,0)
    for i in range(0,int(2 * pivot)):
        i = 0.5* i
        y = np.sqrt((r1**2)-((r1-i)**2))
        Y.append(i)
        X.append(y)
        prp.append([y,i])
        
    for i in range(int(pivot)+1):
        x = np.sqrt((r1**2)-((i)**2))
        X.append(x)
        Y.append(r1+i)
        prp.append([x,r1+i])
    
    
    ############################  
    ######### second arc #######
    ############################
    r2 = 2 * r1
    for i in range(1,r2+1):
        y = np.sqrt((r2**2)-((i)**2))
        Y.append(y)
        X.append(-i)
        prp.append([-i,y])
        
    for i in range(r2+1):
        y = np.sqrt((r2**2)-((i)**2))
        Y.append(-i)
        X.append(-y)
        prp.append([-y,-i])
    #################################
    ########## third  arc ###########
    #################################
    r3 = 1.5 * r2
    for i in range(1,int(r3+1)):
        x = np.sqrt((r3**2)-((i)**2))
        Y.append(-x+r2-r1)
        X.append(i)
        prp.append([i,-x+r2-r1])
        
    for i in range(int(r3+1)):
        y = np.sqrt((r3**2)-((i)**2))
        Y.append(i+r2-r1)
        X.append(y)
        prp.append([y,i+r2-r1])
        
    # #################################
    # #### Path Reward Points #########
    # #################################
    prpt = prp[::]
    prpt.append(prp[-1])
    # #################################
    # ######### Arc Length  ###########
    # #################################
    L = 0
    L += np.pi * r1
    L +=  np.pi * r2
    L +=  np.pi * r3
        
    return prpt,X,Y,L


##################################
########## To Check ##############
##################################
# prp,X,Y,L = spiral(25)
# xx,yy = [],[]
# for i in range(len(prp)):
#     xx.append(prp[i][0])
#     yy.append(prp[i][1])

# plt.figure(figsize=(9,6))
# plt.plot(X,Y,'y',label = "Requied Trajectory")
# plt.scatter(xx,yy,color = "purple",label = "waypoints")
# plt.axvline(x=0,color='green',alpha = 0.5)
# plt.axhline(y=0,color='green',alpha = 0.5)
# plt.title("Spiral Trajectory")
# plt.xlabel("Transfer(in meters)")
# plt.ylabel("Advance (in meters)")
# plt.grid()
# plt.legend()
# plt.show()
# print("The lemgth of trajectory is :",  L)
##################################
##################################
##################################
    


def straight_line(inertial_frame_limit,theta):
        """
        Parameters
        ----------
        inertial_frame_limit : required way points range
        
        Returns
        -------
        prp : Path Reward Points
        -------
        Warning:
            Heading Angle should be in a range of (-90) to (90) degree
        """
        ### Assertion ###
        if theta > 180 :
            theta = theta - 360
        elif theta < -180:
            theta = theta + 360
        
        #################################
        #### path reward Declaration ####
        #################################
        a = (theta/180) * np.pi # radian mode
        prp = list() # path reward points
        # starting_point = [0,0] #starting point of the ship
        # prp.append(starting_point)
        
        if -45 <= theta <= 45:
            for e in range(inertial_frame_limit):
                y_t = e*(np.tan(a))
                if abs(y_t) < abs(inertial_frame_limit):
                    temp = [e,y_t]
                    prp.append(temp)
        elif -135 >= theta >= -180 or 135 <= theta <= 180:
            for e in range(inertial_frame_limit):
                y_t = -e*(np.tan(a))
                if abs(y_t) < inertial_frame_limit:
                    if e == 0:
                        temp = [e,-y_t]
                    else:
                        temp = [-e,y_t]
                    prp.append(temp)
                        
        elif 45 < theta < 135 :
            for e in range(inertial_frame_limit):
                x_t = -e/(np.tan(a))
                if abs(x_t) < inertial_frame_limit:
                    temp = [-x_t,e]
                    prp.append(temp)
        elif -45 > theta > -135 :
            for e in range(inertial_frame_limit):
                x_t = -e/(np.tan(a))
                if abs(x_t) < inertial_frame_limit:
                    temp = [x_t,-e]
                    prp.append(temp)
        
        
        ############################
        #### path reward end #######
        ############################ 
        x,y = list(),list()
        for i in range(len(prp)):
            x.append(prp[i][0])
            y.append(prp[i][1])
        prp = prp[::]
        x = x[::]
        y = y[::]
        
        ### Length of Trajectory ###
        L = np.sqrt((prp[-1][0]**2) + (prp[-1][1]**2))
        
        return prp,x,y,L
    
    
################################
###### To evaluate #############
################################
# prp,x,y,L = straight_line(300,-10)
# xx,yy = [],[]
# for i in range(len(prp)):
#     xx.append(prp[i][0])
#     yy.append(prp[i][1])

# plt.figure(figsize=(9,6))
# plt.plot(x,y,'y',label = "Requied Trajectory")
# plt.scatter(xx[::1],yy[::1],color = "purple",label = "waypoints")
# plt.axvline(x=0,color='green',alpha = 0.5)
# plt.axhline(y=0,color='green',alpha = 0.5)
# plt.title("Straight Line Trajectory")
# plt.xlabel("Transfer(in meters)")
# plt.ylabel("Advance (in meters)")
# plt.xlim(-300,300)
# plt.ylim(-300,300)
# plt.grid()
# plt.legend()
# plt.show()
# print("The lemgth of trajectory is :",  L)
# ################################
###### evaluation end ##########
################################


def Fibbanaci_Trajectory(pivot):
    prp = list()
    X,Y = list(),list()
    ############################   
    ###### first quadrant ######
    ############################
    r1 = pivot #(pivot,0)
    for i in range(0,int(2 * pivot)):
        i = 0.5* i
        x = np.sqrt((r1**2)-((r1-i)**2))
        X.append(x)
        Y.append(i)
        prp.append([x,i])
    
    ############################  
    ####### second quadrant ####
    ############################
    r2 = 2*r1
    for i in range(r2):
        x = np.sqrt((r2**2)-((i)**2))
        X.append(x-r1)
        Y.append(r1+i)
        prp.append([x-r1,r1+i])
    ############################  
    ####### third quadrant ####
    ############################
    r3 = 2 * r2
    for i in range(r3):
        y = np.sqrt((r3**2)-((i)**2))
        X.append(-i-r1)
        Y.append(y-r1)
        prp.append([-i-r1,y-r1])
    #################################
    #######  fourth quadrant ########
    #################################
    r4 = 2 * r3
    for i in range(r4):
        x = np.sqrt((r4**2)-((i)**2))
        X.append(-x+r2+r1)
        Y.append(-i-r1)
        prp.append([-x+(r2+r1),-i-r1,])
    # #################################
    # #######  fifth quadrant ########
    # #################################
    r5 = 2 * r4
    for i in range(r5):
        y = np.sqrt((r5**2)-((i)**2))
        X.append(i+r2+r1)
        Y.append(-y+r4-r1)
        prp.append([i+r2+r1,-y+r4-r1])
        
    #################################
    #### Path Reward Points #########
    #################################
    prpt     = prp[::2]
    Xt       = X[::2]
    Yt       = Y[::2]
    
    prpt.append(prp[-1])
    Xt.append(X[-1])
    Yt.append(Y[-1])
    #################################
    ######### Arc Length  ###########
    #################################
    L = 0
    L += 0.5*np.pi * r1
    L += 0.5 * np.pi * r2
    L += 0.5 * np.pi * r3
    L += 0.5 * np.pi * r4
    L += 0.5 * np.pi * r5
    return prpt,Xt,Yt,L


###################################
########### To Check ##############
###################################
# r= 30
# prp,X,Y,L = Fibbanaci_Trajectory(r)
# xx,yy = [],[]
# for i in range(len(prp)):
#     xx.append(prp[i][0])
#     yy.append(prp[i][1])

# plt.figure(figsize=(9,6))
# plt.plot(X,Y,'r',label = "Requied Trajectory")
# # plt.plot([0,r],[0,0],color="k")
# # plt.plot([r,r],[r,-r],color="k")
# # plt.plot([3*r,-r],[-r,-r],color="k")
# # plt.plot([-r,-r],[-4.8*r,3*r],color="k")
# # plt.plot([-9*r,5.5*r],[3*r,3*r],color="k")
# # plt.plot([5.5*r,5.5*r],[3*r,19*r],color="k")
# plt.scatter(xx[::8],yy[::8],color = "purple",label = "waypoints",alpha=0.5)
# plt.axvline(x=0,color='green',alpha = 0.5)
# plt.axhline(y=0,color='green',alpha = 0.5)
# plt.title("Fibbonacci Trajectory")
# plt.xlabel("Transfer(in meters)")
# plt.ylabel("Advance (in meters)")
# plt.grid()
# plt.legend()
# plt.show()
# print("The lemgth of trajectory is :",  L)
###################################
###################################
###################################
    


