import numpy as np
import matplotlib.pyplot as plt

def distance_scrutinizer(wp):
    """
    Parameters
    ----------
    wp   : waypoints might be in unequal distances

    Returns
    -------
    S_wp : waypoints in a distance of 2*LBP

    """
    S_wp = [wp[0]]
    for i in range(1,len(wp)):
        A  = S_wp[-1] # wp_k
        B = wp[i]     # wp_k1
        D = np.sqrt(np.square(A[0]-B[0]) + np.square(A[1]-B[1]))
        
        if D >= 14:
            S_wp.append(B)
    return S_wp
    


def spiral(pivot):
    wp = list()
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
        wp.append([y,i])
        
    for i in range(int(pivot)+1):
        y = np.sqrt((r1**2)-((i)**2))
        Y.append(r1+i)
        X.append(y)
        wp.append([y,r1+i])
    ############################  
    ######### second arc #######
    ############################
    r2 = 2 * r1
    for i in range(1,r2+1):
        y = np.sqrt((r2**2)-((i)**2))
        Y.append(y)
        X.append(-i)
        wp.append([-i,y])
        
    for i in range(r2+1):
        y = np.sqrt((r2**2)-((i)**2))
        Y.append(-i)
        X.append(-y)
        wp.append([-y,-i])
    #################################
    ########## third  arc ###########
    #################################
    r3 = 1.5 * r2
    for i in range(1,int(r3+1)):
        x = np.sqrt((r3**2)-((i)**2))
        Y.append(-x+r2-r1)
        X.append(i)
        wp.append([i,-x+r2-r1])
        
    for i in range(int(r3+1)):
        y = np.sqrt((r3**2)-((i)**2))
        Y.append(i+r2-r1)
        X.append(y)
        wp.append([y,i+r2-r1])
        
    
    # #################################
    # ######### Arc Length  ###########
    # #################################
    L = 0
    L += np.pi * r1
    L +=  np.pi * r2
    L +=  np.pi * r3
    S_wp = distance_scrutinizer(wp)
    return S_wp,X,Y,L



def straight_line(inertial_frame_limit,theta):
        """
        Parameters
        ----------
        inertial_frame_limit : required way points range
        
        Returns
        -------
        wp : waypoints
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
        wp = list() # path reward points
        # starting_point = [0,0] #starting point of the ship
        # prp.append(starting_point)
        
        if -45 <= theta <= 45:
            for e in range(inertial_frame_limit):
                y_t = e*(np.tan(a))
                if abs(y_t) < abs(inertial_frame_limit):
                    temp = [e,y_t]
                    wp.append(temp)
        elif -135 >= theta >= -180 or 135 <= theta <= 180:
            for e in range(inertial_frame_limit):
                y_t = -e*(np.tan(a))
                if abs(y_t) < inertial_frame_limit:
                    if e == 0:
                        temp = [e,-y_t]
                    else:
                        temp = [-e,y_t]
                    wp.append(temp)
                        
        elif 45 < theta < 135 :
            for e in range(inertial_frame_limit):
                x_t = -e/(np.tan(a))
                if abs(x_t) < inertial_frame_limit:
                    temp = [-x_t,e]
                    wp.append(temp)
        elif -45 > theta > -135 :
            for e in range(inertial_frame_limit):
                x_t = -e/(np.tan(a))
                if abs(x_t) < inertial_frame_limit:
                    temp = [x_t,-e]
                    wp.append(temp)
        
        
        ############################
        #### path reward end #######
        ############################ 
        x,y = list(),list()
        for i in range(len(wp)):
            x.append(wp[i][0])
            y.append(wp[i][1])
       
        x = x[::]
        y = y[::]
        
        ### Length of Trajectory ###
        L = np.sqrt((wp[-1][0]**2) + (wp[-1][1]**2))
        S_wp = distance_scrutinizer(wp)
        return S_wp,x,y,L
    
    



def Fibbanaci_Trajectory(pivot):
    wp = list()
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
        wp.append([x,i])
    
    ############################  
    ####### second quadrant ####
    ############################
    r2 = 2*r1
    for i in range(r2):
        x = np.sqrt((r2**2)-((i)**2))
        X.append(x-r1)
        Y.append(r1+i)
        wp.append([x-r1,r1+i])
    ############################  
    ####### third quadrant ####
    ############################
    r3 = 2 * r2
    for i in range(r3):
        y = np.sqrt((r3**2)-((i)**2))
        X.append(-i-r1)
        Y.append(y-r1)
        wp.append([-i-r1,y-r1])
    #################################
    #######  fourth quadrant ########
    #################################
    r4 = 2 * r3
    for i in range(r4):
        x = np.sqrt((r4**2)-((i)**2))
        X.append(-x+r2+r1)
        Y.append(-i-r1)
        wp.append([-x+(r2+r1),-i-r1,])
    # #################################
    # #######  fifth quadrant ########
    # #################################
    r5 = 2 * r4
    for i in range(r5):
        y = np.sqrt((r5**2)-((i)**2))
        X.append(i+r2+r1)
        Y.append(-y+r4-r1)
        wp.append([i+r2+r1,-y+r4-r1])
        
    #################################
    ######### Arc Length  ###########
    #################################
    L = 0
    L += 0.5*np.pi * r1
    L += 0.5 * np.pi * r2
    L += 0.5 * np.pi * r3
    L += 0.5 * np.pi * r4
    L += 0.5 * np.pi * r5
    S_wp = distance_scrutinizer(wp)
    return S_wp,X,Y,L



    
def cardioid(a):
    X,Y = [],[]
    wp = []
    for i in range(0,-180,-1):
        x = 2*a*(1-np.cos(np.deg2rad(i)))*np.cos(np.deg2rad(i))
        y = 2*a*(1-np.cos(np.deg2rad(i)))*np.sin(np.deg2rad(i))
        X.append(x)
        Y.append(y)
        wp.append([x,y])
        
    for i in range(180,0,-1):
        x = 2*a*(1-np.cos(np.deg2rad(i)))*np.cos(np.deg2rad(i))
        y = 2*a*(1-np.cos(np.deg2rad(i)))*np.sin(np.deg2rad(i))
        X.append(x)
        Y.append(y)
        wp.append([x,y])
        
    wp_new = [] # to change the clockwise / counter cloakwise
    for j in range(len(wp)):
        temp = wp[j]
        wp_new.append(temp)
    
    L = 8*a # length of the cardioid formula
    S_wp = distance_scrutinizer(wp_new)
    return S_wp,X,Y,L



def parametric(a):
    X,Y = [],[]
    wp = []
    for i in range(0,-180,-1):
        x = 2*a*(1-np.cos(np.deg2rad(i)))*np.cos(np.deg2rad(i))*np.cos(np.deg2rad(i))
        y = 2*a*(1-np.cos(np.deg2rad(i)))*np.sin(np.deg2rad(i))
        X.append(x)
        Y.append(y)
        wp.append([x,y])
        
    for i in range(180,0,-1):
        x = 2*a*(1-np.cos(np.deg2rad(i)))*np.cos(np.deg2rad(i))*np.cos(np.deg2rad(i))
        y = 2*a*(1-np.cos(np.deg2rad(i)))*np.sin(np.deg2rad(i))
        X.append(x)
        Y.append(y)
        wp.append([x,y])
        
    wp_new = [] # to change to counter clockwise 
    for j in range(len(wp)):
        temp = wp[-j]
        wp_new.append(temp)
    
    L = 0
    S_wp = distance_scrutinizer(wp_new)
    return S_wp,X,Y,L

################################
###### To evaluate #############
################################
# wp,X,Y,L = straight_line(300,-45)
# # wp,X,Y,L = spiral(65)
# # wp,X,Y,L = Fibbanaci_Trajectory(15)
# # wp,X,Y,L = cardioid(70)
# wp,X,Y,L = parametric(70)
# xx,yy = [],[]
# for i in range(len(wp)):
#     xx.append(wp[i][0])
#     yy.append(wp[i][1])

# plt.figure(figsize=(9,6))
# plt.plot(X,Y,'y',label = "Requied Trajectory")
# plt.scatter(xx[::1],yy[::1],color = "purple",label = "waypoints")
# plt.axvline(x=0,color='green',alpha = 0.5)
# plt.axhline(y=0,color='green',alpha = 0.5)
# plt.title("Straight Line Trajectory")
# # plt.title("Spiral Trajectory")
# # plt.title("Fibbonacci Trajectory")
# # plt.title("Cardioid Trajectory")
# # plt.title("Parametric Curve")
# plt.ylabel("Transfer(in meters)")
# plt.xlabel("Advance (in meters)")
# # plt.xlim(-300,300)
# # plt.ylim(-300,300)
# plt.grid()
# plt.legend()
# plt.show()
# print("The lemgth of trajectory is :",  L)
# ################################
###### evaluation end ############
##################################

