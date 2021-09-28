import numpy as np
TP = np.load("TP.npy")

def Quadrant_position(A):
    """
    Parameters
    ----------
    A : (x,y) position of the ship

    Returns
    -------
    op : corresponding quadrant

    """
    x,y = A[0],A[1]
    op = 0
    if int(x)>0 and int(y)>0:
        op = 1
    elif int(x) < 0 and int(y) >0:
        op = 2
    elif int(x)<0 and int(y) < 0 :
        op = 3
    elif int(x)>0 and int(y)<0:
        op = 4
    ############################
    elif int(y) == 0 and int(x) > 0:
        op = 5
    elif int(y) == 0 and int(x)<0:
        op = 6
    elif int(x) == 0 and int(y) > 0:
        op = 7
    elif int(x) == 0 and int(y) < 0:
        op = 8
    else:
        op = 0
    return op
################################################
########### Reward_1 index setter###############
################################################
def R1_index(G,TP):
    """
    Parameters
    ----------
    G    : Goal
    TP   : [cw,ccw]
    cw   : clockwise points
    ccw  : counter clockwise points

    Returns
    -------
    index step upto which reward should based on HE.

    """
    HA     = np.arctan2(G[1],G[0])
    cw,ccw = TP[0],TP[1]
    
    index = 0
    if G[1] >= 0:
        Flag = True # counter clockwise in the MMG model
    elif G[1]<0:
        Flag = False
        
    if Flag == True:
        done = True
        while done:
            del_x = ccw[index+1][0]-ccw[index][0]
            del_y = ccw[index+1][1]-ccw[index][1]
            g_p = np.arctan2(del_y, del_x)
            if abs(HA) - abs(g_p) <= np.deg2rad(60) :
                done = False
            else:
                index +=1
    elif Flag == False:
        done = True
        while done:
            del_x = cw[index+1][0]-cw[index][0]
            del_y = cw[index+1][1]-cw[index][1]
            g_p = np.arctan2(del_y, del_x)
            if abs(HA) - abs(g_p) <= np.deg2rad(60) :
                done = False
            else:
                index +=1
    if abs(HA) > np.deg2rad(60):
        index +=3 # here, we are adding the numerical value 3, just for tolerence
    return index
################################################
####### Goal setter algorithm ##################
################################################

def Euclidean_Distance(A,B):
    X = np.square(A[0]-B[0])
    Y = np.square(A[1]-B[1])
    D = np.sqrt(X+Y)
    return D

def strightline_goal_setter(wp,j):
    """
    Parameters
    ----------
    wp : set of waypoints
    j  : last used point
    Returns
    -------
    next goal (mostly expexted to be in straight line) , Length of the curve
    
    Assertions
    -----------
    when dx = 0, it needs unique checking, becuase, slope(dy/dx) goes to infinity

    """
    wp = wp[j:]
    I = 1
    #####################################
    ####### Assertions for dx =0 ########
    ##################################### 
    if round((wp[1][0] - wp[0][0]),1) == 0.0:
        m = 0.0
    else:
        m = (wp[1][1]-wp[0][1]) / (wp[1][0] - wp[0][0])
    
    done = True
    i = 2
    while done:
        #####################################
        ####### Assertions for dx =0 ########
        ##################################### 
        if round((wp[i][0] - wp[i-1][0]),1) == 0.0:
            m_temp = 0.0
        else:
            m_temp = (wp[i][1]-wp[i-1][1]) / (wp[i][0] - wp[i-1][0])
        #####################################
        i += 1
        I = i
        if round(m_temp,1) != round(m,1) or i == len(wp) -1:
            done = False
            
    X = np.square(wp[I][0]  - wp[0][0])  
    Y = np.square(wp[I][1]  - wp[0][1]) 
    D = np.sqrt(X+Y)
    return I,D # waypoint index and Length

def spline_goal_setter(wp,j):
    """
    Parameters
    ----------
    wp : waypoints

    Returns
    -------
    next goal from current point

    """
    L    = 0
    P0   = wp[j]
    done = True
    i    = j+1
    while done:
        L1 = Euclidean_Distance(P0,wp[i])
        P0 = wp[i]
        L += L1
        i += 1
        if L >= 105 or i == len(wp) - 1: # 10 times of lbp
            done = False
    return i,L # waypoint index and Length
    


def Goal_declaration(wp,N):
    """
    Parameters
    ----------
    wp : waypoints

    Returns
    -------
    goal points in required distance(lbp) # generally 10 times lbp
    [goals, steps, length of episodes]

    """
    step_distance = 0.75 # according to froud scale for L7 model
    done          = True
    i             = 0 #initial point
    Goals,Steps,Episodes = [[0,0]],[0],[0] # to initate the whole process
    
    while done:
        j,L = strightline_goal_setter(wp,i)
        if L >= 70 :
            step = round(L/step_distance) + Steps[-1]
            Goals.append(wp[j])
            Steps.append(step) 
            Episodes.append(N) # from previous study as 6000
            i = j
        else:
            j,L = spline_goal_setter(wp, i)
            step = round(L/step_distance) + Steps[-1]
            Goals.append(wp[j])
            Steps.append(step) 
            Episodes.append(N) # from previous study as 6000
            i = j
        if j == len(wp) -1 :
            done = False
    Steps[1]  += 60
    return Goals,Steps,Episodes
###########################################
############# Conglomeration ##############
###########################################
def activate(wp,N=4000):
    """

    Parameters
    ----------
    wp :  waypoints
    N  :  No of episodes
    Returns
    -------
    Quadrant Analysis Report: (as list)
       A =  [1. Starting Quadrant 
        2. Major Quadrant 
        3. Quadrant 1 points 
        4. Quadrant 2 points
        5. Quadrant 3 points
        6. Quadrant 4 points 
        7. X_axis_positive points
        8. X_axis_negative points
        9. Y_axis_positive points
        10. Y_axis_negative points]
        
        B = [1.Quadrant sequence
         2.separated points by sequence wise]
        
        C = [goals,steps,number of episodes]
        D = reinforcement index for starting position

    """
    Starting_quadrant = 0
    ##########################################
    #### Finding the  Quadrant Sequence ######
    ##########################################
    Quad_sequence = [0]
    for j in range(len(wp)):
        pivot = Quadrant_position(wp[j])
        if pivot != Quad_sequence[-1]:
            Quad_sequence.append(pivot)
    ###########################################
    ########### Points Separation #############
    ###########################################
    separated_points = list()
    
    for k in range(len(Quad_sequence)):
        separated_points.append([])
    
    split_pivot     = 0
    quad_pivot      = 0  
    for ii in range(len(wp[split_pivot:])):
        pivot = Quadrant_position(wp[ii])
        
        if pivot == Quad_sequence[quad_pivot]:
            separated_points[quad_pivot].append(wp[ii])
        else:
            separated_points[quad_pivot+1].append(wp[ii])
            quad_pivot += 1
            split_pivot = ii
    
    ##########################
    Q1,Q2,Q3,Q4 = 0,0,0,0
    X_axis_positive, X_axis_negative = 0,0
    Y_axis_positive, Y_axis_negative = 0,0
    ##########################
    for i in range(len(wp)):
        
        x,y = wp[i][0],wp[i][1]
        
        ################################################
        #### Finding the inital waypoints's quadrant ###
        ################################################
        
        if i == 20: # As per need, we can change the value 20
            Starting_quadrant = np.argmax([Q1,Q2,Q3,Q4,
                                           X_axis_positive, X_axis_negative,
                                           Y_axis_positive, Y_axis_negative]) + 1
        ########################################
        ### Assigning Points to the Quadrant ###
        ########################################
        if int(x)>0 and int(y)>0:
            Q1 += 1
        elif int(x) < 0 and int(y) >0:
            Q2 += 1
        elif int(x)<0 and int(y) < 0 :
            Q3 += 1
        elif int(x)>0 and int(y)<0:
            Q4 += 1
        ############################
        elif int(y) == 0 and int(x) > 0:
            X_axis_positive += 1
        elif int(y) == 0 and int(x)<0:
            X_axis_negative += 1
        elif int(x) == 0 and int(y) > 0:
            Y_axis_positive += 1
        elif int(x) == 0 and int(y) < 0:
            Y_axis_negative += 1
        ##########################################
        ####### Waypoints major Quadrant #########
        ##########################################
        Major_Quadrant = np.argmax([Q1,Q2,Q3,Q4,X_axis_positive, X_axis_negative,
                                    Y_axis_positive, Y_axis_negative]) + 1
    
    
    ###### Gathering #######
    A = [Starting_quadrant,Major_Quadrant,Q1,Q2,Q3,Q4,
            X_axis_positive, X_axis_negative,
            Y_axis_positive, Y_axis_negative] 
    B = [Quad_sequence,separated_points]
    C = Goal_declaration(wp,N)
    D = R1_index(C[0][1],TP)
    return [A,B,C,D]


###########################################
########### To Check ######################
###########################################
# import waypoints
# # wp,x,y,L   = waypoints.Fibbanaci_Trajectory(25)
# # wp,x,y,L   = waypoints.spiral(15)
# wp,x,y,L = waypoints.straight_line(250,45)
# wp,x,y,L   = waypoints.cardioid(25)

# St_angle = np.arctan2((wp[1][1]-wp[0][1]),(wp[1][0]-wp[0][0]))
# print(np.rad2deg(St_angle),"the starting angle")
# A,B,C,D         = activate(wp)
# print("The goal points are",C[0])
# print("#############################")
# print("The Starting Quadrant        : ", A[0])
# print("The Major Quadrant           : ", A[1])
# print("Quadrant 1 points    :",A[2])
# print("Quadrant 2 points    :",A[3])
# print("Quadrant 3 points    :",A[4])
# print("Quadrant 4 points    :",A[5])
# print("#############################")
# print("X axis Points        : ",A[6]+A[7])
# print("Y axis Points        : ",A[8]+A[9])
# print("#############################")
# print("Quadrant Sequence    : ", B[0])
# print("#############################")
# print("The number of steps are ", C[1])
# print("#############################")
###########################################
################ End ######################
###########################################
