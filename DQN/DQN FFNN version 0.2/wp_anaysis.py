import numpy as np

def Quadrant_position(A):
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



def activate(prp):
    """

    Parameters
    ----------
    prp : Path Reward Points or waypoints

    Returns
    -------
    Quadrant Analysis Report: (as list)
        [1. Starting Quadrant 
        2. Major Quadrant 
        3. Quadrant 1 points 
        4. Quadrant 2 points
        5. Quadrant 3 points
        6. Quadrant 4 points 
        7. X_axis_positive points
        8. X_axis_negative points
        9. Y_axis_positive points
        10.axis_negative points 
        11.Quadrant sequence]
        
        12.separated points by sequence wise

    """
    Starting_quadrant = 0
    ##########################################
    #### Finding the  Quadrant Sequence ######
    ##########################################
    Quad_sequence = [0]
    for j in range(len(prp)):
        pivot = Quadrant_position(prp[j])
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
    for ii in range(len(prp[split_pivot:])):
        pivot = Quadrant_position(prp[ii])
        
        if pivot == Quad_sequence[quad_pivot]:
            separated_points[quad_pivot].append(prp[ii])
        else:
            separated_points[quad_pivot+1].append(prp[ii])
            quad_pivot += 1
            split_pivot = ii
    
    ##########################
    Q1,Q2,Q3,Q4 = 0,0,0,0
    X_axis_positive, X_axis_negative = 0,0
    Y_axis_positive, Y_axis_negative = 0,0
    ##########################
    for i in range(len(prp)):
        
        x,y = prp[i][0],prp[i][1]
        
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
       
    return [[Starting_quadrant,Major_Quadrant,Q1,Q2,Q3,Q4,
            X_axis_positive, X_axis_negative,
            Y_axis_positive, Y_axis_negative],[Quad_sequence,separated_points]]


###########################################
########### To Check ######################
###########################################
# import waypoints
# prp,x,y,L   = waypoints.Fibbanaci_Trajectory(25)
# # prp,x,y,L   = waypoints.spiral(10)
# # prp,x,y,L = waypoints.straight_line(200,181)

# St_angle = np.arctan2((prp[1][1]-prp[0][1]),(prp[1][0]-prp[0][0]))
# print(St_angle,"the starting angle")
# A,B         = activate(prp)
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
###########################################
################ End ######################
###########################################










