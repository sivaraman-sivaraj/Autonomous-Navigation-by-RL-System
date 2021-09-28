import numpy as np

def los_mmg_normalizer(theta):
    
    pivot  = np.sign(theta)
    if pivot >= 0:
        theta  = theta % (2*np.pi)
    else:
        theta  = theta % (-2*np.pi)
    
    if theta > 0 :
        if 0 < theta <= np.pi:
            theta_new  = theta 
        elif theta > np.pi:
            theta_new  = theta - (2*np.pi) 
            
    elif theta < 0:
        if 0 > theta > -np.pi:
            theta_new = theta 
        elif theta < -np.pi:
            theta_new = theta + (2*np.pi) 
    elif theta == 0:
        theta_new = 0
    else : 
        theta_new = theta 
    return theta_new
            
def nearest_point(x,y,SP):
    """
    Parameters
    ----------
    x,y : spatial position of the agent
    SP  : separated points in the prior quadrant

    Returns
    -------
    nearest waypoints index

    """
    D              = dict()
    error_distance = list()                         # calculating the euclidian distance of all Separated Points
    for i in range(len(SP)):
       er_temp         = np.sqrt(((SP[i][0]-x)**2)+((SP[i][1]-y)**2))
       error_distance.append(er_temp)
       D[str(er_temp)] = i

    sorted_distance = sorted(error_distance)    # arranging the points in ascending order
    k               = D[str(sorted_distance[0])] 
    return k                                    # point index

def get_y_e_HE(ip,wp_k,wp_k_1):
    """
    Parameters
    ----------
    current_state :     [u,v,r,x,y,psi,delta,t]- position of ship
    wp_k          :     (x_k,y_k)              - K_th way point  
    wp_k_1        :     (x_k+1,y_k+1)          - K+1_th way point 
    
    Returns
    -------
    cross track error, Heading Angle Error, Desired Heading Angle

    """
    ###############################################
    ## Horizontal path tangential angle/ gamma  ###
    ###############################################
    del_x = wp_k_1[0]-wp_k[0]
    del_y = wp_k_1[1]-wp_k[1]
    g_p = np.arctan2(del_y, del_x)
    #########################################
    ###cross track error calculation (CTE) ##
    #########################################
    y_e     = -(ip[3]-wp_k[0])*np.sin(g_p) + (ip[4]-wp_k[1])*np.cos(g_p)  # Equation 24
    #############################
    ## finding the del_h value ##
    #############################
    lbp            = 7                  # Length between perpendicular
    delta_h        = 2*lbp              # look ahead distance
    ##########################################
    ## Calculation of desired heading angle ##
    ##########################################
    beta           = np.arctan2(-ip[1],ip[0])               # drift angle
    psi_d          = g_p + np.arctan2(-y_e,delta_h) - beta # Desired Heading angle # equation 29
    
    psi_a   = los_mmg_normalizer(ip[5])
    HE      = psi_d - psi_a
    
    if abs(HE) > np.pi:
        theta1  = np.pi  - abs(psi_a)
        theta2  = np.pi  - abs(psi_d)
        theta   = theta1 + theta2
        HE_     =  -np.sign(HE) * theta
        HE      = HE_
    
    return y_e, HE



def activate(ip,wpA,H):
    """
    Parameters
    ----------
    ip      : MMG model input state
    wpA     : waypoints Analysis report
                [separated path reward points in prior order B[1], Quadrant Sequence B[0],
                 Starting Quadrant A[0]]
    H       : History of the points already used [Quadrant,waypoint index,last heading error]

    Returns
    -------
    cross track error, Heading Angle Error,History

    """
    
    S_prp   = wpA[1][1]     # Separated waypoint
    QS      = wpA[1][0]     # Quadrant Sequence
    HE_old  = H[2]
    #############################################
    ######## Choosing the best way points #######
    #############################################
    SP        = S_prp[H[0]]
    wp_near   = nearest_point(ip[3],ip[4], SP) # nearest waypoint index
    
    End_flag = False                           # ensure that the last waypoint
    if H[0] == len(QS)-1 and wp_near == len(S_prp[-1]) -1:
        End_flag = True 
    
    if End_flag == True:
        wp_k, wp_k_1 =  S_prp[-1][-2], S_prp[-1][-1]
    
    elif End_flag == False:
        if wp_near == len(SP)-1:
            wp_k, wp_k_1 =  S_prp[H[0]][H[1]],S_prp[H[0]+1][0]
            
        elif wp_near >= H[1] and wp_near < len(SP)-1:
            wp_k, wp_k_1 =  S_prp[H[0]][wp_near],S_prp[H[0]][wp_near+1]
            
        elif wp_near < H[1] :
            wp_k, wp_k_1 =  S_prp[H[0]][wp_near],S_prp[H[0]][H[1]]
    
    
    ###########################################
    ##### Asserting the Final Point ###########
    ###########################################
    if H[0] >= len(QS) -1 and H[1] >= len(S_prp[-1]) - 1:
        wp_k        = S_prp[-1][-1]
        wp_k_1      = [wp_k[0]+0.001,wp_k[1]+0.001]
    
    #############################################
    ###### Calculating the CTE and HE ###########
    #############################################
    y_e, HE         =  get_y_e_HE(ip, wp_k, wp_k_1)
    #############################################
    ########## Updating  the Memory #############
    #############################################
    if End_flag == False:
            
        if wp_near == len(SP)-1:
            H       = [H[0]+1,0,HE_old] 
       
        elif wp_near >= H[1] and wp_near < len(SP)-1:
            H       = [H[0],wp_near+1,HE_old] 
        
        elif wp_near < H[1]:
            H       = [H[0],H[1],HE_old]
            
    elif End_flag == True:
        H = [H[0],len(S_prp[-1])-1,HE_old]
    
    return y_e, HE, H


#########################################
############## To Check #################
#########################################
# import matplotlib.pyplot as plt
# import waypoints
# import wp_analysis

# wp,x,y,L = waypoints.straight_line(150,45)
# # prp,x,y,L   = waypoints.Fibbanaci_Trajectory(25)
# wpA     = wp_analysis.activate(wp)
# H  = [0,0,0]

# R = []
# for i in range(50):
#     ip   =[0,0,0,-i-1,-i-1,3.14]
#     op   = [0,0,0,i,i,0]
#     y_e, HE, H  = activate(ip,wpA,H)
#     print(H)
    
#     R.append(HE)
#     # R.append(HE)
    
# plt.plot(R)
# # plt.ylim(-10,110)
# print(len(wpA[1][1][-1]))
########################################
######### End ##########################
########################################










