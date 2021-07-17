import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn #,torch.nn.functional as F
import torch.optim as optim
import time,random
from collections import namedtuple, deque
from itertools import count

class MMG():

    def MMG_Time_Derivative(u,v,r,x,y,psi,delta,rps = 12.05):
        """
        ----------
        ip : current state of the ship [u,v,r,x,y,psi]
        delta : commanded rudder angle
        rpm : commanded propeller speed
    
        Returns
        -------
        Time derivative of ship state[u_dot,v_dot,r_dot,U,V,r]
    
        """
        ### Assertion ###
        if u == 0 or u < 0:
            u = 0.00001
        if rps == 0 or rps < 0:
            rps = 1
        ### ###
        U             = np.sqrt((u**2)+(v**2))   # Resultant speed
        beta          = np.arctan(-v/u)          # Hull drift angle at midship
        Np            = rps                      #just for notation
        
        ################################################
        ###### Kinematic Parameters of the ship ########
        ################################################
        Lpp     = 7      # length between perpenticular
        d       = 0.46   # ship draft
        xG      = 0.25   # Longitudinal coordinate of center of gravity of ship
        rho     = 1025   # Water density
        Volume  = 3.27   # Displacement volume of ship
        
        ######################################################
        #### Mass, Added Mass and Added Moment of inertia ####
        ######################################################
        mx,my,jz = 0.022,0.223,0.011
        
        ### Non DiMensionalizing(ndm) parameters ###
        ndm_force   = 0.5*rho*Lpp*d*(U**2)
        ndm_moment  = 0.5*rho*(Lpp**2)*d*(U**2)
        ndm_mass    = 0.5*rho*d*(Lpp**2)
        ndm_massMoI = 0.5*rho*d*(Lpp**4)
        
        ### Non Dimensional mass ###
        mx = mx*ndm_mass
        my = my*ndm_mass
        jz = jz*ndm_massMoI
        m = Volume*rho
        IzG = m * ((0.25*Lpp)**2) #Moment of inertia of ship around center of gravity
        #####################################################
        ####### Governing Equation (Equation 4) #############
        #####################################################
        
        ### Mass Matrix ### (u_dot,v_dot,r_dot)
        M = np.array([[(m+mx),0,0],[0,(m+my),m*xG],[0,m*xG,(jz+((xG**2)*m)+IzG)]])
        
        ### LHS ### (Other than u_dot,v_dot,r_dot) #which should be added on right side
        LHS_r = np.array([[-(m+my)*v*r-xG*m*(r**2)], [(m+mx)*u*r], [m*xG*u*r]])
        
        #####################################################
        ####### Hydrodynamic Force on Hull(F_hull)  #########
        #####################################################
        ### Non Dimensional Velocity ###
        v_ndm = v/U
        r_ndm = r*Lpp/U
        
        R0 = 0.022 # Ship resistance coefficient in straight moving
        
        ### Non Dimensional Hydrodynamic Derivatives ###
        X_vv, X_vr ,X_rr ,X_vvvv = -0.04, 0.002, 0.011, 0.771
        Y_v, Y_R, Y_vvv, Y_vvr, Y_vrr, Y_rrr = -0.315, 0.083, -1.607, 0.379, -0.391, 0.008
        N_v, N_R, N_vvv, N_vvr, N_vrr, N_rrr = -0.137, -0.049, -0.03, -0.294, 0.055, -0.013
        
        ### Force and Moment ###
        X_hull =  -R0 + (X_vv*(v_ndm**2)) + (X_vr*v_ndm*r_ndm) + (X_rr*(r_ndm**2)) + (X_vvvv*(v_ndm**4))
                    
        Y_hull =  (Y_v*v_ndm) + (Y_R*r_ndm) + (Y_vvv*(v_ndm**3)) + (Y_vvr*(v_ndm**2)*r_ndm) + (Y_vrr*v_ndm*(r_ndm**2)) + (Y_rrr*(r_ndm**3))
                    
        N_hull = (N_v*v_ndm) + (N_R*r_ndm) + (N_vvv*(v_ndm**3)) + (N_vvr*(v_ndm**2)*r_ndm) + (N_vrr*v_ndm*(r_ndm**2)) + (N_rrr*(r_ndm**3))
        
        ### Equation 6 ###
        X_hull = ndm_force * X_hull
        Y_hull = ndm_force * Y_hull
        N_hull = ndm_moment* N_hull
        
        ### Hull Force ###
        F_hull = np.array([[X_hull],[Y_hull],[N_hull]])
        
        #####################################################
        ##### Propellar Force on Ship(F_propellar)  #########
        #####################################################
        tp          = 0.220                    # Thrust deduction factor
        Dp          = 0.216                    # Propeller diameter
        wp0         = 0.40                     # Effective wake in straight moving
        k0,k1,k2    = 0.2931, -0.2753, -0.1385 # 2nd order polynomial function coefficient
        xP          = -0.48                    # Longitudinal coordinate of propellerposition
        beta_P      = beta - (xP * r_ndm)      # geometrical inflow angle 
        
        
        C1,C2       = 2, 1.6 if beta_P > 0 else 1.2 # wake  change characteristic coefficents
        wp          = (1 - wp0)* (1 + ((1 - np.exp(-C1*abs(beta_P)))*(C2 - 1)))
                                                    # Equation 16
        
        Jp          =  u*(1-wp)/(Np*Dp)             # Equation 11
        KT          =  k0 + k1*Jp + k2*(Jp**2)      # Equation 10
        T           =  rho*(Np**2)*(Dp**4)*KT       # Equation 9
        X_propellar =  (1 - tp)*T                   # Equation 8
        
        ### Propellar Force ###
        F_propellar = np.array([[X_propellar],[0],[0]])
        
        #####################################################
        ######### Rudder Force on Ship(F_rudder)  ###########
        #####################################################
        
        ### Equation 46 ###
        epsilon    = 1.09                   # Ratio of wake fraction
        k          = 0.5                    # An experimental constant forexpressing uR
        HR         = 0.345                  # Rudder span length
        eta        = Dp/HR                  # Ratio of propeller diameter to rudderspan 
        uP         = (1 - wp)*u             # propeller inflow velocity
        
        uR1   = np.sqrt(1 + ((8*KT)/(np.pi*(np.square(Jp)))))
        uR2   = np.square(1 + k*(uR1 -1))
        uR    = epsilon * uP * np.sqrt( (eta*uR2) + (1-eta))
        
        ### Equation 23 ###
        lR          = -0.710                        # Effective longitudinal coordinate
        beta_R      =  beta -(lR*r_ndm)             # Effective inflow angle to rudder in maneuvering motions
        gamma_r     = 0.396 if beta_R < 0 else 0.64 # Flow straightening coefficient
        
        vR    =  U * gamma_r * beta_R
       
        ### Equation 19 ###
        AR          = 0.0539                    # Profile area of movable part of mariner rudder
        f_alpha     = 2.747                     # Rudder lift gradient coefficient
        alpha_R     = delta - np.arctan(vR/uR)  # Effective inflow angle to rudder
        Ur          = np.sqrt((uR**2)+(vR**2))  # Resultant inflow velocity to rudder
        
        F_normal = 0.5 * rho * AR * (Ur**2) * f_alpha * np.sin(alpha_R)
        
        ### Equation 18 ###
        tR         = 0.387                      # Steering resistance deduction factor
        aH         = 0.312                      # Rudder force increase factor
        xH         = -0.45* Lpp  # -0.464       # Longitudinal coordinate of acting point of the additional lateral force
        xR         = -0.5 * Lpp                 # Longitudinal coordinate of rudder position 
        
        X_rudder =  -(1 - tR) * F_normal * np.sin(delta)
        Y_rudder =  -(1 + aH) * F_normal * np.cos(delta)
        N_rudder =  -(xR + (aH*xH)) * F_normal * np.cos(delta)
        
        ### Rudder Force ###
        F_rudder = np.array([[X_rudder],[Y_rudder],[N_rudder]])
       
        #####################################################
        ##### Solution of Governing Equation (A.X = b)  #####
        #####################################################
        A_inverse       = np.linalg.inv(M) 
        b               = F_hull + F_propellar + F_rudder - LHS_r
        
        ### u_dot,v_dot,r_dot ###
        X   = A_inverse.dot(b)
        #####################################################
        ############## Kinetics of the Ship   ###############
        #####################################################
        ### Rotation Matrix ###
        R = np.array( [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi),  np.cos(psi), 0], [0, 0, 1]])
        ### Velocity and Moment ###
        Vel_Mom = R.dot(np.array([u,v,r]))
        
        ### time derivatives ###
        TD = [X[0].item(), X[1].item(), X[2].item(),Vel_Mom[0].item(), Vel_Mom[1].item(), Vel_Mom[2].item(),]
        
        return np.array(TD)
    
    
    def activate(ip,delta,rps = 12.05):
          h = 1 # time step as increment
          u,v,r,x,y,psi = ip[0],ip[1],ip[2],ip[3],ip[4],ip[5]
         
          if delta > 0.611:
              delta = 0.611
          elif delta < -0.611:
              delta = -0.611
          else :
              delta = delta
        ###############################    
        ### RK4's K1,K2,K3,K4 Value ###
        ###############################
          K   = np.zeros(6)
          K0  = np.array([u,v,r,x,y,psi])
        
          K1_ = MMG.MMG_Time_Derivative(u,v,r,x,y,psi,delta,rps)
          K1  = h*K1_[:3] 
          k1  = h*K0[:3]
         
          K2_ = MMG.MMG_Time_Derivative(u+(K1[0]/2), v+(K1[1]/2), r+(K1[2]/2), x+(k1[0]/2), y+(k1[1]/2), psi+(k1[2]/2),delta,rps)
          K2  = h*K2_[:3]
          k2  = h*(K0[:3] + 0.5*K1)
              
          K3_ = MMG.MMG_Time_Derivative(u+(K2[0]/2), v+(K2[1]/2), r+(K2[2]/2),x+(k2[0]/2), y+(k2[1]/2), psi+(k2[2]/2),delta,rps)
          K3  = h*K3_[:3]
          k3  = h*(K0[:3] + 0.5*K2)
         
          K4_ = MMG.MMG_Time_Derivative(u+(K3[0]), v+(K3[1]), r+(K3[2]), x+(k3[0]), y+(k3[1]), psi+(k3[2]),delta,rps)
          K4  = h*K4_[:3]
          k4  = h*(K0[:3] + 0.5*K3)
         
         
          K[:3]  =  (1/6) * (K1 + (2*K2) + (2*K3) + K4)
         
          R      = np.array( [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi),  np.cos(psi), 0], [0, 0, 1]])
          del_k  =  ((1/6) * (k1 + (2*k2) + (2*k3) + k4)).reshape(3,1)
          K[3:6] = (R.dot(del_k)).reshape(1,3)
          
          nxt_state = np.array(ip) + K
         
          return nxt_state.tolist()
      
class waypoints():
    
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
        S_wp = waypoints.distance_scrutinizer(wp)
        return S_wp,x,y,L

class wp_Analysis():
    
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
            pivot = wp_Analysis.Quadrant_position(prp[j])
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
            pivot = wp_Analysis.Quadrant_position(prp[ii])
            
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
    

class LOS():
    
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
        lbp            = 7                  # Lpp
        delta_h        = 2*lbp              # look ahead distance
        ##########################################
        ## Calculation of desired heading angle ##
        ##########################################
        beta           = np.arctan2(ip[1],ip[0])               # drift angle
        psi_d          = g_p + np.arctan2(-y_e,delta_h) - beta # Desired Heading angle # equation 29
        
        psi_a   = LOS.los_mmg_normalizer(ip[5])
        HE      = psi_d - psi_a
        
        if abs(HE) > 3.14:
            theta1  = np.pi  - abs(psi_a)
            theta2  = np.pi  - abs(psi_d)
            theta   = theta1 + theta2
            HE_     =  -np.sign(HE) * theta
            HE      = HE_
        
        return y_e, HE,psi_d

    
    
    def activate(ip,wpA,H):
        """
        Parameters
        ----------
        ip      : MMG model input state
        wpA     : waypoints Analysis report
                    [separated path reward points in prior order B[1], Quadrant Sequence B[0],
                     Starting Quadrant A[0]]
        H       : History of the points already used [Quadrant,waypoint index]
    
        Returns
        -------
        cross track error, Heading Angle Error,History, Desired Heading Angle
    
        """
        
        S_prp   = wpA[1][1]     # Separated path reward point
        QS      = wpA[1][0]     # Quadrant Sequence
        St_Q    = wpA[0][0]     # Starting Quadrant
        Flag    = H[2]
        Prec_AT = H[-1]
        #############################################
        ######## Choosing the best way points #######
        #############################################
        SP        = S_prp[H[0]]
        wp_near   = LOS.nearest_point(ip[3],ip[4], SP) # nearest waypoint index
        
        End_flag = False                           # ensure that the last waypoint
        if H[0] == len(QS)-1 and H[1] == len(S_prp[-1]) -1:
            End_flag = True 
        
        if End_flag == False:
            if wp_near == len(SP)-1:
                wp_k, wp_k_1 =  S_prp[H[0]][H[1]],S_prp[H[0]+1][0]
                
            elif wp_near >= H[1] and wp_near < len(SP)-1:
                wp_k, wp_k_1 =  S_prp[H[0]][wp_near],S_prp[H[0]][wp_near+1]
                
            elif wp_near < H[1] :
                wp_k, wp_k_1 =  S_prp[H[0]][wp_near],S_prp[H[0]][H[1]]
        
        elif End_flag == True:
            wp_k, wp_k_1 =  S_prp[H[0]][H[1]-1], S_prp[H[0]][H[1]]
        ###########################################
        ##### Asserting the Final Point ###########
        ###########################################
        if H[0] >= len(QS) -1 and H[1] >= len(S_prp[-1]) - 1:
            wp_k        = S_prp[-1][-1]
            wp_k_1      = [wp_k[0]+0.001,wp_k[1]+0.001]
        
        #############################################
        ###### Calculating the CTE and HE ###########
        #############################################
        y_e, HE,psi_d      =  LOS.get_y_e_HE(ip, wp_k, wp_k_1)
        #############################################
        ########## Updating  the Memory #############
        #############################################
        if End_flag == False:
                
            if wp_near == len(SP)-1:
                H       = [H[0]+1,0,Flag,Prec_AT] 
           
            elif wp_near >= H[1] and wp_near < len(SP)-1:
                H       = [H[0],wp_near+1,Flag,Prec_AT] 
            
            elif wp_near < H[1]:
                H       = [H[0],H[1],Flag,Prec_AT]
                
        elif End_flag == True:
            H = H
        
        return y_e, HE, H,psi_d
        
    
    
class Reward():
    
    def get(ip,op,y_e,HE,G):
    
        x_d0 = np.square(G[0] - ip[3])
        y_d0 = np.square(G[1] - ip[4])
        D0 = np.sqrt(x_d0+y_d0)
        
        x_d1 = np.square(G[0] - op[3])
        y_d1 = np.square(G[1] - op[4])
        D1 = np.sqrt(x_d1 + y_d1)
        
        if D0 - D1 >=  0 :
            if abs(HE) <= (np.pi):
                if abs(y_e) <= 0.5:
                    R =100
                elif 0.5 < abs(y_e) <= 1.0:
                    R = 20
                else:
                    c0 = abs(y_e)/63
                    c1 = 1 - c0
                    R  = 10 * c1 
            elif abs(HE) > (np.pi):
                R = -0.5
        else:
            R = -0.5
        ################################
        ########### Assertion ##########
        ################################
       
        Rf = np.array([R])
            
        return Rf



    
class MLFFNN(nn.Module):
    def __init__(self):
        super(MLFFNN, self).__init__()
        self.ipL = nn.Linear(6,128)   # has pre trained weight
        self.HL1 = nn.Linear(128,128) # has pre trained weight
        self.HL2 = nn.Linear(128,128) # has pre trained weight
        self.opL = nn.Linear(128,7)  # output layer
        
    def forward(self, x):
        x = torch.sigmoid(self.ipL(x))#sigmiodal activation function
        x = torch.sigmoid(self.HL1(x))
        x = torch.sigmoid(self.HL2(x))
        x = self.opL(x)
        return x

    
class Surrounding(): #basinGrid-s0
     
     def __init__(self,wpA,grid_size,u,wp):
         
        '''
        The initialization of the grid-world parameters.

        Parameters
        ----------
        wpA                 : waypoints Analysis
        grid_size           : totol grid size
        u                   : intial velocity of ship
        wp                  : waypoints
        Returns
        -------
        environment 

        '''
        self.wp            = wp
        self.wpA            = wpA
        self.S_wp          = self.wpA[1][1]
        self.St_Q           = self.wpA[0][1]
        self.grid_size      = grid_size
        self.u              = u
        self.goals          = self.S_wp[-1][-1]         # end point of ship
        self.done,self.viewer   = False, None            # see here
        self.st_x, self.st_y    = 0,0
        self.actions_set        = {'0':-35,'1':-20,'2':-5,'3':0,'4':5,'5':20,'6':35}
        self.St_angle           = np.arctan2((self.wp[3][1]-self.wp[0][1]),(self.wp[3][0]-self.wp[0][0]))
        
     
     def reset(self):
        
        self.current_state = torch.tensor([self.u,0,0,self.st_x, self.st_y,0])
        self.done      = False
        self.H         = [0,0,False,0]
        return self.current_state,self.H
    
     def step(self,C):
         self.action,self.H        = C[0],C[1]
         self.Flag                 = self.H[2] # True
         
         if self.Flag == True:
             self.ip         = self.current_state.clone().detach()
             self.op         = MMG.activate(self.ip,np.deg2rad(self.actions_set[str(self.action)]))
             self.y_e,self.HE,self.H,self.psi_cte = LOS.activate(self.op,self.wpA,self.H)
             self.reward_a                        = Reward.get(self.ip,self.op,self.y_e,self.HE,self.goals)
             #print(np.rad2deg(self.HE),"he eerroorr")
             #################################
             ###### Next State Update ########
             #################################
             self.current_state      = torch.tensor(self.op)
             self.H[-1]              = self.action
             #################################
             ### Epi Termination Creteria ####
             #################################
             ### @1 ### by  reward
             if self.reward_a < 0 :
                 self.done = True
             ### @2 ### by final point
             self.x_d0 = np.square(self.goals[0] - self.op[3])
             self.y_d0 = np.square(self.goals[1] - self.op[4])
             self.D0 = np.sqrt(self.x_d0+ self.y_d0)
              
             if self.D0 < 3 :
                 self.done = True
                 
             
         
         if self.Flag == False:
             self.ip         = self.current_state.clone().detach()
             if self.St_angle > 0:
                 if (self.St_angle-self.ip[5]) <= np.pi :
                     self.action_F   = 6            
             if self.St_angle < 0:
                 if (self.St_angle-self.ip[5]) >= -np.pi :
                     self.action_F   = 0  
             if self.St_angle == 0:
                 self.action_F = 3
             
             self.op                            = MMG.activate(self.ip,np.deg2rad(self.actions_set[str(self.action_F)]))
             self.y_e,self.HE,self.H,self.psi_d = LOS.activate(self.op,self.wpA,self.H)
             self.reward_a                      = np.array([0.0]) # we are inducing the step #Reward.get(self.ip,self.op,self.y_e,self.HE,self.goals)
             
             #################################
             ###### Next State Update ########
             #################################
             self.current_state      = torch.tensor(self.op)
             self.H[-1]              = self.action_F
             #################################
             ####### Flag Declaration ########
             #################################
             
             if abs(self.HE) <= np.deg2rad(75):
                 self.H[2]  = True
             self.done = False
         return self.current_state, self.reward_a, self.done,self.H
             
         

     def action_space_sample(self):
        n = np.random.randint(0,7)
        return n
    
     def action_space(self):
        return np.arange(0,7,1)
     

######################################
############## Memory ################
######################################
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    def all_sample(self):
        return self.memory
    
memory = ReplayMemory(3000)

########################################
########## eps calculation #############
########################################
import math
def eps_calculation(i_episode):
    """
    decaying epision value is an exploration parameter
    
    """
    
    start = 0.999
    end   = 0.2
    eps = end + (start- end) * math.exp(-1 * i_episode / 20)
    return eps

##########################################
########### action selection #############
##########################################

def select_action(state,eps):
    
    sample = random.random()
    
    if sample > eps:
        with torch.no_grad():
            temp  = state.detach().tolist()
            op    = policy_net(torch.tensor(temp,device=device))
            return np.argmax(op.cpu().detach().numpy())
    else:
        return env.action_space_sample()

#########################################
############# optimizer #################
#########################################

def optimize_model():
    if len(memory) < batch_size:
        return 0
    
    transitions = memory.sample(batch_size)
    batch       = Transition(*zip(*transitions))
    # print(batch)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)),
                                  device=device, dtype=torch.bool)
    
    non_final_next_states = torch.tensor(batch.next_state, device=device)
    state_batch  = torch.tensor(batch.state, device=device)
    action_batch = torch.tensor(batch.action, device=device)
    reward_batch = torch.tensor(batch.reward, device=device)
    action_batch = action_batch.unsqueeze(1)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values   = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)
    # Compute Mean Square Loss
    criterion = nn.MSELoss()#.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss
    

########################################
############ DQN Training ##############
########################################
def Train_DQN(N):
    """
    Parameters
    ----------
    N           : Number of Episodes

    Returns
    -------
    NoEpi                : Episode Duration for N episodes
    CUmulative_reward    : Cumulative reward for N episodes
    HEs                  : Average Heading Error for N episodes
    MSEs                 : Mean Square Error for N episodes
    """
    NoEpi             = [] # episode duration
    Cumulative_reward = [] # cumulative reward
    HEs               = [] # Average Heading error for an episode
    MSEs              = [] # Mean Square error for N episode
    for i_episode in range(N):
        total_reward = 0
        total_he     = 0 
        total_mse    = 0
        eps   = eps_calculation(i_episode)
        if i_episode % 200 == 0:
            print("Episode : ",i_episode, "Running....!")
        ##############################################
        #### Initialize the environment and state ####
        ##############################################
        ship_current_state,H = env.reset()
        state = ship_current_state
        
        for it in count():
            
            C = [select_action(state,eps),H]
            observation, reward, done, H = env.step(C) # Select and perform an action
            #print(reward)
            if it > 350:
                done = True
                
            if done:
                NoEpi.append(it+1)
                break
            
            next_state = observation                   # Observe new state
   
            #######################################
            #### Store the transition in memory ###
            #######################################
            st_m   = state.tolist()
            n_st_m = next_state.tolist()
            r_m    = reward.item()
            
            memory.push(st_m, H[-1], n_st_m, r_m)
            #################################
            ###### Move to the next state ###
            #################################
            state = observation.clone().detach()
            #################################
            ######### optimization ##########
            #################################
            loss = optimize_model()
            total_reward += reward
            total_he     += (abs(np.rad2deg(observation[5])) - theta)
            total_mse    += float(loss)
        
        HEs.append(total_he/it) # theta is global declaration
        Cumulative_reward.append(total_reward/it)
        MSEs.append(total_mse/it)
        ##############################################################################
        ####### Update the target network, copying all weights and biases in DQN #####
        ###############################################################################
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
    L1 = policy_net.cpu().ipL.weight.data
    LL1 = L1.numpy()
    np.save("layer1 weight",LL1)
    L2 = policy_net.HL1.weight.data
    LL2 = L2.numpy()
    np.save("layer2 weight",LL2)
    L3 = policy_net.HL2.weight.data
    LL3 = L3.numpy()
    np.save("layer3 weight",LL3)
    L4 = policy_net.opL.weight.data
    LL4 = L4.numpy()
    np.save("layer4 weight",LL4)
    ##### bias saving ################
    Lb1 = policy_net.ipL.bias.data
    LLb1 = Lb1.numpy()
    np.save("layer1 bias",LLb1)
    Lb2 = policy_net.HL1.bias.data
    LLb2 = Lb2.numpy()
    np.save("layer2 bias",LLb2)
    Lb3 = policy_net.HL2.bias.data
    LLb3 = Lb3.numpy()
    np.save("layer3 bias",LLb3)
    Lb4 = policy_net.opL.bias.data
    LLb4 = Lb4.numpy()
    np.save("layer4 bias",LLb4)
    
    return NoEpi, Cumulative_reward, HEs, MSEs
            
##########################################
########### Image Plotting ###############
##########################################

def plot_result1(NoEpi, Cumulative_reward):
    plt.figure(figsize=(9,12))
    #############################
    plt.subplot(2,1,1)
    N = len(Cumulative_reward)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(Cumulative_reward[max(0, t-100):(t+1)])
    plt.plot(running_avg1,color='r',label="Cumulative Reward")
    plt.title("DQN Training Resuts : Episode Durations & CUmulative Rewards ")
    plt.xlabel("No of Episode")
    plt.ylabel("Reward Unit")
    # plt.ylim(0,110)
    plt.legend(loc="best")
    #plt.grid()
    ##############################
    plt.subplot(2,1,2)
    N = len(NoEpi)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(NoEpi[max(0, t-100):(t+1)])
    plt.plot(running_avg2,color='g', label = "Episode Durations")
    plt.xlabel("No of Episode")
    plt.ylabel("Length of Episodes")
    # plt.ylim(0,300)
    plt.legend(loc="best")
    #plt.grid()
    plt.legend(loc="best")
    #plt.show()
    plt.savefig('pic1.jpg')
    
    
def plot_result2(HEs,MSE):
    plt.figure(figsize=(9,12))
    ##############################
    plt.subplot(2,1,1)
    N = len(MSE)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(MSE[max(0, t-100):(t+1)])
    plt.plot(running_avg1,color='m',label="Mean Square Error")
    plt.title("DQN Training Resuts : MSE Loss , Heading Error ")
    plt.xlabel("No of Episode")
    plt.ylabel("MSE Loss")
    # plt.ylim(0,110)
    plt.legend(loc="best")
    #plt.grid()
    
    
    plt.subplot(2,1,2)
    N = len(HEs)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(HEs[max(0, t-10):(t+1)])
    plt.plot(running_avg2,color='b', label = "Cumulative Heading Error")
    plt.xlabel("No of Episode")
    plt.ylabel("Heading Error in degree")
    plt.axhline(y=0, color = 'yellow')
    plt.legend(loc="best")
    #plt.grid()
    # plt.ylim(0,120)
    #plt.show()
    plt.savefig('pic2.jpg')
###############################################
###############################################
########## Parameters Declaration #############
###############################################
####### Froude Scaling #########
scaled_u              = np.sqrt((7/320)*7.75*7.75)
GFL                   = 300 #Grid Full Length 
theta                 = 90
initial_velocity      = scaled_u # you can fix it as zero for convienient
###############################################
############ waypoints maker ##################
###############################################
wp,x_path,y_path,L = waypoints.straight_line(GFL,theta)
wpA                 = wp_Analysis.activate(wp)
print("The required starting and end point of the ship are ", wp[0],"&",wp[-1])
print("The Length of trajectory is :", L)

###################################################
###### Initializing the Grid Environment ##########
###################################################
env                     = Surrounding(wpA, GFL, initial_velocity, wp)
device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu" ) # if torch.cuda.is_available() else "cpu"

ship_current_state      = torch.tensor([0,0,0,0,0,0])

H                       = [0,0,False,0]     
###################################################
######## Initialing the Q - network ###############
###################################################
batch_size     = 128
gamma          = 0.99
target_update  = 10
done           = False
policy_net     = MLFFNN()
target_net     = MLFFNN()
policy_net     = policy_net.to(device=device)
target_net     = target_net.to(device=device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer      = optim.Adam(policy_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#optim.RMSprop()
##############################################
################ Training DQN ################
##############################################
N   = 5

start                              = time.time()
NoEpi, Cumulative_reward, HEs,MSEs = Train_DQN(N)
end                                = time.time()

print("Total time taken for training the DQN by " +str(N) +" episodes :    ", (end - start),'seconds')
###################################################
############# Plotting the Result #################
###################################################
plot_result1(NoEpi, Cumulative_reward)
plot_result2(HEs,MSEs)
############################################
######### End of DQN Training ##############
############################################


