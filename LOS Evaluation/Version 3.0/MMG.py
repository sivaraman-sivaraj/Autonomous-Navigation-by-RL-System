import numpy as np
import matplotlib.pyplot as plt

def rad_to_degree(r):
    return (r/np.pi)*180.0

def degree_to_rad(theta):
    return (theta/180)*np.pi


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
    
    
    # C1,C2       = 2, 1.6 if beta_P > 0 else 1.2 # wake  change characteristic coefficents
    # wp          = (1 - wp0)* (1 + ((1 - np.exp(-C1*abs(beta_P)))*(C2 - 1)))
                                                # Equation 16
    wp          =  wp0*np.exp(-4*(beta_P**2))   # Equation 12
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
      K   = np.zeros(7)
      K0  = np.array([u,v,r,x,y,psi])
    
      K1_ = MMG_Time_Derivative(u,v,r,x,y,psi,delta,rps)
      K1  = h*K1_[:3] 
      k1  = h*K0[:3]
     
      K2_ = MMG_Time_Derivative(u+(K1[0]/2), v+(K1[1]/2), r+(K1[2]/2), x+(k1[0]/2), y+(k1[1]/2), psi+(k1[2]/2),delta,rps)
      K2  = h*K2_[:3]
      k2  = h*(K0[:3] + 0.5*K1)
          
      K3_ = MMG_Time_Derivative(u+(K2[0]/2), v+(K2[1]/2), r+(K2[2]/2),x+(k2[0]/2), y+(k2[1]/2), psi+(k2[2]/2),delta,rps)
      K3  = h*K3_[:3]
      k3  = h*(K0[:3] + 0.5*K2)
     
      K4_ = MMG_Time_Derivative(u+(K3[0]), v+(K3[1]), r+(K3[2]), x+(k3[0]), y+(k3[1]), psi+(k3[2]),delta,rps)
      K4  = h*K4_[:3]
      k4  = h*(K0[:3] + 0.5*K3)
     
     
      K[:3]  =  (1/6) * (K1 + (2*K2) + (2*K3) + K4)
     
      R      = np.array( [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi),  np.cos(psi), 0], [0, 0, 1]])
      del_k  =  ((1/6) * (k1 + (2*k2) + (2*k3) + k4)).reshape(3,1)
      K[3:6] = (R.dot(del_k)).reshape(1,3)
      K[6]   = 0.1 # for numerical stablity
      
      
      nxt_state = np.array(ip) + K
      
      return nxt_state.tolist()



####################################################
############ To check ##############################
####################################################
# def los_mmg_normalizer(theta):
    
#     if theta >= 0 :
#         if 0 <= theta <= np.pi:
#             theta_new  = theta 
#         elif theta > np.pi:
#             theta_new  = theta - (2*np.pi) 
            
#     elif theta < 0:
#         if 0 > theta > -np.pi:
#             theta_new = theta 
#         elif theta < -np.pi:
#             theta_new = theta + (2*np.pi) 
            
#     else : 
#         theta_new = theta 
#     return theta_new

# ip = [1.179,0,0,0,0,0,0]
# print(ip)
# data= list()
# data1 = list()
# data.append(ip)
# data1.append(ip)

# x1,y1 = [ip[2]],[ip[3]]
# x2,y2 = [ip[2]],[ip[3]]
# N = 670
# ss= []

# psi_cw = []
# psi_ccw = []


# for i in range(N):
#     if i < 370:
#         temp = activate(data[-1],0.61,12.05) #-0.61
#     else:
#         temp = activate(data[-1],0.61,12.05)
#     data.append(temp)
#     x1.append(temp[3])
#     y1.append(temp[4])
#     ss.append(los_mmg_normalizer(temp[5]))
# for j in range(N):
#     if j < 70:
#         temp1 = activate(data1[-1],-0.61,12.05) #-0.61
#     else:
#         temp1 = activate(data1[-1],-0.61,12.05)
#     data1.append(temp1)
#     x2.append(temp1[3])
#     y2.append(temp1[4])

# plt.plot(ss)
# plt.title("MMG Model Heading Angle while doing spiral test ")
# plt.grid()


# plt.figure(figsize=(9,6))
# # plt.plot([0,30],[0,30])
# plt.plot(y1,x1,'purple',label = " Counter Clockwise Test ")
# plt.plot(y2,x2,'green',label = " Clockwise Test ")
# plt.scatter(0,0,marker="P",label = "starting point")
# plt.ylabel("Transfer (in meters)")
# plt.xlabel("Advance (in meters)")
# plt.axhline(y=0,color="red",alpha=0.5)
# plt.axvline(x=0,color="red",alpha=0.5)
# plt.title("KVLCC2(L7 MMG Model) Turning Circle Test/ simulation for 1200 seconds ")
# plt.legend(loc="best")
# plt.grid()
# plt.show()
# print(data[-1])
#####################################################
######################################################
         
    
    
    






