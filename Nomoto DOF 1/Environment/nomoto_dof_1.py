import numpy as np
import matplotlib.pyplot as plt

def r_dot(delta,r,K,T): #gradient value
    return ((K*delta)-r)/T

def rad_to_degree(r):
    return (r/np.pi)*180.0

def degree_to_rad(theta):
    return (theta/180)*np.pi

def RK4(delta,r,K,T,ra_change):
    """
    Parameters
    ----------
    delta : rudder angle(in radian)
    r : yaw rate
    K,T : Nomoto model constant
    ra_change : rudder angle change(our input for next state)(in radian)

    Returns
    -------
    yaw_rate(at next time step)(rad/s)

    """
    delta,r,K,T,ra_change =delta,r,K,T,round(ra_change,6)
    f1 = ra_change*r_dot(delta,r,K,T)
    f2 = ra_change*r_dot(delta+(ra_change/2), r+(f1/2), K,T)
    f3 = ra_change*r_dot(delta+(ra_change/2), r+(f2/2), K,T)
    f4 = ra_change*r_dot(delta+ra_change, r+f3, K,T)
    
    del_r = (f1 + (2*f2)+ (2*f3)+f4)/6
    
    r_next = r + del_r
    
    return r_next

############################################
######## to Runge Kutta method #############
############################################
# print(rad_to_degree(0.020207889638299257))
# print(degree_to_rad(45))
############################################
######## to Runge Kutta method #############
############################################
# d = RK4(0.9,0.7,1,1,0.2)
# print(d)
############################################
################ End #######################
############################################



def activate(ip,t_i,ra_change): #U,delta,psi
    """
    ------------
    Parameters
    ------------
    ip          : [u,r,x,y,psi,delta,n] (current state)
    num         : [0,1,2,3,4  ,5    ,6]
    
    ra_change   : given rudder angle change for next state (t0+1 second)
    t_i         : time interval
    
    --------
    Returns
    --------
    time derivative : [u,r,x,y,psi,delta,n]
    
    u       : Ship Total Velocity - here- surge velocity
    r       : yaw rate in (rad/s)
    (x,y)   : position in local frame
    psi     : heading angle
    delta   : given rudder input (just to plot and reccord)
    n       : propeller RPM
    
    -----------------------
    Description
    -----------------------
    Tr' + r = K.𝛿
    
    We can refer MMG model paper for calculating constants T and K
    """
    ############################################
    ##### Parameters for calculating T and K ###
    ############################################
    Iz = 1.99*(10**12)
    m = 3.126*(10**8)
    u0 = 7.75
    xG = 11.2
    Nd = 5.8*(10**11)
    Nr = -1.3309*(10**11)
    Nr_dot = 1.199*(10**12)
    D = 9.86 # propellar diameter for calculating RPM (pi*D*N)/60
    ############################################
    ########## calculating T and K #############
    ############################################
    K = (Iz-Nr_dot)/((m*xG*u0)-Nr)
    T = Nd/((m*xG*u0)-Nr)
    ############################################
    ######## RK Method for finding Psi #########
    ############################################
    u_t = ip[0]
    r_t = ip[1]
    x_t = ip[2]
    y_t = ip[3]
    psi_t = ip[4]
    delta_t = ip[5]
    n_t = ip[6]
    
    
    r_nxt = RK4(delta_t,r_t,K,T,ra_change) # we are using RK 4 method for finding next gradient
    
    delta_psi = (r_nxt-r_t)*t_i # calculating change in heading angle
   
    psi_nxt  =  psi_t + delta_psi
    
    ############################################
    ######## Updating to next state ############
    ############################################
    u_next = u_t
    r_next = delta_psi/t_i
    if ra_change >= 0:
        x_next = x_t + u_t*(np.sin(delta_psi))*t_i
    elif ra_change < 0:
        x_next = x_t - u_t*(np.sin(delta_psi))*t_i
    y_next = y_t + u_t*(np.cos(delta_psi))*t_i
    psi_next = psi_nxt
    delta_next = delta_t + ra_change
    n_next = int(60*u_t/(np.pi*D))
    
    return [u_next,r_next,x_next,y_next,psi_next,delta_next,n_next]


######################################################
############## To check ##############################
######################################################
# ip = [10,1,0,0,23,0,15]
# op = activate(ip,0.5,0.5)
# print(op)
######################################################
######################################################



