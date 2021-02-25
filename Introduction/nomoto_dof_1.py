"""
Created on Sun Jan 17 09:18:42 2021

@author: Sivaraman Sivaraj, Suresh Rajendran
"""

import numpy as np
import math,time,sys
import scipy as sp
import matplotlib.pyplot as plt

def r_dot(delta,r,K,T):
    rt = ((K*delta)-r)/T
    return rt


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
    delta,r,K,T,ra_change = round(delta,6),round(r,6),K,T,round(ra_change,6)
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
    next state : [u,r,x,y,psi,delta,n]
    
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
    
    D = 9.86 # propellar diameter for calculating RPM
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
    
    delta_psi = r_nxt*t_i # calculating change in heading angle
    
    psi_nxt  =  psi_t + delta_psi
    
    ############################################
    ######## Updating to next state ############
    ############################################
    u_next = u_t
    r_next = r_nxt
    x_next = x_t + u_t*(np.sin(psi_nxt))*t_i
    y_next = y_t + u_t*(np.cos(psi_nxt))*t_i
    psi_next = psi_nxt
    delta_next = delta_t + ra_change
    n_next = int(60*u_t/(np.pi*D))
    
    return [u_next,r_next,x_next,y_next,psi_next,delta_next,n_next]



############################################
######## to check nomoto_dof_1 #############
############################################
# ip = [7.75,0.1,0,0,1,0.1,90]
# # op = activate(ip,0.1,-0.5)
# print(op)

# x = list()
# y =[]
# for _ in range(200):
#     temp = activate(ip,0.5,0.5)
#     ip = temp
#     x.append(temp[4])
#     y.append(temp[5])
#     print(temp)

# plt.plot(x,y)
# plt.plot(y)
############################################
################ End #######################
############################################


############################################
############### second order ###############
############################################
# m  = 0.00792
# Iz = 0.000456
# Yv,Yr,Yd = -0.0116,0.00242,0 
# Nv,Nr,Nd = -0.0038545,-0.00222, 0.000213
# Yv_dot = 0.00003# dummy values
# Nr_dot = 0.00003# dummy values

# C = (Yv*Nr) - (Nv*Yr)    
# K = ((Nv*Yd)-(Nd*Yv))/C
# T1 = -((m-Yv_dot)*Nr + (Iz-Nr_dot)*Yr)/C
# T2 = (m-Yv_dot)*Nr/((Nv*Yd)-(Nd*Yv))
# T = T1-T2
############################################
############### second order ###############
############################################


