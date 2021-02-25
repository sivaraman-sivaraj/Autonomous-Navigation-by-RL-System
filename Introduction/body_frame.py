"""
Created on Thu Jan 21 19:22:13 2021

@author: Sivaraman Sivaraj
"""
import numpy as np

def get(ip,observation):
    """
    Parameters
    ----------
    ip : ship state from nomoto_dof_1 function [u,r,x,y,psi,delta,n] 
    observation : grid world observation (x,y,psi)

    Returns
    -------
    required rudder angle change(in radian)

    """
    psi_current = ip[4]
    a1,b1 = ip[2],ip[3]
    a2,b2 = observation[0],observation[1]
    y = b2-b1
    x = a2-a1
    
    if x != 0:
        angle_required = np.arctan(y/x)
    elif x == 0:
        angle_required = 0
   
    rudder_angel_change = angle_required - psi_current
    
    if np.ceil(y) == 0 or np.ceil(y) < 0:
        rudder_angel_change = 0
    
    return rudder_angel_change
    


######################################
########## to check ##################
######################################
# ip = [7.75,0,100,100,0,35,160]
# observation = (120,112,0)
# r_c = get(ip,observation)
# print(r_c)
######################################
########## end ##################
######################################















