import numpy as np


def get(ip,op,y_e,HE,HE_old,G,R_1,i_episode):
    """
    Parameters
    ----------
    ip          : input state
    op          : output state
    y_e         : cross track error
    HE          : heading error
    HE_old      : previous heading error
    G           : goal
    R_1         : reward one index number for starting angle greater than 75
    i_episode   : episode number

    Returns
    -------
    Rf : Reward

    """
    x_d0 = np.square(G[0] - ip[3])
    y_d0 = np.square(G[1] - ip[4])
    D0 = np.sqrt(x_d0+y_d0)
    
    x_d1 = np.square(G[0] - op[3])
    y_d1 = np.square(G[1] - op[4])
    D1 = np.sqrt(x_d1 + y_d1)
    R = 0
    
    if  D0 - D1 >=  0 :
        if abs(y_e) <= 0.5:
                R =100
        elif 0.5 < abs(y_e) <= 1.0:
            R = 20
        else:
            c0 = abs(y_e)/63
            c1 = 1 - c0
            R  = 20 * c1 
        
    if D0 - D1 < 0 and abs(HE) < abs(HE_old):
        R = 3
        
    
    if abs(op[5]) >= (1.7*np.pi): #270 degree, for the case of heading action only
        R = -0.5
    if abs(y_e) > 63:
        R = -0.5
    ################################
    ########### Assertion ##########
    ################################
   
    Rf = np.array([R])
        
    return Rf


########################################
############# To check #################
########################################
# ip = [7.75,0,0,15,15,0]
# op = [7.75,0,0,16,16,0]
# G = [300,300]
# ss = get(ip,op,0,0,0,G,0,15)
# print(ss)
########################################
########################################
