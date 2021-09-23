import numpy as np

def HE_reward(x):
    R = (5/np.pi)*abs(x)
    return R

def CTE(x):
    if abs(x) <= 2.5:
        R = 100
    elif 2.5 < abs(x) <=5:
        R = 50
    else:
        R = 90*np.exp(-0.1*abs(x))
    return R


def HE_ye_monitor(HE,HE_old):
    """
    Returns
    -------
    flag : when HE error get minimizes, flag is True (same can be applied for y_e )

    """
    Flag = False
    if HE == HE_old == 0:
        Flag = True 
    elif abs(HE) < abs(HE_old):
        Flag = True 
    return Flag 

def Goal_monitor(ip,op,G):
    """
    Returns
    -------
    flag : when agent moves towards the goal, flag is True

    """
    Flag = False
    x_d0 = np.square(G[0] - ip[3])
    y_d0 = np.square(G[1] - ip[4])
    D0 = np.sqrt(x_d0+y_d0)
    
    x_d1 = np.square(G[0] - op[3])
    y_d1 = np.square(G[1] - op[4])
    D1 = np.sqrt(x_d1 + y_d1)
    
    if D0 >= D1 :
        Flag = True 
    return Flag
    
    


def get(ip,op,y_e,y_e_old,
        HE,HE_old,
        G):
    """
    Parameters
    ----------
    ip          : input state
    op          : output state
    y_e         : cross track error
    y_e_old     : previous cross track error
    HE          : heading error
    HE_old      : previous heading error
    G           : goal
    T_i         : Tolerence index for 
    i_episode   : episode number

    Returns
    -------
    Rf : Reward

    """
    Flag_CTE = HE_ye_monitor(y_e,y_e_old)
    Flag_HE  = HE_ye_monitor( HE,HE_old)
    Flag_G   = Goal_monitor(ip,op,G)
    
    R = 0
    ### When Moving Towards Goal ###
    if Flag_G ==  True :
        R = CTE(y_e)
    
    ### When Not moving towards Goal ###
    elif Flag_HE == True and Flag_G == Flag_CTE ==  False:
        R = HE_reward(HE)
    
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
# ss = get(ip,op,0,0,0,0,G,0,15)
# print(ss)
########################################
########################################
