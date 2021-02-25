import numpy as np
import matplotlib.pyplot as plt


def heading(theta, test = "simple heading"):
    """
    Parameters
    ----------
    theta : Heading Angle need to achieve
    test : simply some degree in heading, it can be modifed as any particular
    like ZigZag and Turning circle test...etc.,
    Returns
    -------
    M : reward function matrix (grid world - matrix)
    prp : Path Reward Points
    -------
    Note : we choose 2500 as grid world size with our previous studies (sprial test)
    
    Warning:
        Heading Angle should be in a range of (-90) to (90) degree
    """
    
    M = np.ones((3200,3200)) #size of the grid
    ############################
    #### environment reward ####
    ############################
    for i in range(100):
        for j in range(3200):
            M[i][j] = -100
    for i1 in range(100):
        for j1 in range(3200):
            M[i1+3100][j1] = -100
    for i2 in range(3000):
        for j2 in range(100):
            M[i2+100][j2] = -100
    for i3 in range(3000):
        for j3 in range(100):
            M[i3+100][j3+3100] = -100
    
    #################################
    #### path reward Declaration ####
    #################################
    a = (theta/180) * np.pi # radian mode
    prp = list() # path reward points
    
    if a >0:
        for e in range(2500):
            x_t = np.ceil(e*(np.tan(a)))
            if (1600+x_t) < 2850 and (e+350) < 2850:
                temp = [int(1600+x_t),int( 350+e)]
                prp.append(temp)
        
        for g in range(len(prp)):
            a1 = prp[g][0]
            a2 = prp[g][1]
            #print(a1,a2)
            M[a1][a2] = 100 # path reward declaration
            for k in range(1,6): # near to path and its rewards
                M[a1-k][a2] = (6-k)*100
                M[a1+k][a2] = (6-k)*100
    
    if a < 0:
        for e in range(2500):
            x_t = abs(np.floor(e*(np.tan(a))))
            if (1600-x_t) > 350 and (e+350) < 2850:
                temp = [int(1600-x_t),int( 350+e)]
                prp.append(temp)
        
        for g in range(len(prp)):
            a1 = prp[g][0]
            a2 = prp[g][1]
            #print(a1,a2)
            M[a1][a2] = 100 # path reward declaration
            for k in range(1,6): # near to path and its rewards
                M[a1-k][a2] = (6-k)*100
                M[a1+k][a2] = (6-k)*100
                
    ############################
    #### path reward end #######
    ############################   
        
    return M ,prp





###########################
######## Start ofTest #####
###########################
# a,b = heading(10)
# print(b[0],b[-1])
# plt.matshow(a)
###########################
######## End of Test ######
###########################
