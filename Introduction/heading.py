"""
Created on Wed Jan 20 22:06:09 2021

@author: Sivaraman Sivaraj
"""

import numpy as np
import matplotlib.pyplot as plt
import time,sys

def activate(grid_size,green_water,land,theta, test = "simple heading"):
    """
    Parameters
    ----------
    grid_size : dynamic region
    green_water : safety region in water
    land : size of land
    theta : Heading Angle need to achieve
    test : simply some degree in heading, it can be modifed as any particular
    like ZigZag and Turning circle test...etc.,

    Returns
    -------
    M : reward function matrix (grid world - matrix)
    prp : Path Reward Points
    -------
    Note : we choose generally 2500 as grid world size with our previous studies (sprial test)
    
    Warning:
        Heading Angle should be in a range of (-90) to (90) degree

    """
    
    gs = grid_size
    water_grid_size = gs - (2*land) - (2*green_water)
    
    M = np.zeros((gs,gs)) #size of the grid
    
    for i in range(land): # left land
        for j in range(gs):
            M[i][j] = -100
    for i1 in range(land): # right land
        for j1 in range(gs):
            t1 = land + water_grid_size+ (2*green_water)
            M[i1+t1][j1] = -100
    for i2 in range(gs-(2*land)): #bottom land
        for j2 in range(land):
            M[i2+land][j2] = -100
    for i3 in range(gs-(2*land)): #top land
        for j3 in range(land):
            t2 = land + water_grid_size+ (2*green_water)
            M[i3+land][t2+j3] = -100
            
    #################################
    #### path reward Declaration ####
    #################################
    a = (theta/180) * np.pi # radian mode
    prp = list() # path reward points
    starting_point = [int(gs/2),land+green_water] #starting point of the ship
    prp.append(starting_point)
    
    if a == 0:
        for e in range(water_grid_size):
            x_t = 0
            temp_scale = water_grid_size + land + green_water
            if (starting_point[0]+x_t) < temp_scale and (e+land+green_water) < temp_scale:
                temp = [int((gs/2)+x_t), e+land+green_water]
                prp.append(temp)
        
        for g in range(len(prp)):
            a1 = prp[g][0]
            a2 = prp[g][1]
            #print(a1,a2)
            M[a1][a2] = 100 # path reward declaration
            for k in range(1,6): # near to path and its rewards
                M[a1-k][a2] = (6-k)*10
                M[a1+k][a2] = (6-k)*10
    
    elif a >0:
        for e in range(water_grid_size):
            x_t = np.ceil(e*(np.tan(a)))
            temp_scale = water_grid_size + land + green_water
            if (starting_point[0]+x_t) < temp_scale and (e+land+green_water) < temp_scale:
                temp = [int((gs/2)+x_t),int(e+land+green_water)]
                prp.append(temp)
        
        for g in range(len(prp)):
            a1 = prp[g][0]
            a2 = prp[g][1]
            #print(a1,a2)
            M[a1][a2] = 100 # path reward declaration
            for k in range(1,6): # near to path and its rewards
                M[a1-k][a2] = (6-k)*10
                M[a1+k][a2] = (6-k)*10
    
    elif a < 0:
        for e in range(water_grid_size):
            x_t = abs(np.floor(e*(np.tan(a))))
            temp_scale = water_grid_size + land + green_water
            temp_scale1 = land + green_water
            if (starting_point[0]-x_t) > temp_scale1 and (e+temp_scale1) < temp_scale:
                temp = [int((gs/2)-x_t),int( temp_scale1+e)]
                prp.append(temp)
        
        for g in range(len(prp)):
            a1 = prp[g][0]
            a2 = prp[g][1]
            #print(a1,a2)
            M[a1][a2] = 100 # path reward declaration
            for k in range(1,6): # near to path and its rewards
                M[a1-k][a2] = (6-k)*10
                M[a1+k][a2] = (6-k)*10
                
    ############################
    #### path reward end #######
    ############################ 
    
    x,y = list(),list()
    for i in range(len(prp)):
        x.append(prp[i][0])
        y.append(prp[i][1])
    
    
    return M,prp,x,y


#################################
####### To evaluate #############
#################################
# M,prp,x,y = activate(3000,250,100,0)
# plt.matshow(M)
# print(prp[0])
# print(prp[-1])
# plt.plot(x,y)
# plt.xlim(0,1400)
# plt.ylim(0,1400)
#################################
####### evaluation end ##########
#################################











    
    
    
