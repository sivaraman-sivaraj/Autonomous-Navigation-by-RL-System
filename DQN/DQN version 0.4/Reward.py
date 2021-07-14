import numpy as np

def get(ip,op,y_e,HE,G):
    
    x_d0 = np.square(G[0] - ip[3])
    y_d0 = np.square(G[1] - ip[4])
    D0 = np.sqrt(x_d0+y_d0)
    
    x_d1 = np.square(G[0] - op[3])
    y_d1 = np.square(G[1] - op[4])
    D1 = np.sqrt(x_d1 + y_d1)
    
    if D0 - D1 >=  0 :
        if abs(HE) <= (np.pi/2):
            if abs(y_e) <= 0.5:
                R =100
            elif 0.5 < abs(y_e) <= 1.0:
                R = 10
            else:
                c0 = abs(y_e)/35
                c1 = 1 - c0
                R  = 10 * c1 
        elif abs(HE) > (np.pi/2):
            R = -0.5
    else:
        R = -0.5
        
    Rf = R*y_e/y_e # just to 
        
    return Rf

########################################
############# To check #################
########################################
ip = [7.75,0,0,15,15,0]
op = [7.75,0,0,16,16,0]
G = [300,300]
ss = get(ip,op,8.6,1,G)
print(ss)