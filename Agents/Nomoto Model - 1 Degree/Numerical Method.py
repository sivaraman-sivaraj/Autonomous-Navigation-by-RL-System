import numpy as np
import matplotlib.pyplot as plt

def rad_to_degree(r):
    return (r/np.pi)*180.0

def degree_to_rad(theta):
    return (theta/180)*np.pi

def f(t,psi,r):
    return r

def g(t,psi,r,K,T,delta):
    val = ((K*delta)-r)/T
    return val 

def RK4_second_order(t_0,psi_0,r_0,K,T,delta):
    """
    Parameters
    ----------
    t_0 : time 
    psi_0 : heading angle at t0
    r_0 : yaw rate at t0
    K : Turning Ability
    T : Quickness
    delta : rudder angle

    Returns
    -------
    psi_1, r+1
    
    """
    h = 0.1 # time interval
    hf = h/2
    K1 = h*f(t_0,psi_0,r_0)
    L1 = h*g(t_0,psi_0,r_0,K,T,delta)
    
    K2 = h * f(t_0+hf, psi_0+(K1/2), r_0+(L1/2))
    L2 = h * g(t_0,psi_0+(K1/2),r_0+(L1/2),K,T,delta)
    
    K3 = h * f(t_0+hf, psi_0+(K2/2), r_0+(L2/2))
    L3 = h * g(t_0,psi_0,r_0,K,T,delta)
    
    K4 = h * f(t_0+h, psi_0+K3, r_0+L3)
    L4 = h * g(t_0,psi_0+K3,r_0+L3,K,T,delta)
    
    del_psi = (1/6) * (K1 + (2*K2) + (2*K3) + K4)
    del_r = (1/6) * (L1 + (2*L2) + (2*L3) + L4)
    psi_1 = psi_0 + del_psi
    r_1 = r_0 + del_r
    
    return psi_1, r_1



def nomoto(ip,delta):
    """
    Parameters
    ----------
    K, T : Nomoto Parameters
    delta : given rudder angle
  
    Returns
    -------
    yaw rate at given time step input
    """
    psi_0,r_0,t_0 = ip[4],ip[1],ip[6]
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
    
    ############################################
    ########## calculating T and K #############
    ############################################
    K = ((Iz-Nr_dot)/((m*xG*u0)-Nr))/10 #0.49 
    T = Nd/((m*xG*u0)-Nr) #3.619
    
    ############################################
    ################ Yaw Rate ##################
    ############################################
    psi_1, r_1 = RK4_second_order(t_0,psi_0,r_0,K,T,delta)
    return psi_1, r_1


def state_update(ip,psi_1,r_1,delta):
    """
    Parameters
    ----------
    ip : [u,r,x,y,psi,delta,t] at state
    r :  yaw rate 

    Returns
    -------
    updated state
    """
    u = ip[0]
    psi_nxt = psi_1
    x,y = ip[2],ip[3]
    r_nxt = r_1
    #####################################
    ###### updating the x,y value #######
    #####################################
    x_nxt = x + u* np.sin(psi_nxt)*0.1
    y_nxt = y + u* np.cos(psi_nxt)*0.1
    delta_nxt = delta
    t = ip[6]+0.1
    #####################################
    ########## End of  Updating #########
    #####################################
    state = [u,r_nxt,x_nxt,y_nxt,psi_nxt,delta_nxt,t]
    return state

    
def activate(ip,delta): #U,delta,psi
    """
    ------------
    Parameters
    ------------
    ip          : [u,r,x,y,psi,delta,t] (current state)
    num         : [0,1,2,3,4  ,5    ,6]
    
    delta       : given rudder angle for next state (t0+1 second)
   
    --------
    Returns
    --------
    next state : [u,r,x,y,psi,delta,n]
    
    u       : Ship Total Velocity - here- surge velocity
    r       : yaw rate in (rad/s)
    (x,y)   : position in local frame
    psi     : heading angle
    delta   : given rudder input (just to plot and reccord)
    t       : time of simulation
    
    -----------------------
    Description
    -----------------------
    Tr' + r = K.𝛿
    
    We can refer MMG model paper for calculating constants T and K
    """
    #################################
    ########### Yaw rate ############
    #################################
    psi_1,r_1 = nomoto(ip,delta)
    ip_next = state_update(ip,psi_1,r_1,delta)
    return ip_next

#####################################################
############# To check ##############################
#####################################################
# ip = [10,0,10,10,0,0,0]
# op = activate(ip,0.61)
# print(op)
#####################################################
#####################################################

#####################################################
############# To check ##############################
#####################################################
ip = [7.75,0,0,0,0,0,0]
print(ip)
data= list()
data.append(ip)

x,y = [ip[2]],[ip[3]]

for i in range(270):
    if i < 1050:
        temp = activate(data[-1],0.61) #-0.61
    else:
        temp = activate(data[-1],0.61)
    data.append(temp)
    x.append(temp[2])
    y.append(temp[3])
    
plt.figure(figsize=(6,6))
plt.plot(x,y,'g',label = "clockwise test ")
plt.xlabel("Transfer (in meters)")
plt.ylabel("Advance (in meters)")
plt.title("KVLCC2 Turning Circle Test")
plt.legend(loc="best")
plt.grid()
plt.show()
######################################################
######################################################
    

    
    
    
    
    
    
    
    
    
    
    



