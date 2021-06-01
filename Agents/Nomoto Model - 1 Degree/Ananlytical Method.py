import numpy as np
import matplotlib.pyplot as plt

def rad_to_degree(r):
    return (r/np.pi)*180.0

def degree_to_rad(theta):
    return (theta/180)*np.pi

def nomoto(delta_0,t):
    """
    Parameters
    ----------
    K, T : Nomoto Parameters
    delta_0 : intial delta value
    t : time step input

    Returns
    -------
    yaw rate at given time step input
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
    
    # Iz = 1.99*(10**12)
    # m = 3.126*(10**8)
    # u0 = 7.75
    # xG = 11.425
    # Nd = 1.578*(10**9)
    # Nr = -1.6814*(10**9)
    # Nr_dot = 1.2155*(10**12)
    
    ############################################
    ########## calculating T and K #############
    ############################################
    # K = ((Iz-Nr_dot)/((m*xG*u0)-Nr))/10
    # T = Nd/((m*xG*u0)-Nr)
    
    # K,T = 0.49,3.6
    K,T = 0.9822,24.33
    # print(K,T)
    ############################################
    ################ Yaw Rate ##################
    ############################################
    r = K*delta_0*(1-np.exp(-t/T))
    return r

def state_update(ip,r,delta_0):
    """
    Parameters
    ----------
    ip : [u,r,x,y,psi,delta,n,pivot] at state
    r :  yaw rate 

    Returns
    -------
    updated state
    """
    u = ip[0]
    psi = ip[4]
    x,y = ip[2],ip[3]
    r_nxt = r
    #####################################
    ###### updating the x,y value #######
    #####################################
    x_nxt = x + u* np.cos(psi+r)
    y_nxt = y + u* np.sin(psi+r)
    psi_nxt = (psi+r) #% np.pi
    delta_nxt = delta_0
    n = 120 # for further model developements
    if ip[5] == delta_0:
        pivot = ip[7] + 1
    else:
        pivot = 0
    #####################################
    ########## End of  Updating #########
    #####################################
    state = [u,r_nxt,x_nxt,y_nxt,psi_nxt,delta_nxt,n,pivot]
    return state


def activate(ip,delta): #U,delta,psi
    """
    ------------
    Parameters
    ------------
    ip          : [u,r,x,y,psi,delta,n,pivot] (current state)
    num         : [0,1,2,3,4  ,5    ,6,7]
    
    delta       : given rudder angle for next state (t0+1 second)
   
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
    pivot   : time index in Nomoto model (exponential growth rate)
    -----------------------
    Description
    -----------------------
    Tr' + r = K.𝛿
    
    We can refer MMG model paper for calculating constants T and K
    """
    #################################
    ########### Yaw rate ############
    #################################
    t = ip[-1] #looking for nomoto time index
    r = nomoto(delta, t+1)
    ip_next = state_update(ip,r,delta)
    return ip_next
    

######################################################
############## To check ##############################
######################################################
# ip = [10,0,0,0,0,0.61,120,11]
# op = activate(ip,0.61)
# print(op)
######################################################
######################################################

######################################################
############## To check ##############################
######################################################
# ip = [10,1,0,0,23,0,15]
# op = activate(ip,0.5,0.5)
# print(op)

ip = [8.75,0,0,0,0,0,90,0]
print(ip)
data= list()
data.append(ip)

x,y = [ip[2]],[ip[3]]

for i in range(50):
    if i < 110:
        temp = activate(data[-1],0.4) #-0.61
    else:
        temp = activate(data[-1],0.4)
    # temp = update(data[-1],tm_d,-0.1)
    data.append(temp)
    x.append(temp[2])
    y.append(temp[3])
    # print(temp)
    # print(i)
# plt.scatter(x,y)
plt.figure(figsize=(9,6))
plt.plot(y,x,'y',label = "Trajectory of KVLCC2 for 50 step units")
plt.xlabel("Transfer (in meters)")
plt.ylabel("Advance (in meters)")
plt.title("KVLCC2 Ship in Nomoto Model $T$ = 24.33 and $K$= 0.9822 ")
plt.legend(loc="best")
plt.grid()
plt.show()
######################################################
######################################################

    











