"""
Created on Tue Jan 12 20:13:29 2021

@author: Sivaraman Sivaraj
"""


import numpy as np
import sys

def activate(x,ui,ti):
    """
    Parameters
    ----------
    x    : the state vector = [ u, v, r, x, y, psi, p, phi, delta, n ]
    ui   : [ delta_c, n_c ]
    #################################################
    for a container ship L = 175 m, where
    
    delta_c : commanded rudder angle   (rad)
    n_c     : commanded shaft velocity (rpm)
    ##################################################    
     u     = surge velocity          (m/s)
     v     = sway velocity           (m/s)
     r     = yaw velocity            (rad/s)
     x     = position in x-direction (m)
     y     = position in y-direction (m)
     psi   = yaw angle               (rad)
     p     = roll velocity           (rad/s)
     phi   = roll angle              (rad)
     delta = actual rudder angle     (rad)
     n     = actual shaft velocity   (rpm)
    ##################################################
    Returns
    -------
    xdot : time derivative of the state vector
    U    : speed U in m/s
    
    ##################################################
    Reference:  Son og Nomoto (1982). On the Coupled Motion of Steering and 
                Rolling of a High Speed Container Ship, Naval Architect of Ocean Engineering,
                20: 73-83. From J.S.N.A. , Japan, Vol. 150, 1981.
 
    Author:    Trygve Lauvdal
    Date:      12th May 1994
    Revisions: 18th July 2001 (Thor I. Fossen): added output U, changed order of x-vector
                20th July 2001 (Thor I. Fossen): changed my = 0.000238 to my = 0.007049

    """
    
    assert len(x) == 10,"Length should be 10 for state vector."
    assert len(ui) ==2,"u-vector length should be 2"
    
    #################################
    ### Normalization variables #####
    #################################
    L = 175                     #length of ship (m)
    U = np.sqrt((x[0]**2) + (x[1]**2))   #service speed (m/s)
    
    #################################
    ###### Check service speed ######
    #################################
    if U <= 0:
        print("Numerical Error, The ship must have speed greater than zero.")
        sys.exit()
    if x[9] <= 0:
        print("Numerical Error, The propeller rpm must be greater than zero")
        sys.exit()
        
    delta_max  = 10             # max rudder angle (deg)
    Ddelta_max = 5              # max rudder rate (deg/s)
    n_max      = 160            # max shaft velocity (rpm)
    
    ###################################################
    ###### Non-dimensional states and inputs ##########
    ###################################################
    delta_c = ui[0]
    n_c     = ui[1]/60*L/U  

    u = x[0]/U
    v = x[1]/U  
    p = x[6]*L/U
    r = x[2]*L/U 
    phi = x[7]
    psi = x[5]
    delta = x[8]
    n   = x[9]/60*L/U
    
    ######################################################################
    ####### Parameters, hydrodynamic derivatives and main dimensions #####
    ######################################################################
    
    m  = 0.00792
    mx     = 0.000238
    my = 0.007049
    Ix = 0.0000176
    alphay = 0.05
    lx = 0.0313
    ly = 0.0313
    Ix = 0.0000176
    Iz = 0.000456
    Jx = 0.0000034
    Jz = 0.000419
    xG = 0
    
    B, dF, g   = 25.40,8.00,9.81
    dA,d,nabla   = 9.00,8.50,21222 
    KM,KB,AR   = 10.39,4.6154,33.0376
    Delta,D,GM = 1.8219,6.533,0.3/L
    rho,t,T   = 1025,0.175,0.0005
    
    W = rho*g*nabla/(rho*(L**2)*(U**2)/2)
    
    Xuu      = -0.0004226
    Xvr    = -0.00311
    Xrr      = 0.00020
    Xphiphi  = -0.00020
    Xvv    = -0.00386

    Kv,Kr,Kp      =  0.0003026, -0.000063,-0.0000075 
    Kphi, Kvvv, Krrr    = -0.000021,0.002843, -0.0000462 
    Kvvr,Kvrr, Kvvphi    = -0.000588, 0.0010565, -0.0012012 
    Kvphiphi,Krrphi, Krphiphi = -0.0000793,-0.000243,0.00003569
    
    Yv,Yr,Yp = -0.0116,0.00242,0 
    Yphi,Yvvv,Yrrr = -0.000063, -0.109,0.00177 
    Yvvr,Yvrr, Yvvphi  =  0.0214,-0.0405, 0.04605
    Yvphiphi,Yrrphi,Yrphiphi =  0.00304,0.009325,-0.001368
    
    Nv,Nr,Np = -0.0038545,-0.00222, 0.000213 
    Nphi,Nvvv,Nrrr = -0.0001424,0.001492,-0.00229 
    Nvvr,Nvrr, Nvvphi   = -0.0424, 0.00156,-0.019058 
    Nvphiphi, Nrrphi, Nrphiphi  = -0.0053766,-0.0038592,0.0024195
    
    kk,epsilon,xR =  0.631,0.921,-0.5
    wp,tau,xp =  0.184,1.09,-0.526 
    cpv,cpr,ga =  0.0,0.0,0.088 
    cRr,cRrrr,cRrrv    = -0.156,-0.275,1.96 
    cRX,aH,zR   =  0.71,0.237,0.033
    xH     = -0.48
    
    ##############################################################
    ############# Masses and moments of inertia ##################
    ##############################################################
    
    m11 = (m+mx)
    m22 = (m+my)
    m32 = -my*ly
    m42 = my*alphay
    m33 = (Ix+Jx)
    m44 = (Iz+Jz)
    
    #############################################################
    ###### Rudder saturation and dynamics #######################
    #############################################################
    
    if abs(delta_c) >= delta_max*np.pi/180:
        delta_c = np.sign(delta_c) * delta_max*np.pi/180.0
        
    delta_dot = delta_c - delta
    
    if abs(delta_dot) >= Ddelta_max*np.pi/180:
        delta_dot = np.sign(delta_dot)*Ddelta_max*np.pi/180.0
    
    #############################################################
    ###### Shaft velocity saturation and dynamics ###############
    #############################################################
    n_c = n_c*U/L
    n   = n*U/L
    
    if abs(n_c) >= n_max/60:
        n_c = np.sign(n_c)*n_max/60
    
    if n > 0.3:
        Tm=5.65/n
    else:
        Tm=18.83
        
    n_dot = 1/Tm*(n_c-n)*60;
    #############################################################
    ############ Calculation of state derivatives ###############
    #############################################################
    
    vR = ga*v + cRr*r + cRrrr*(r**3) + (cRrrv*(r**2)*v)
    uP     = np.cos(v)*((1 - wp) + tau*((v + xp*r)**2 + cpv*v + cpr*r))
    J     = uP*U/(n*D)
    KT     = 0.527 - 0.455*J
    uR     = uP*epsilon*(np.sqrt(1 + 8*kk*KT/(np.pi*(J**2))))
    alphaR = delta + np.arctan(vR/uR)
    FN     = - ((6.13*Delta)/(Delta + 2.25))*(AR/L**2)*(uR**2 + vR**2)*(np.sin(alphaR))
    T      = 2*rho*(D**4)/((U**2)*(L**2)*rho)*KT*n*abs(n)
    #############################################################
    #################### Forces and moments #####################
    #############################################################
    
    X    = Xuu*(u**2) + (1-t)*T + Xvr*v*r + Xvv*(v**2) + Xrr*(r**2) + Xphiphi*(phi**2) + cRX*FN*np.sin(delta) + (m + my)*v*r
    
    Y    = Yv*v + Yr*r + Yp*p + Yphi*phi + Yvvv*(v**3) + Yrrr*(r**3) + Yvvr*(v**2)*r + Yvrr*v*(r**2) + Yvvphi*(v**2)*phi + Yvphiphi*v*(phi**2) + Yrrphi*(r**2)*phi + Yrphiphi*r*(phi**2) + (1 + aH)*FN*np.cos(delta) - (m + mx)*u*r

    K    = Kv*v + Kr*r + Kp*p + Kphi*phi + Kvvv*(v**3) + Krrr*(r**3) + Kvvr*(v**2)*r + Kvrr*v*(r**2) + Kvvphi*(v**2)*phi + Kvphiphi*v*(phi**2) + Krrphi*(r**2)*phi + Krphiphi*r*(phi**2) - (1 + aH)*zR*FN*np.cos(delta) + mx*lx*u*r - W*GM*phi

    N    = Nv*v + Nr*r + Np*p + Nphi*phi + Nvvv*(v**3) + Nrrr*(r**3) + Nvvr*(v**2)*r + Nvrr*v*(r**2) + Nvvphi*(v**2)*phi + Nvphiphi*v*(phi**2) + Nrrphi*(r**2)*phi + Nrphiphi*r*(phi**2) + (xR + aH*xH)*FN*np.cos(delta)
 
    #############################################################
    ################# state derivatives - xdot ##################
    #############################################################
    
    detM = m22*m33*m44-(m32**2)*m44-(m42**2)*m33
    
    xdot = [  X*((U**2)/L)/m11,
            -((-m33*m44*Y+m32*m44*K+m42*m33*N)/detM)*((U**2)/L),
            ((-m42*m33*Y+m32*m42*K+N*m22*m33-N*(m32**2))/detM)*((U**2)/(L**2)),
            (np.cos(psi)*u-np.sin(psi)*np.cos(phi)*v)*U,
            (np.sin(psi)*u+np.cos(psi)*np.cos(phi)*v)*U,
            np.cos(phi)*r*(U/L),
            ((-m32*m44*Y+K*m22*m44-K*(m42**2)+m32*m42*N)/detM)*((U**2)/(L**2)),
            p*(U/L),
            delta_dot,
            n_dot      ]
    
    return np.array(xdot)*ti,U







#################################################
############### to check start###################
#################################################
# st = np.array([2,0,0,0,0,0,0,0,0,10])
# ui=[0.2,100]
# ui1 = [0.2,100]
# x,y = [0],[0]
# for i in range(2000):
#     print(i)
#     if i < 430:
#         xd_temp,U = activate(st, ui,0.2)
#     else:
#         xd_temp,U = activate(st, ui1,0.2)
#     temp = st + np.array(xd_temp)
#     x.append(temp[3])
#     y.append(temp[4])
#     st = temp
# import matplotlib.pyplot as plt
# plt.plot(y,x) 

#################################################
############### to check End  ###################
#################################################
    


