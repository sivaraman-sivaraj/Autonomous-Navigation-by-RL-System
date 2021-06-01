import numpy as np
import matplotlib.pyplot as plt
import math

def activate(x,ui,U0 = 7.7175):
    """
    Parameters
    ----------
    x : [ u v r x y psi delta]
    ui : commanded rudder angle (rad)
    U0 : nominal speed (optionally). Default value is U0 = 7.7175 m/s = 15 knots.

    Returns
    -------
    xdot : Time derivative of the state vector
    U : speed
    
    
    Descriptions:
        for the Mariner class vessel L = 160.93 m, where
            u     = pertubed surge velocity about Uo (m/s)
            v     = pertubed sway velocity about zero (m/s)
            r     = pertubed yaw velocity about zero (rad/s)
            x     = position in x-direction (m)
            y     = position in y-direction (m)
            psi   = pertubed yaw angle about zero (rad)
            delta = actual rudder angle (rad)
            
    
    Reference: M.S. Chislett and J. Stroem-Tejsen (1965). Planar Motion Mechanism Tests 
               and Full-Scale Steering and Maneuvering Predictions for a Mariner Class Vessel,
               Technical Report Hy-5, Hydro- and Aerodynamics Laboratory, Lyngby, Denmark.
               
    
     Author:    Trygve Lauvdal
     Date:      12th May 1994
     Revisions: 19th July 2001 (Thor I. Fossen): added input/ouput U0 and U, changed order of x-vector
                20th July 2001 (Thor I. Fossen): replaced inertia matrix with correct values
                11th July 2003 (Thor I. Fossen): max rudder is changed from 30 deg to 40
                                deg to satisfy IMO regulations for 35 deg rudder execute

    

    """
    # Normalization variables
    L = 160.93
    u1 = U0+x[0]
    U = np.sqrt((u1**2) +(x[1]**2))
    
    #Non-dimensional states and inputs
    delta_c = -ui   #% delta_c = -ui such that positive delta_c -> positive r
    
    u     = x[0]/U   
    v     = x[1]/U  
    r     = x[2]*L/U 
    psi   = x[5] 
    delta = x[6] 
    
    #Parameters, hydrodynamic derivatives and main dimensions
    
    delta_max  = 40          #max rudder angle      (deg)
    Ddelta_max = 5           #max rudder derivative (deg/s)
    
    m  = 798e-5
    Iz = 39.2e-5
    xG = -0.023
    
    [Xudot,Xu,Xuu,Xuuu,Xvv,Xrr,Xdd,Xudd,Xrv,Xvd,Xuvd] = [-42e-5,-184e-5,-110e-5,-215e-5,-899e-5,
                                                         18e-5,-95e-5,-190e-5,798e-5,93e-5,93e-5]
    
    [Yvdot,Yrdot,Yv,Yr,Yvvv,Yvvr,Yvu,Yru,
     Yd,Yddd,Yud,Yuud,Yvdd,Yvvd,Y0,Y0u,Y0uu ]=[-748e-5,-9.354e-5,-1160e-5,-499e-5,-8078e-5,15356e-5,
                                               -1160e-5,-499e-5,278e-5,-90e-5,556e-5,278e-5,-4e-5,
                                               1190e-5,-4e-5,-8e-5,-4e-5]
                                               
    [Nvdot,Nrdot,Nv,Nr,Nvvv,Nvvr,Nvu,Nru,
     Nd,Nddd,Nud,Nuud,Nvdd,Nvvd,N0,N0u,N0uu]=[4.646e-5,-43.8e-5,-264e-5,-166e-5,1636e-5,
                                              -5483e-5,-264e-5,-166e-5,-139e-5,45e-5,-278e-5,
                                              -139e-5, 13e-5,-489e-5,3e-5,6e-5,3e-5]
    
    # Masses and moments of inertia
    m11 = m-Xudot
    m22 = m-Yvdot
    m23 = m*xG-Yrdot
    m32 = m*xG-Nvdot
    m33 = Iz-Nrdot

    #Rudder saturation and dynamics
    if abs(delta_c) >= (delta_max*np.pi)/180:
        delta_c = np.sign(delta_c)*delta_max*np.pi/180
        
    delta_dot = delta_c - delta
    
    if abs(delta_dot) >= Ddelta_max*np.pi/180:
        delta_dot = np.sign(delta_dot)*Ddelta_max*np.pi/180
        
    # Forces and Moments
    
    X = Xu*u + Xuu*(u**2) + Xuuu*(u**3) + Xvv*(v**2) + Xrr*(r**2) + Xrv*r*v + Xdd*(delta**2)+\
        Xudd*u*(delta**2) + Xvd*v*delta + Xuvd*u*v*delta
        
    Y = Yv*v + Yr*r + Yvvv*(v**3) + Yvvr*(v**2)*r + Yvu*v*u + Yru*r*u + Yd*delta+\
        Yddd*(delta**3) + Yud*u*delta + Yuud*(u**2)*delta + Yvdd*v*(delta**2) +\
        Yvvd*(v**2)*delta + (Y0 + Y0u*u + Y0uu*(u**2))
        
    N = Nv*v + Nr*r + Nvvv*(v**3) + Nvvr*(v**2)*r + Nvu*v*u + Nru*r*u + Nd*delta +\
        Nddd*(delta**3) + Nud*u*delta + Nuud*(u**2)*delta + Nvdd*v*(delta**2) +\
            Nvvd*(v**2)*delta + (N0 + N0u*u + N0uu*(u**2))
            
    # Dimensional state derivative
    detM22 = m22*m33-m23*m32
    
    xdot = [X*((U**2)/L)/m11 ,
            -(-m33*Y+m23*N)*((U**2)/L)/detM22,
            (-m32*Y+m22*N)*((U**2)/(L**2))/detM22,
            (np.cos(psi)*(U0/U+u)-np.sin(psi)*v)*U,
            (np.sin(psi)*(U0/U+u)+np.cos(psi)*v)*U,
            r*(U/L),
            delta_dot]
    
    return xdot,U

############################
###### To Check ############
############################
# x =np.array([0.8,0.5,0.3,100,100,40,30])
# d = [0.8,0.5,0.3,100,100,40,30]
# ui = -30
            
# aa,b = activate(x, ui)
# aa1,b1 = activate(d, ui)

# a = np.array(aa)
# print(a.T)
# print(a.shape)
# print(b)
        
# print(aa1)
# print(b1)
############################
############################