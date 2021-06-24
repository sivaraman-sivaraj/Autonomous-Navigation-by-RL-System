import numpy as np

def normal_df(x,mu,sigma):
        factor1 = (1.0/(sigma*((2*np.pi)**0.5)))
        factor2 = np.e**-(((x-mu)**2)/(2*sigma**2))
        y = factor1 * factor2
        return y
    
def conversion_factor(x,mu,sigma):
    y_max = normal_df(x,mu,sigma)
    factor = 100/y_max
    return factor

def get(ye,HE):
    
    ########################################
    ########## Cross Track Error ###########
    ########################################
    sigma1 = 1
    sigma2 = 2.5
    sigma3 = 10 # maximum distance set as 25
    conv_factor1 = conversion_factor(0,0,sigma1)
    conv_factor2 = 255#CTE.conversion_factor(0,0,sigma2)
    conv_factor3 = 290#CTE.conversion_factor(0,0,sigma3)
    if ye <= 1 and ye >= -1:
        y_raw = normal_df(ye,0,sigma1)
        reward = (y_raw*conv_factor1*1)*0.6
    elif ye > 1 and ye < 4 :
        y_raw = normal_df(ye,0,sigma2)
        reward = (y_raw*conv_factor2)*0.6
    elif ye > -4 and ye <-1:
        y_raw = normal_df(ye,0,sigma2)
        reward = (y_raw*conv_factor2)*0.6
    else:
        y_raw = normal_df(ye,0,sigma3)
        reward = (y_raw*conv_factor3)*0.6
    
        
    ########################################
    ############# Heading Error ############
    ########################################
    if abs(HE) < np.deg2rad(10):
        r2 =  1 - ( (abs(HE) % (np.pi)) / 15)
        reward += (r2*60)

    if abs(ye) > 35:
      reward += -100
      
    if abs(np.rad2deg(HE))>60:
      reward += -10
        
    return reward



##############################
##### To Check ###############
##############################
# ss = get(0,np.deg2rad(12))
# print(ss)
# ab = -40
# X = np.arange(-40,40,0.1)
# Y = []
# N = 40/0.1
# for i in range(2*int(N)):
#     temp = get(ab,20)
#     Y.append(temp)
#     ab += 0.1
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(X,Y,'g',label = "Range of -40 m to 40 m for CTE")
# plt.plot([-15,0,15],[0,40,0],label ="Heading Error(in degree)")
# plt.fill([-15,0,15],[0,40,0])
# plt.axhline(y=0.5,color="r")
# plt.axvline(x=25,color="k")
# plt.axvline(x=-25,color="k")
# plt.grid()
# plt.fill(X, Y, 'g',alpha=0.4)
# plt.xlabel("Cross Track Error")
# plt.ylabel("Reward")
# plt.title("Model Reward Distribution based on CTE")
# plt.legend(loc="best")
# plt.show()
##############################
############ end #############
##############################
