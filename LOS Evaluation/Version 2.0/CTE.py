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

def get(ye):
    sigma1 = 1
    sigma2 = 2
    sigma3 = 22 # maximum distance set as 60.1
    conv_factor1 = conversion_factor(0,0,sigma1)
    conv_factor2 = 125#CTE.conversion_factor(0,0,sigma2)
    conv_factor3 = 275#CTE.conversion_factor(0,0,sigma3)
    if ye <= 2 and ye >= -2:
        y_raw = normal_df(ye,0,sigma1)
        reward = y_raw*conv_factor1*1
    elif ye > 2 and ye < 3 :
        y_raw = normal_df(ye,0,sigma2)
        reward = y_raw*conv_factor2
    elif ye >= -3 and ye <-2:
        y_raw = normal_df(ye,0,sigma2)
        reward = y_raw*conv_factor2
    else:
        y_raw = normal_df(ye,0,sigma3)
        reward = y_raw*conv_factor3
    return reward

##############################
##### To Check ###############
##############################
# ss = get(40.2)
# print(ss)
# ab = -40
# X = np.arange(-40,40,0.1)
# Y = []
# N = 40/0.1
# for i in range(2*int(N)):
#     temp = get(ab)
#     Y.append(temp)
#     ab += 0.1
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(X,Y,'g',label = "Range of -40 m to 40 m")
# plt.axhline(y=0.5,color="r")
# plt.axvline(x=40,color="k")
# plt.axvline(x=-40,color="k")
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
