import numpy as np
import matplotlib.pyplot as plt
##########################################
########### Image Plotting ###############
##########################################

def plot_result1(NoEpi, Cumulative_reward,phase):
    plt.figure(figsize=(9,12))
    #############################
    plt.subplot(2,1,1)
    N = len(Cumulative_reward)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(Cumulative_reward[max(0, t-100):(t+1)])
    plt.plot(running_avg1,color='r',label="Cumulative Reward")
    plt.title("Phase"+str(phase)+"DQN Training Resuts : Episode Durations & Cumulative Rewards ")
    plt.xlabel("No of Episode")
    plt.ylabel("Reward Unit")
    # plt.ylim(0,110)
    plt.legend(loc="best")
    plt.grid()
    ##############################
    plt.subplot(2,1,2)
    N = len(NoEpi)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(NoEpi[max(0, t-100):(t+1)])
    plt.plot(running_avg2,color='g', label = "Episode Durations")
    plt.xlabel("No of Episode")
    plt.ylabel("Length of Episodes")
    # plt.ylim(0,300)
    plt.legend(loc="best")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig("phase_"+str(phase)+"_pic1.jpg")
    plt.show()
    
    
def plot_result2(HEs,MSE,phase):
    plt.figure(figsize=(9,12))
    ##############################
    plt.subplot(2,1,1)
    N = len(MSE)
    running_avg1 = np.empty(N)
    for t in range(N):
            running_avg1[t] = np.mean(MSE[max(0, t-100):(t+1)])
    plt.plot(running_avg1,color='m',label="Mean Square Error")
    plt.title("Phase "+str(phase)+" DQN Training Resuts : Mean Squared Loss , Heading Error ")
    plt.xlabel("No of Episode")
    plt.ylabel("MSE")
    plt.legend(loc="best")
    plt.grid()
    
    
    plt.subplot(2,1,2)
    N = len(HEs)
    running_avg2 = np.empty(N)
    for t in range(N):
            running_avg2[t] = np.mean(HEs[max(0, t-10):(t+1)])
    plt.plot(running_avg2,color='b', label = "Cumulative Heading Error")
    plt.xlabel("No of Episode")
    plt.ylabel("Heading Error in degree")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("phase_"+str(phase)+"_pic2.jpg")
    plt.show()
