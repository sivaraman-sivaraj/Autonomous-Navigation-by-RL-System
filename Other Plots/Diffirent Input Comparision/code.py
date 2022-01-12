import numpy as np 
import matplotlib.pyplot as plt


A = np.load("2h.npy")
B = np.load("3h.npy")[:2000]
C = np.load("5h.npy")
D = np.load("8h.npy")[:2000]

def Plot(A,B,C,D):
    A1,B1,C1,D1 = [],[],[],[]
    for i in range(len(A)):
        A1.append(np.mean(A[max(0,i-100):i]))
    for j in range(len(B)):
        B1.append(np.mean(B[max(0,j-100):j]))
    for k in range(len(C)):
        C1.append(np.mean(C[max(0,k-100):k]))
    for l in range(len(D)):
        D1.append(np.mean(D[max(0,l-100):l]))
    plt.figure(figsize=(9,6))
    plt.plot(A1,color = "crimson",label="2 inputs(HE, CTE)")
    plt.plot(B1,color = "teal",label="3 inputs(x,y,$\psi$)")
    plt.plot(C1,color="darkviolet",label = "5 inputs(x,y,$\psi$,HE,CTE)")
    plt.plot(D1,color="m",label = "8 inputs(u,v,r,x,y,$\psi$,HE,CTE)")
    plt.xlabel("Number of Episodes") 
    plt.ylabel("Reward Units") 
    plt.legend(loc="best")
    plt.title("Learning Comparison for Different Inputs")
    plt.grid(linestyle="--")
    plt.savefig("DIC.jpg",dpi = 420)
    plt.show()
    
Plot(A,B,C,D)