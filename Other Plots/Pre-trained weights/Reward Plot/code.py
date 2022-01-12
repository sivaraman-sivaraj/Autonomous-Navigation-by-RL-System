import numpy as np
import matplotlib.pyplot as plt


Ar = np.load("R55.npy")[:350]
Br = np.load("R105.npy")[:350]
Cr = np.load("R90.npy")[:500]

A,B,C = np.zeros(len(Ar)),np.zeros(len(Br)),np.zeros(len(Cr))

for t in range(len(Ar)):
    A[t] = np.mean(Ar[max(0, t-200):(t+1)])
    
for t in range(len(Br)):
    B[t] = np.mean(Br[max(0, t-200):(t+1)])
    
for t in range(len(Cr)):
    C[t] = np.mean(Cr[max(0, t-200):(t+1)])


plt.figure(figsize=(9,6))
plt.plot(Ar[:340],color="g",alpha=0.3)
plt.plot(A,color="g", label="-45$^\circ$ to -55$^\circ$ learning($\epsilon$ = 0.35 $\longrightarrow$ 0.2)")
plt.plot(Br[:350],color="b",alpha=0.1)
plt.plot(B,color="b",label="-90$^\circ$ to -105$^\circ$ learning ($\epsilon$ = 0.4 $\longrightarrow$ 0.2)")
plt.plot(Cr,color="crimson",alpha=0.2)
plt.plot(C,color="crimson",label="45$^\circ$ to 90$^\circ$ learning($\epsilon$ = 0.6 $\longrightarrow$ 0.2)")
plt.legend(loc="best")
plt.xlabel("Number of Episodes")
plt.ylabel("Reward")
plt.title("Reward for Learning in Pre-trained weights")
plt.grid()
plt.savefig("EPW_R.jpg",dpi = 480)



