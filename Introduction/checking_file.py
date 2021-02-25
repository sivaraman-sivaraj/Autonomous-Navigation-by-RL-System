import nomoto_dof_1
import numpy as np
import matplotlib.pyplot as plt

ip = [10,0,1600,350,0,0,120]

data = [ip]

x,y=[],[]
for i in range(100):
    a = np.random.choice([0.05,-0.08,0,-0.01,-0.05])
    temp = nomoto_dof_1.activate(data[-1],0.1,-0.03)
    data.append(temp)
    x.append(temp[2])
    y.append(temp[3])
    print(i)
    
plt.plot(y,x)
