import torch
import torch.nn as nn
import numpy as np


class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.ipL = nn.Linear(8,128)   # has pre trained weight
        self.HL1 = nn.Linear(128,128) # has pre trained weight
        self.HL2 = nn.Linear(128,128) # has pre trained weight
        self.opL = nn.Linear(128,5)   # output layer
        
    def forward(self, x):
        x = torch.sigmoid(self.ipL(x))   # sigmiodal,relu,.etc.,
        x = torch.sigmoid(self.HL1(x))
        x = torch.sigmoid(self.HL2(x))
        x = self.opL(x)
        return x
    
policy_net              = FFNN()
policy_net.load_state_dict(torch.load("W9250.pt"))

def select_action(state):
    actions_set        = {'0':-35,'1':-20,'2':0,'3':20,'4':35}
    # actions_set        = {'0':-35,'1':-15,'2':-5,'3':-1,'4':0, '5': 1 ,'6': 5 ,'7': 15 ,'8' : 35 }
    
    with torch.no_grad():
        temp  = state.detach().tolist()
        op    = policy_net(torch.tensor(temp))
        return actions_set[str(np.argmax(op.detach().numpy()))]
    

#####################################
############# for example ###########
#####################################
ip = torch.tensor([1.2,0,0,0,0,0,0,0])
op = select_action(ip) 
print("The action for current state is ", op, "rudder angle")
#####################################
#####################################









