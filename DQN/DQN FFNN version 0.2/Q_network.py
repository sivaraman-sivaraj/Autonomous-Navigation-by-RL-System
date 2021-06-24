import torch
import torch.nn as nn
import numpy as np


class MLFFNN(nn.Module):
    def __init__(self):
        super(MLFFNN, self).__init__()
        self.ipL = nn.Linear(6,96)  # has pre trained weight
        self.HL1 = nn.Linear(96,96) # has pre trained weight
        self.HL2 = nn.Linear(96,96) # has pre trained weight
        self.opL = nn.Linear(96,9)  # output layer
        
    def forward(self, x):
        x = torch.relu(self.ipL(x))#sigmiodal,relu,.etc.,
        x = torch.relu(self.HL1(x))
        x = torch.relu(self.HL2(x))
        x = torch.softmax(self.opL(x),dim=0)
        return x

##################################
########## To Check ##############
##################################
# net = MLFFNN()
# print(net)
# ip = torch.tensor([7.75,0,0,0,0,0])
# op = net(ip)
# print(op)
# print( net(ip).max())
# print(np.argmax(op.detach().numpy()))
# sss = net.parameters()
# print(sss)
###################################
########## End ####################
###################################
