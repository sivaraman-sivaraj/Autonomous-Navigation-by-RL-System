import torch
import torch.nn as nn
import numpy as np


class MLFFNN(nn.Module):
    def __init__(self):
        super(MLFFNN, self).__init__()
        self.ipL = nn.Linear(7,128)   # has pre trained weight
        self.HL1 = nn.Linear(128,128) # has pre trained weight
        self.HL2 = nn.Linear(128,128) # has pre trained weight
        self.opL = nn.Linear(128,23)  # output layer
        
    def forward(self, x):
        x = torch.tanh(self.ipL(x))#sigmiodal,relu,.etc.,
        x = torch.tanh(self.HL1(x))
        x = torch.tanh(self.HL2(x))
        x = self.opL(x)
        return x

##################################
########## To Check ##############
##################################
# net = MLFFNN()
# print(net)
# ip = torch.tensor([7.75,0,0,0,0,0,0])
# op = net(ip)
# print(op)
# print( net(ip).max())
# print(np.argmax(op.detach().numpy()))
# sss = net.parameters()
# print(sss)
###################################
########## End ####################
###################################
