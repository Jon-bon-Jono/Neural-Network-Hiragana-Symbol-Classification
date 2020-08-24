#By Jonathan Williams
"""
3 models
NetLin - linear function
NetFull - 2 layer fully connected
NetConv - convolutional neural network with 2 convolutional layers and 1 fully connected layer
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    def __init__(self):
        super(NetLin, self).__init__()
        self.in_to_out = torch.nn.Linear(28*28, 28*28)

    def forward(self, x):
        input = x.view(-1, 28*28)
        x = self.in_to_out(input)
        output = torch.log_softmax(x, dim=1)
        return output


class NetFull(nn.Module):
    def __init__(self):
        super(NetFull, self).__init__()
        self.in_to_hu1 = torch.nn.Linear(28*28, 90)
        self.hu1_to_out = torch.nn.Linear(90, 10)

    def forward(self, x):
        row_input = x.view(-1, 28*28)
        hu1_out = torch.tanh(self.in_to_hu1(row_input))
        output = self.hu1_to_out(hu1_out)
        output_probs = torch.log_softmax(output, dim=0)
        return output_probs

class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1,10,8, padding=1)
        self.conv2 = nn.Conv2d(10,27,8, padding=2)
        #self.pool = nn.MaxPool2d(5,5)
        self.fc1 = nn.Linear(27*20*20, 10) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = self.pool(x)

        x = x.view(-1,27*20*20)
        x = self.fc1(x)
        x = torch.log_softmax(x, dim=1)
        return x