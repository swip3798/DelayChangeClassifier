import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import NETWORK_WIDTH

class BinaryClassification(nn.Module):
    def __init__(self, network_width=NETWORK_WIDTH):
        super(BinaryClassification, self).__init__()        # Number of input features is 11.
        self.layer_1 = nn.Linear(13, network_width*2) 
        self.layer_2 = nn.Linear(network_width*2, int(network_width * 1.5))
        self.layer_3 = nn.Linear(int(network_width * 1.5), network_width)
        self.layer_out = nn.Linear(network_width, 1) 
        
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(network_width*2)
        self.batchnorm2 = nn.BatchNorm1d(int(network_width * 1.5))
        self.batchnorm3 = nn.BatchNorm1d(network_width)
        
    def forward(self, inputs):
        x = self.sigm(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.sigm(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.sigm(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x