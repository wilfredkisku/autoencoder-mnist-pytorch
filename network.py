import torch
#import torch.nn as nn
#import torch.nn.functional as F

#from torch.autograd import Variable

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
#from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.e1 = torch.nn.Linear(784, 392)
        self.e2 = torch.nn.Linear(392, 196)
        self.e3 = torch.nn.Linear(196, 98)

        self.lv = torch.nn.Linear(98, 49)
        
        self.d1 = torch.nn.Linear(49, 98)
        self.d2 = torch.nn.Linear(98, 196)
        self.d3 = torch.nn.Linear(196, 392)

        self.output_layer = torch.nn.Linear(392, 784) 

    def forward(self, x):

        x = torch.nn.functional.relu(self.e1(x))
        x = torch.nn.functional.relu(self.e2(x))
        x = torch.nn.functional.relu(self.e3(x))

        x = torch.sigmoid(self.lv(x))

        x = torch.nn.functional.relu(self.d1(x))
        x = torch.nn.functional.relu(self.d2(x))
        x = torch.nn.functional.relu(self.d3(x))

        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    ae = AutoEncoder()
    print(ae)
