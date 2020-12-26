import torch

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.e1 = torch.nn.Linear(784, 400)
        self.e2 = torch.nn.Linear(400, 256)
        self.e3 = torch.nn.Linear(256, 100)

        #self.lv = torch.nn.Linear(100, 64)
        
        self.d1 = torch.nn.Linear(100, 256)
        self.d2 = torch.nn.Linear(256, 400)
        self.d3 = torch.nn.Linear(400, 784)

        self.output_layer = torch.nn.Linear(784, 10) 

    def forward(self, x):

        x = torch.nn.functional.relu(self.e1(x))
        x = torch.nn.functional.relu(self.e2(x))
        x = torch.nn.functional.relu(self.e3(x))

        #x = torch.sigmoid(self.lv(x))

        x = torch.nn.functional.relu(self.d1(x))
        x = torch.nn.functional.relu(self.d2(x))
        x = torch.nn.functional.relu(self.d3(x))
        
        x = self.output_layer(x)
        
        return torch.nn.Functional.log_softmax(x)

if __name__ == "__main__":
    ae = AutoEncoder()
    print(ae)
