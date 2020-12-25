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

def show_torch_image(torch_tensor):
    plt.imshow(torch_tensor.numpy().reshape(28, 28), cmap='gray')
    plt.show()

def initialize(trn_x, val_x, trn_y, val_y):
    trn_x_torch = torch.from_numpy(trn_x).type(torch.FloatTensor)
    trn_y_torch = torch.from_numpy(trn_y)

    val_x_torch = torch.from_numpy(val_x).type(torch.FloatTensor)
    val_y_torch = torch.from_numpy(val_y)

    trn = TensorDataset(trn_x_torch,trn_y_torch)
    val = TensorDataset(val_x_torch,val_y_torch)

    trn_dataloader = torch.utils.data.DataLoader(trn,batch_size=100,shuffle=False, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val,batch_size=100,shuffle=False, num_workers=4)

    return trn_dataloader, val_dataloader


if __name__ == "__main__":
    ae = AutoEncoder()
    print(ae)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr = 1e-3)
    losses = []
    EPOCHS = 5

    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(trn_dataloader):
            
            data = torch.autograd.Variable(data)

            optimizer.zero_grad()
            pred = ae(data)

            loss = loss_func(pred, data)
            losses.append(loss.cpu().data.item())

            loss.backward()

            optimizer.step()

            if batch_idx % 100 == 1:
                print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch+1,EPOCHS,batch_idx * len(data),len(trn_dataloader.dataset),100.*batch_idx/len(trn_dataloader),loss.cpu().data.item()),end='')

