import random
import matplotlib.pyplot as plt
import utils.utilities as ut
import os

from sklearn.model_selection import train_test_split
import numpy as np
import network 
import torch



def show_torch_image(torch_tensor):
    plt.imshow(torch_tensor.numpy().reshape(28, 28), cmap='gray')
    plt.show()
    return None

def initialize(trn_x, val_x, trn_y, val_y):
    trn_x_torch = torch.from_numpy(trn_x).type(torch.FloatTensor)
    trn_y_torch = torch.from_numpy(trn_y)

    val_x_torch = torch.from_numpy(val_x).type(torch.FloatTensor)
    val_y_torch = torch.from_numpy(val_y)

    trn = torch.utils.data.TensorDataset(trn_x_torch,trn_y_torch)
    val = torch.utils.data.TensorDataset(val_x_torch,val_y_torch)

    trn_dataloader = torch.utils.data.DataLoader(trn,batch_size=100,shuffle=False, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val,batch_size=100,shuffle=False, num_workers=4)

    return trn_x_torch, val_x_torch, trn_dataloader, val_dataloader

# Set file paths based on added MNIST Datasets
input_path = 'data'

training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte')

#print(training_images_filepath)

# Helper function to show a list of images with their relating titles
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.show()

# Load MINST dataset
mnist_dataloader = ut.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = np.array(x_train)
X_train = x_train.reshape(x_train.shape[0], -1)
Y_train = np.array(y_train)
print(X_train.shape)

x_test = np.array(x_test)
Y_test = np.array(y_test)
X_test = x_test.reshape(x_test.shape[0], -1)
print(X_test.shape)

'''
'''

trn_x,val_x,trn_y,val_y = train_test_split(X_train, Y_train, test_size=0.20)

print(trn_x.shape)
print(val_x.shape)

print(trn_y.shape)
print(val_y.shape)

plt.imshow(trn_x[0].reshape(28,-1), cmap='gray')
plt.show()
print(trn_y[0])

ae = network.AutoEncoder()
print(ae)

trn_x_torch, val_x_torch, trn_dataloader, val_dataloader = initialize(trn_x, val_x, trn_y, val_y)
show_torch_image(trn_x_torch[1])

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
            print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch+1,EPOCHS,batch_idx * len(data),len(trn_dataloader.dataset),100.*batch_idx/len(trn_dataloader),loss.cpu().data.item()/len(trn_dataloader)),end='')

ae.eval()
predictions = []

for batch_idx, (data,target) in enumerate(val_dataloader):
        
        data = torch.autograd.Variable(data)

        pred = ae(data)
        
        for prediction in pred:
            predictions.append(prediction)

len(predictions)
show_torch_image(val_x_torch[1])
show_torch_image(predictions[1].detach())

model_weights = []
model_layers = []

model_children = list(ae.children())

print(model_children[3].weight.data.numpy())
print(model_children[3].weight.data.numpy().shape)
# Show some random training and test images 
'''
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)
'''
