import random
import matplotlib.pyplot as plt
import utils.utilities as ut
import os

import numpy as np
import network 

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

#print(np.array(x_train).shape)
#print(np.array(x_test).shape)

#print(np.array(y_train).shape)
#print(np.array(y_test).shape)

x_train = np.array(x_train)
X_train = x_train.reshape(x_train.shape[0], -1)
print(X_train.shape)

x_test = np.array(x_test)
X_test = x_test.reshape(x_test.shape[0], -1)
print(X_test.shape)

#x = X_train[0]
#img = x.reshape(28, -1)
#plt.imshow(img)
#plt.show()

obj = network.AutoEncoder()
print(obj)

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
