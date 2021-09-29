#!/usr/bin/env python
# coding: utf-8

# ## Autoencoders
# Modified by Mahir Jain
# ###  Reference taken https://github.com/techshot25/Autoencoders
#
# This simple code shows you how to make an autoencoder using Pytorch. The idea is to bring down the number of dimensions (or reduce the feature space) using neural networks.
#
# The idea is simple, let the neural network learn how to make the encoder and the decoder using the feature space as both the input and the output of the network.





import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
import pandas as pd
import urllib.request as urllib2

# Here I am using the swiss roll example and reduce it from 3D to 2D

# In[10]:

use_dataset = 2  # 1.- coil 2000 2.- Madelon 3.- Swiss roll
if use_dataset == 1:
    in_shape = 85
    enc_shape = 50
    in_one = 70
    in_two = 60
    X = pd.read_csv('data\coildataupdated.txt', sep="\t", header=None)

elif use_dataset == 2:
    in_shape = 500
    enc_shape = 300
    in_one = 450
    in_two = 400
    train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
    #train_data_labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
    X = np.loadtxt(urllib2.urlopen(train_data_url))
    #madelon_train_sample_label = np.loadtxt(urllib2.urlopen(train_data_labels_url))

elif use_dataset == 3:
    in_shape = 3
    enc_shape = 2
    in_one = 128  # 100
    in_two = 64  # 10
    n_samples = 5000
    noise = 0.05
    X, color = make_swiss_roll(n_samples, noise)

device = ('cuda' if torch.cuda.is_available() else 'cpu')



X = MinMaxScaler().fit_transform(X)
#X = torch.from_numpy(X).to(device)

#fig = plt.figure()
#ax = p3.Axes3D(fig)
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap=plt.cm.jet)
#plt.title('Swiss roll')
#plt.show()

# In[38]:


x = torch.from_numpy(X).to(device)


class Autoencoder(nn.Module):
    """Makes the main denoising auto
    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(in_shape, in_one),
            #nn.ReLU(True),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(in_one, in_two),
            #nn.ReLU(True),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(in_two, enc_shape),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, in_two),
            #nn.ReLU(True),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(in_two, in_one),
            #nn.ReLU(True),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(in_one, in_shape)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


encoder = Autoencoder(in_shape=in_shape, enc_shape=enc_shape).double().to(device)

error = nn.MSELoss()

#optimizer = optim.Adam(encoder.parameters())
optimizer = torch.optim.SGD(encoder.parameters(),lr=0.01)


# In[39]:


def train(model, error, optimizer, n_epochs, x):
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()

        if epoch % int(0.1 * n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')


# You can rerun this function or just increase the number of epochs. Dropout was added for denoising, otherwise it will be very sensitive to input variations.

# In[42]:


train(encoder, error, optimizer, 5000, x)

# In[43]:


with torch.no_grad():
    encoded = encoder.encode(x)
    decoded = encoder.decode(encoded)
    mse = error(decoded, x).item()
    enc = encoded.cpu().detach().numpy()
    dec = decoded.cpu().detach().numpy()

# In[44]:
if use_dataset==3:

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.jet)
    #ax.set_title("Original data")
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.savefig('Swiss_Roll_Original')

    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X[:, 0], X[:, 1], c=color, cmap=plt.cm.jet)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.savefig('Swiss_Roll_Original_2D')
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(projection= '3d')
    ax.scatter(dec[:, 0], dec[:, 1], dec[:, 2], c=color, cmap=plt.cm.jet)
    plt.axis('tight')
    plt.xticks(size=10)
    plt.yticks(size=10)
    #plt.title('Projected data')
    #plt.show()
    plt.savefig('Swiss_Roll_Reconstructed.png')
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(dec[:, 0], dec[:, 1], c=color, cmap=plt.cm.jet)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.savefig('Swiss_Roll_Re_2D')
    plt.clf()

#plt.title('Encoded Swiss Roll')
#plt.show()

# In[45]:




print(f'Root mean squared error: {np.sqrt(mse):.4g}')

# Obviously there are some losses in variance due to the dimensionality reduction but this reconstruction is quite interesting. This is how the model reacts to another roll.

# In[118]:

if use_dataset==3:
    n_samples = 2500
    noise = 0.1
    X, colors = make_swiss_roll(n_samples, noise)

    X = MinMaxScaler().fit_transform(X)

    x = torch.from_numpy(X).to(device)

    with torch.no_grad():
        encoded = encoder.encode(x)
        decoded = encoder.decode(encoded)
        mse = error(decoded, x).item()
        enc = encoded.cpu().detach().numpy()
        dec = decoded.cpu().detach().numpy()

