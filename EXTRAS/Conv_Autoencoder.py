import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])


#transform = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 0
# how many samples per batch to load
batch_size = 30
# number of epochs to train the model
n_epochs = 20
use_dataset = int(input("Enter dataset to use: 1. MNIST 2. STL-10 3. CIFAR10 4.Fashion-MNIST   "))


if (use_dataset == 1):
    train_data = datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)
    #encoding_dim = 64
    shape = 28
    #img_shape = 28 * 28
elif (use_dataset == 2):
    train_data = datasets.STL10(root="~/torch_datasets", split='train', transform=transform, download=True)
    test_data = datasets.STL10(root="~/torch_datasets", split='test', transform=transform, download=True)
    #encoding_dim = 2048
    shape = 96
    #img_shape = 96 * 96
elif (use_dataset == 3):
    train_data = datasets.CIFAR10(root="~/torch_datasets", train=True, transform=transform,target_transform=None, download=True)
    test_data = datasets.CIFAR10(root="~/torch_datasets", train=False, transform=transform,target_transform=None, download=True)
    #encoding_dim = 256
    shape = 32
    #img_shape = 32 * 32
elif (use_dataset == 4):
    train_data = datasets.FashionMNIST(root="~/torch_datasets", train=True, transform=transform,target_transform=None, download=True)
    test_data = datasets.FashionMNIST(root="~/torch_datasets", train=False, transform=transform,target_transform=None, download=True)
    #encoding_dim = 256
    shape = 28

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))

        return x

# initialize the NN
model = ConvAutoencoder()
# specify loss function
criterion = nn.MSELoss()
# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images, _ = data
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * images.size(0)

    # print avg training statistics
    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
    ))



# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output= model(images)
# prep images for display
images = images.numpy()

# output is resized into a batch of iages
output = output.view(batch_size, 1, shape, shape)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()