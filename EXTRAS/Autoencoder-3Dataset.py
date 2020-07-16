import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# hyper parameters
batch_size = 30  # no of training examples in one forward/backward pass
epochs = 40  # no of passes through complete training set
learning_rate = 1e-3
encoding_dim = 0
  # 1. MNIST 2.STL-10 3.Flickr8k
shape = 0

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

if (use_dataset == 1):
    train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)
    encoding_dim = 64
    shape = 28
    img_shape = 28 * 28
elif (use_dataset == 2):
    train_dataset = torchvision.datasets.STL10(root="~/torch_datasets", split='train', transform=transform,
                                               download=True)
    test_dataset = torchvision.datasets.STL10(root="~/torch_datasets", split='test', transform=transform, download=True)
    encoding_dim = 2048
    shape = 96
    img_shape = 96 * 96
elif (use_dataset == 3):
    train_dataset = torchvision.datasets.CIFAR10(root="~/torch_datasets", train=True, transform=transform,
                                                 target_transform=None, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="~/torch_datasets", train=False, transform=transform,
                                                target_transform=None, download=True)
    encoding_dim = 256
    shape = 32
    img_shape = 32 * 32

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10, shuffle=True)


# Autoencoder
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=encoding_dim)
        self.encoder_output_layer = nn.Linear(in_features=encoding_dim, out_features=encoding_dim)
        self.decoder_hidden_layer = nn.Linear(in_features=encoding_dim, out_features=encoding_dim)
        self.decoder_output_layer = nn.Linear(in_features=encoding_dim, out_features=kwargs["input_shape"])

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        #print(reconstructed.shape)
        #print(reconstructed[0][500])
        return reconstructed


# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=img_shape).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# mean-squared error loss
criterion = nn.MSELoss()

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, img_shape).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

# Testing trained model
test_examples = None

with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features.view(-1, img_shape)
        reconstruction = model(test_examples)
        break

# Visualising result for test data
with torch.no_grad():
    number = 10  # batch-size of test dataset
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_examples[index].numpy().reshape(shape, shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(reconstruction[index].numpy().reshape(shape, shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()