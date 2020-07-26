#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mahir
python 3.6 --  working great
cae with mcmc in torch
"""

#  %reset
#  %reset -sf


import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import torch
import torch.nn as nn
import numpy as np
import random
import math
import copy
import os
import matplotlib.pyplot as plt
import times


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


problemfolder='results'
expno = open('Experiment No.txt', 'r+')
exp = int(expno.read())
exp_write = exp+1
expno.seek(0)
expno.write(str(exp_write))
expno.truncate()


num_workers = 0
# how many samples per batch to load
batch_size = 10
# number of epochs to train the model
n_epochs = 1
use_dataset = int(input("Enter dataset to use: 1. MNIST 2. STL-10 3. CIFAR10 4.Fashion-MNIST   "))
lrate=0.01
burnin =0.25
numSamples = int(input("Enter no of samples: "))
ulg = True
no_channels = 1
size_train = 1000
size_test = 1000


if use_dataset == 1:
    shape = 28
    problemfolder += '/autoencoder_' + str(exp) + '_MNIST_  ' + str(numSamples)
    PATH = 'saved_model' + 'MNIST.pt'
elif use_dataset == 2:
    shape = 96
    problemfolder += '/autoencoder_' + str(exp) + '_STL-10_' + str(numSamples)
    PATH = 'saved_model' + 'STL-10.pt'
elif use_dataset == 3:
    shape = 32
    problemfolder += '/autoencoder_' + str(exp) + '_CIFAR10_' + str(numSamples)
    PATH = 'saved_model' + 'CIFAR-10.pt'
else:
    shape=28
    problemfolder += '/autoencoder_' + str(exp) + '_Fashion-MNIST_' + str(numSamples)
    PATH = 'saved_model' + 'Fashion-MNIST.pt'

os.makedirs(problemfolder)
outres = open(problemfolder+'/results.txt', 'w')


def data_load(data='train'):
    #print("Accepting Input")
    if data == 'test':
        if (use_dataset == 1):
            test_data = datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)
            test_data, _ = torch.utils.data.random_split(test_data, [size_test, len(test_data) - size_test])
            shape = 28
            # img_shape = 28 * 28
        elif (use_dataset == 2):
            test_data = datasets.STL10(root="~/torch_datasets", split='test', transform=transform, download=True)
            test_data, _ = torch.utils.data.random_split(test_data, [size_test, len(test_data) - size_test])
            shape = 96
            # img_shape = 96 * 96
        elif (use_dataset == 3):
            test_data = datasets.CIFAR10(root="~/torch_datasets", train=False, transform=transform,
                                         target_transform=None, download=True)
            test_data, _ = torch.utils.data.random_split(test_data, [size_test, len(test_data) - size_test])
            shape = 32
            # img_shape = 32 * 32
        elif (use_dataset == 4):
            test_data = datasets.FashionMNIST(root="~/torch_datasets", train=False, transform=transform,
                                              target_transform=None, download=True)
            test_data, _ = torch.utils.data.random_split(test_data, [size_test, len(test_data) - size_test])
            shape = 28
            # img_shape= 28*28
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return test_loader

    else:
        if (use_dataset == 1):
            train_data = datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
            train_data, _ = torch.utils.data.random_split(train_data, [size_train, len(train_data) - size_train])
            shape = 28
            # img_shape = 28 * 28
        elif (use_dataset == 2):
            train_data = datasets.STL10(root="~/torch_datasets", split='train', transform=transform, download=True)
            train_data, _ = torch.utils.data.random_split(train_data, [size_train, len(train_data) - size_train])
            shape = 96
            # img_shape = 96 * 96
        elif (use_dataset == 3):
            train_data = datasets.CIFAR10(root="~/torch_datasets", train=True, transform=transform,target_transform=None, download=True)
            train_data, _ = torch.utils.data.random_split(train_data, [size_train, len(train_data) - size_train])
            shape = 32
            # img_shape = 32 * 32
        elif (use_dataset == 4):
            train_data = datasets.FashionMNIST(root="~/torch_datasets", train=True, transform=transform,target_transform=None, download=True)
            train_data, _ = torch.utils.data.random_split(train_data, [size_train, len(train_data) - size_train])
            shape = 28
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return train_loader

def f(): raise Exception("Found exit()")

def reshape(images,outputs):
    outputs_1 = outputs
    outputs_1 = outputs_1.reshape([batch_size, shape * shape])
    images_1 = images
    images_1 = images_1.reshape([batch_size, shape * shape])
    return images_1, outputs_1


class Model(nn.Module):
    # Defining input size, hidden layer size, output size and batch size respectively
    def __init__(self):
        super(Model, self).__init__()
        self.los = 0
        self.criterion = torch.nn.MSELoss()

        ## encoder##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 64, 3,stride=1, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(64, 32, 3,stride=1 ,padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 1, 2, stride=2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x= self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))

        #################
        x= self.pool(x)  # compressed representation
        #################

        ## decode ##
        # add transpose conv layers, with relu activation function
        #x = self.unpool(x, indices2, output_size=size2)
        x = self.t_conv1(x)
        x = F.relu(x)
        # output layer (with sigmoid for scaling from 0 to 1)
        x = self.t_conv2(x)
        x = F.sigmoid(x)
        return x

    def evaluate_proposal(self, data, w=None):
        #print("Inside evaluate Proposal")
        y_pred=[]
        self.los = 0
        if w is not None:
            self.loadparameters(w)
        for i, sample in enumerate(data, 0):
            inputs, labels = sample
            #print(inputs.shape)
            output = self.forward(inputs)
            loss = self.criterion(output, inputs)
            #loss = loss.item()*inputs.size(0)
            y_pred.append(loss)
            self.los += loss
        return y_pred

    def langevin_gradient(self, x, w=None):
        #print("Inside langevin gradient")
        if w is not None:
            self.loadparameters(w)
        for epoch in range(1, n_epochs + 1):
            self.los = 0
            for i, sample in enumerate(x, 0):
                images, _ = sample
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = self.forward(images)
                #images_1, outputs_1 = reshape(images, outputs)
                # calculate the loss
                loss = self.criterion(outputs, images)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                self.los += loss.item()*batch_size
                # print(lo,' is loss')
        return copy.deepcopy(self.state_dict())




    def getparameters(self, w=None):
        l = np.array([1, 2])
        dic = {}
        if w is None:
            dic = self.state_dict()
        else:
            dic = copy.deepcopy(w)
        for name in sorted(dic.keys()):
            l = np.concatenate((l, np.array(copy.deepcopy(dic[name])).reshape(-1)), axis=None)
        l = l[2:]
        return l

    def dictfromlist(self, param):
        dic = {}
        i = 0
        for name in sorted(self.state_dict().keys()):
            dic[name] = torch.FloatTensor(param[i:i + (self.state_dict()[name]).view(-1).shape[0]]).view(
                self.state_dict()[name].shape)
            i += (self.state_dict()[name]).view(-1).shape[0]
        # self.loadparameters(dic)
        return dic

    def loadparameters(self, param):
        self.load_state_dict(param)

    def addnoiseandcopy(self, mea, std_dev):
        dic = {}
        w = self.state_dict()
        for name in (w.keys()):
            dic[name] = copy.deepcopy(w[name]) + torch.zeros(w[name].size()).normal_(mean=mea, std=std_dev)
        self.loadparameters(dic)
        return dic


class MCMC:
    def __init__(self, samples, use_langevin_gradients, batch_size):
        self.samples = samples
        self.cae = Model()
        #self.cae.load_state_dict(torch.load(PATH)) #to load from pretrained model
        self.criterion = nn.MSELoss()
        #self.optimizer = torch.optim.Adam(self.cae.parameters(), lr=lrate)
        self.traindata = data_load(data='train')
        self.testdata = data_load(data='test')
        self.use_langevin_gradients = use_langevin_gradients
        self.batch_size = batch_size
        self.l_prob = 0.7
        outres.write("Langevin probability used " + str(self.l_prob))
        outres.write('\n')
        self.adapttemp=5
        self.train_loss = 0
        # ----------------


    def likelihood_func(self, cae, data, w,tau_sq):
        #print("Inside likelihood_func")
        #y = data
        fx = cae.evaluate_proposal(data, w)
        fx = [x*batch_size for x in fx]
        mse = torch.sum(torch.Tensor(fx))/len(data)
        loss = np.sum(-0.5 * np.log(2 * math.pi * tau_sq) - 0.5 * np.square(fx) / tau_sq)
        #print(type(loss))
        return [torch.sum(loss) / self.adapttemp, fx, mse]



    def prior_likelihood(self, sigma_squared, w_list):
        #print("inside prior likelihood")
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2
        return log_loss


    def accuracy(self, data):
        # Test the model
        # get sample outputs
        #print("Inside accuracy")
        loss_ans=[]
        for i, sample in enumerate(data, 0):
            inputs, labels = sample
            output = self.cae(inputs)
            loss = self.criterion(output, inputs)
            #loss = loss.item() * inputs.size(0)
            loss_ans.append(loss)

        loss_ans = [x*batch_size for x in loss_ans]
        loss = torch.sum(torch.Tensor(loss_ans))/len(data)
        return (1-loss)*100


    def sampler(self):
        samples = self.samples
        cae = self.cae
        w = cae.state_dict()
        w_size = len(cae.getparameters(w))
        mse_train = np.zeros(samples)
        mse_test = np.zeros(samples)
        acc_train = np.zeros(samples)
        acc_test = np.zeros(samples)

        likelihood_proposal_array = np.zeros(samples)
        likelihood_array = np.zeros(samples)
        diff_likelihood_array = np.zeros(samples)
        weight_array = np.zeros(samples)
        weight_array1 = np.zeros(samples)
        weight_array2 = np.zeros(samples)
        weight_array3 = np.zeros(samples)
        sum_value_array = np.zeros(samples)


        eta = 0
        w_proposal = np.random.randn(w_size)
        w_proposal = cae.dictfromlist(w_proposal)
        step_w = 0.05
        train = self.traindata  # data_load(data='train')
        test = self.testdata  # data_load(data= 'test')
        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        delta_likelihood = 0.5  # an arbitrary position
        prior_current = self.prior_likelihood(sigma_squared, cae.getparameters(w))


        [likelihood, pred_train, msetrain] = self.likelihood_func(cae, train, w, tau_sq=1.5)
        [_, pred_test, msetest] = self.likelihood_func(cae, test, w, tau_sq=1.5)

        # Beginning Sampling using MCMC RANDOMWALK

        '''
        y_test = torch.zeros(batch_size)
        for i, dat in enumerate(test, 0):
            inputs, labels = dat
            y_test[i] = inputs
        y_train = torch.zeros((100, 100, 96, 96))
        for i, dat in enumerate(train, 0):
            inputs, labels = dat
            y_train[i] = inputs
        '''
        trainacc = 0
        testacc = 0

        num_accepted = 0
        langevin_count = 0
        init_count = 0
        mse_train[0] = msetrain
        mse_test[0] = msetest
        acc_train[0] = self.accuracy(train)
        acc_test[0] = self.accuracy(test)
        likelihood_proposal_array[0]=0
        likelihood_array[0]=0
        diff_likelihood_array[0]=0
        weight_array[0]=0
        weight_array1[0] = 0
        weight_array2[0] = 0
        weight_array3[0]=0
        sum_value_array[0]=0

        #pytorch_total_params = sum(p.numel() for p in cae.parameters() if p.requires_grad)
        #print(pytorch_total_params)
        # acc_train[0] = 50.0
        # acc_test[0] = 50.0

        # print('i and samples')
        for i in range(samples):  # Begin sampling --------------------------------------------------------------------------
            #print("Sampling")
            lx = np.random.uniform(0, 1, 1)
            old_w = cae.state_dict()
            #and (lx < self.l_prob)
            if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                w_gd = cae.langevin_gradient(train)  # Eq 8
                w_proposal = cae.addnoiseandcopy(0, step_w)  # np.random.normal(w_gd, step_w, w_size) # Eq 7
                w_prop_gd = cae.langevin_gradient(train)
                wc_delta = (cae.getparameters(w) - cae.getparameters(w_prop_gd))
                wp_delta = (cae.getparameters(w_proposal) - cae.getparameters(w_gd))
                sigma_sq = step_w
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                diff_prop = first - second
                diff_prop = diff_prop
                langevin_count = langevin_count + 1
            else:
                diff_prop = 0
                w_proposal = cae.addnoiseandcopy(0, step_w)  # np.random.normal(w, step_w, w_size)


            [likelihood_proposal, pred_train, msetrain] = self.likelihood_func(cae, train, w, tau_sq=0.5)
            [likelihood_ignore, pred_test, msetest] = self.likelihood_func(cae, test, w, tau_sq=0.5)



            prior_prop = self.prior_likelihood(sigma_squared, cae.getparameters(w_proposal))  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            #diff_likelihood = diff_likelihood*-1
            diff_prior = prior_prop - prior_current

            likelihood_proposal_array[i] = likelihood_proposal
            likelihood_array[i] = likelihood
            diff_likelihood_array[i] = diff_likelihood



            #print("\n\n")
            #print("Likelihood Proposal")
            #print(likelihood_proposal)
            #print("\n\n")


            #print("\n\n")
            #print("Likelihood")
            #print(likelihood)
            #print("\n\n")

            #print("Diff_Likelihood")
            #print(diff_likelihood)
            #print("\n\n")

            #print("Diff_Prior")
            #print(diff_prior)
            #print("\n\n")

            #print("Diff_Prop")
            #print(diff_prop)
            #print("\n\n")

            #print("Sum Number")
            #print(diff_likelihood + diff_prior + diff_prop)
            #print("\n\n")
            #+ diff_prior + diff_prop

            #try:
            #    mh_prob = min(1, math.exp(diff_likelihood))
            #except OverflowError as e:
            #    mh_prob = 1

            sum_value=diff_likelihood + diff_prior + diff_prop
            u = np.log(random.uniform(0, 1))

            sum_value_array[i] = sum_value

            #print("Sum_Value")
            #print(sum_value)
            #print("\n\n")

            #print("U")
            #print(u)
            #print("\n\n")
            #print("MH_Prob")
            #print(mh_prob)
            #print("\n\n")

            if u < sum_value:
                num_accepted = num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)  # cae.getparameters(w_proposal)
                acc_train1 = self.accuracy(train)
                acc_test1 =self.accuracy(test)
                print (i, msetrain, msetest, acc_train1, acc_test1, 'accepted')
                mse_train[i] = msetrain
                mse_test[i] = msetest
                acc_train[i, ] = acc_train1
                acc_test[i, ] = acc_test1

            else:
                w = old_w
                cae.loadparameters(w)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print (i, msetrain, msetest, acc_train1, acc_test1, 'rejected')
                #rmse_train[i] = rmsetrain
                #rmse_test[i] = rmsetest
                #acc_train[i,] = acc_train1
                #acc_test[i,] = acc_test1
                mse_train[i,] = mse_train[i - 1,]
                mse_test[i,] = mse_test[i - 1,]
                acc_train[i,] = acc_train[i - 1,]
                acc_test[i,] = acc_test[i - 1,]

            ll = cae.getparameters()
            #print(ll[0])
            weight_array[i] = ll[0]
            weight_array1[i] = ll[100]
            weight_array3[i] = ll[500]
            weight_array2[i] = ll[1000]

            file_name = problemfolder + '/weight[0]' +'.txt'
            np.savetxt(file_name, weight_array, fmt='%1.2f')

            file_name = problemfolder + '/weight[100]' + '.txt'
            np.savetxt(file_name, weight_array1, fmt='%1.2f')

            file_name = problemfolder + '/weight[500]' +'.txt'
            np.savetxt(file_name, weight_array3, fmt='%1.2f')

            file_name = problemfolder + '/weight[1000]' +'.txt'
            np.savetxt(file_name, weight_array2, fmt='%1.2f')






        #print(len(ll))
        print((num_accepted * 100 / (samples * 1.0)), '% was Accepted')
        temp = num_accepted*100/samples*1.0
        outres.write(str(temp)+' % was Accepted')
        outres.write('\n')

        print((langevin_count * 100 / (samples * 1.0)), '% was Langevin')
        temp = langevin_count*100/samples*1.0
        outres.write(str(temp) + ' % was Langevin')
        outres.write('\n')

        ###################################VISUALIZATION OF RESULT#############################################
        dataiter = iter(data_load(data='test'))
        images, labels = dataiter.next()

        # get sample outputs
        output = cae(images)
        # prep images for display
        images = images.numpy()

        # output is resized into a batch of images
        output = output.view(batch_size, 1, shape, shape)
        # use detach when it's an output that requires_grad
        output = output.detach().numpy()

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

        # input images on top row, reconstructions on bottom
        for images, row in zip([images, output], axes):
            for img, ax in zip(images, row):
                ax.imshow(np.squeeze(img), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.savefig(problemfolder + '/results.png')
        plt.clf()
        ##################################################################################################################
        #torch.save(cae.state_dict(), PATH)
        return acc_train, acc_test, mse_train, mse_test, sum_value_array, weight_array, weight_array1, weight_array2, weight_array3



def main():


    time1= times.now()

    mcmc = MCMC(numSamples, ulg, batch_size)  # declare class
    acc_train, acc_test, mse_train, mse_test, sva, wa, wa1, wa2, wa3 = mcmc.sampler()

    time2 = times.now()

    time2 = (time2-time1)
    outres.write("Time for sampling: "+ str(time2))
    outres.write('\n')
    outres.write('\n')

    acc_train = acc_train[int(numSamples*burnin):]
    #print(acc_train)
    acc_test = acc_test[int(numSamples*burnin):]
    mse_train = mse_train[int(numSamples*burnin):]
    mse_test = mse_test[int(numSamples*burnin):]
    sva = sva[int(numSamples*burnin):]
    #print(lpa)

    print("\n\n\n\n\n\n\n\n")
    print("Mean of MSE Train")
    print(np.mean(mse_train))
    outres.write('Mean of MSE Train: '+str(np.mean(mse_train)))
    outres.write('\n')
    print("\n")

    print("Mean of Accuracy Train")
    print(np.mean(acc_train))
    outres.write('Mean of Accuracy Train: ' + str(np.mean(acc_train)))
    outres.write('\n')
    print("\n")

    print("Mean of MSE Test")
    print(np.mean(mse_test))
    outres.write('Mean of MSE Test: ' + str(np.mean(mse_test)))
    outres.write('\n')
    print("\n")

    print("Mean of Accuracy Test")
    print(np.mean(acc_test))
    outres.write('Mean of Accuracy Test: ' + str(np.mean(acc_test)))
    outres.write('\n')
    outres.write('\n')




    x = np.linspace(0, int(numSamples-numSamples*burnin), num=int(numSamples-numSamples*burnin))
    x1 = np.linspace(0, numSamples, num=numSamples)

    plt.plot(x1, wa, label='Weight[0]')
    plt.legend(loc='upper right')
    plt.title("Weight[0] Trace")
    plt.savefig(problemfolder + '/weight[0]_samples.png')
    plt.clf()

    plt.plot(x1, wa1, label='Weight[100]')
    plt.legend(loc='upper right')
    plt.title("Weight[100] Trace")
    plt.savefig(problemfolder+ '/weight[100]_samples.png')
    plt.clf()

    plt.plot(x1, wa3, label='Weight[500]')
    plt.legend(loc='upper right')
    plt.title("Weight[500] Trace")
    plt.savefig(problemfolder+ '/weight[500]_samples.png')
    plt.clf()

    plt.plot(x1, wa2, label='Weight[1000]')
    plt.legend(loc='upper right')
    plt.title("Weight[1000] Trace")
    plt.savefig(problemfolder + '/weight[1000]_samples.png')
    plt.clf()

    plt.plot(x, sva, label='Sum_Value')
    plt.legend(loc='upper right')
    plt.title("Sum Value Over Samples")
    plt.savefig(problemfolder+'/sum_value_samples.png')
    plt.clf()


    #plt.plot(x, acc_train, label='Train')
    #plt.legend(loc='upper right')
    #plt.title("Accuracy Train Values Over Samples")
    #plt.savefig('mnist_torch_single_chain' + '/accuracy_samples.png')
    #plt.clf()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    a, b = 0, 100
    ax1.set_ylim(a, b)
    ax2.set_ylim(a, b)
    color = 'tab:red'
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Accuracy Train', color=color)
    ax1.plot(x, acc_train, color=color)
    ax1.tick_params(axis='y', labelcolor=color)



    color = 'tab:blue'
    ax2.set_ylabel('Accuracy Test', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, acc_test, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    #ax3=ax1.twinx()

    #color = 'tab:green'
    #ax3.set_ylabel('Accuracy Test', color=color)  # we already handled the x-label with ax1
    #ax3.plot(x, acc_test, color=color)
    #ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(problemfolder + '/superimposed_acc.png')
    plt.clf()

    fig1, ax4 = plt.subplots()
    ax5 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
    a, b = 0, 1
    ax4.set_ylim(a, b)
    ax5.set_ylim(a, b)

    color = 'tab:red'
    ax4.set_xlabel('Samples')
    ax4.set_ylabel('MSE Train', color=color)
    ax4.plot(x, mse_train, color=color)
    ax4.tick_params(axis='y', labelcolor=color)



    color = 'tab:blue'
    ax5.set_ylabel('MSE Test', color=color)  # we already handled the x-label with ax1
    ax5.plot(x, mse_test, color=color)
    ax5.tick_params(axis='y', labelcolor=color)

    #ax6 = ax4.twinx()

    #color = 'tab:green'
    #ax6.set_ylabel('MSE Test', color=color)  # we already handled the x-label with ax1
    #ax6.plot(x, mse_test, color=color)
    #ax6.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(problemfolder + '/superimposed_mse.png')
    plt.clf()


    plt.ylim(0,1)
    plt.scatter(x, mse_train)
    plt.title("MSE Train")
    plt.xlabel("Samples")
    plt.ylabel("MSE")
    plt.savefig(problemfolder + '/scatter_train_mse.png')
    plt.clf()


    plt.ylim(0,1)
    plt.scatter(x, mse_test)
    plt.title("MSE Test")
    plt.xlabel("Samples")
    plt.ylabel("MSE")
    plt.savefig(problemfolder + '/scatter_test_mse.png')
    plt.clf()

    plt.ylim(0,100)
    plt.scatter(x, acc_train)
    plt.title("Accuracy Train")
    plt.xlabel("Samples")
    plt.ylabel("%")
    plt.savefig(problemfolder + '/scatter_train_accuracy.png')
    plt.clf()
    
    plt.ylim(0,100)
    plt.scatter(x, acc_test)
    plt.title("Accuracy Test")
    plt.xlabel("Samples")
    plt.ylabel("%")
    plt.savefig(problemfolder + '/scatter_test_accuracy.png')
    plt.clf()


if __name__ == "__main__": main()