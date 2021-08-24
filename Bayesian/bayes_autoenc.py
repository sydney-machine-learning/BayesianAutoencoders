#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mahir
python 3.6 --  working great
cae with mcmc in torch
"""

#  %reset
#  %reset -sf


# import torchvision
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from torchvision import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_swiss_roll
import torch
import torch.nn as nn
import numpy as np
import random
import math
import copy
import os
import matplotlib.pyplot as plt
import time
import multiprocessing
import urllib.request as urllib2
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from pprint import pprint
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import csv
import pandas as pd

device = "cpu"

problemfolder = 'res'
expno = open('Experiment No.txt', 'r+')
exp = int(expno.read())
exp_write = exp + 1
expno.seek(0)
expno.write(str(exp_write))
expno.truncate()

num_workers = 0
# how many samples per batch to load
batch_size = 10
# number of epochs to train the model
n_epochs = 1

# lrate = 0.09
burnin = 0.5
ulg = True
no_channels = 1
step_size = 0.025 #0.005
num_chains = 8  # equal to no of cores available
pt_samples = 0.5
 
mt_val = 2
maxtemp = 2
swap_interval = 5 #10
# noise = 0.0125
use_dataset = 3  # 1.- coil 2000 2.- Madelon 3.- Swiss roll
#use_dataset = int(input('Enter dataset (1/2/3) you want to use [1 (coil 2000) 2 (Madelon) 3 (Swiss roll)]'))  # 1.- coil 2000 2.- Madelon 3.- Swiss roll

if use_dataset == 1:
    in_shape = 85
    enc_shape = 50
    in_one = 70
    in_two = 60
    lrate = 0.09  # 0.05
    step_size = 0.05  # 0.09

elif use_dataset == 2:
    in_shape = 500
    enc_shape = 300
    in_one = 450
    in_two = 400
    step_size = 0.005
    lrate = 0.01
    train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
    train_data_labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
    # does not work, better keep the dataset in repo
    madelon_train_sample = np.loadtxt(urllib2.urlopen(train_data_url))
    madelon_train_sample_label = np.loadtxt(urllib2.urlopen(train_data_labels_url))

elif use_dataset == 3:
    in_shape = 3
    enc_shape = 2
    in_one = 15 #128  # 100
    in_two = 10 # 64  # 10
    lrate = 0.01  # 0.04
    step_size = 0.03  # 0.03


def data_load(data='train'):
    if use_dataset == 1:
        X = pd.read_csv('Datasets/coildataupdated.txt', sep="\t", header=None)  # this does not work in my case (linux)
        X = MinMaxScaler().fit_transform(X)
        #X = torch.from_numpy(X).to(device)
        train_data, test_data = train_test_split(X)
        if data == 'test':
            return test_data
        else:
            return train_data

    elif use_dataset == 2:
        if data == 'test':
            test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
            X = np.loadtxt(urllib2.urlopen(test_data_url))
            X = MinMaxScaler().fit_transform(X)
            #X = torch.from_numpy(X).to(device) 
            return X
        else:
            train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
            X = np.loadtxt(urllib2.urlopen(train_data_url))
            X = MinMaxScaler().fit_transform(X)
            #X = torch.from_numpy(X).to(device) 
            return X

    elif use_dataset == 3:
        X, color = make_swiss_roll(n_samples=5000)
        # print(X.shape)
        X = MinMaxScaler().fit_transform(X)
        #X = torch.from_numpy(X).to(device)
        train_data, test_data = train_test_split(X)
        if data == 'test':
            return test_data
        else:
            return train_data


def f(): raise Exception("Found exit()")


class Model(nn.Module):
    # Defining input size, hidden layer size, output size and batch size respectively
    def __init__(self):
        super(Model, self).__init__()
        self.los = 0
        self.criterion = torch.nn.MSELoss()
        self.encode = nn.Sequential(
            nn.Linear(in_shape, in_one),
            # nn.ReLU(True),
            nn.Sigmoid(),
            #nn.Dropout(0.2),
            nn.Linear(in_one, in_two),
            # nn.ReLU(True),
            nn.Sigmoid(),
            #nn.Dropout(0.2),
            nn.Linear(in_two, enc_shape),
            # nn.Sigmoid()
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, in_two),
            # nn.ReLU(True),
            nn.Sigmoid(),
            #nn.Dropout(0.2), # droputs should not have been used. 
            nn.Linear(in_two, in_one),
            # nn.ReLU(True),
            nn.Sigmoid(),
            #nn.Dropout(0.2),
            nn.Linear(in_one, in_shape),
            # nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=lrate)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def evaluate_proposal(self, data, w=None):
        # print("Inside evaluate Proposal")

        
        data = torch.from_numpy(data).to(device)

        #y_pred = []
        #self.los = 0
        if w is not None:
            self.loadparameters(w)
        output = copy.deepcopy(self.forward(data).detach())
        output = output.data
        #loss = self.criterion(output, data)
        # loss = loss.item()*inputs.size(0)
        #y_pred.append(loss.item())
        #self.los += loss

        
        y_pred = output.detach().numpy()


        return  y_pred

    def langevin_gradient(self, x, w=None): #this needs to be checked

        
        x = torch.from_numpy(x).to(device)

        # print("Inside langevin gradient")
        if w is not None:
            self.loadparameters(w)
        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad()
            output = self.forward(x)
            loss = self.criterion(output, x)  # not sure if this is appropiate 
            loss.backward()
            self.optimizer.step()
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

        #print(l, ' is  l')
        
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


class ptReplica(multiprocessing.Process):
    def __init__(self, use_langevin_gradients, learn_rate, w, minlim_param, maxlim_param, samples, traindata, testdata,
                 burn_in, temperature, swap_interval, path, parameter_queue, main_process, event, batch_size,
                 step_size ):
        self.samples = samples
        self.cae = Model().double().to(device)
        # self.outres1 =outresult
        multiprocessing.Process.__init__(self)
        # self.cae.load_state_dict(torch.load(PATH)) #to load from pretrained model
        self.processID = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event = event
        self.swap_interval = swap_interval
        self.criterion = nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.cae.parameters(), lr=lrate)
        self.traindata = traindata
        self.testdata = testdata
        self.use_langevin_gradients = use_langevin_gradients
        self.sgd_depth = 1  # Keep as 1
        self.batch_size = batch_size
        self.l_prob = 0.7 # 0.7
        self.adapttemp = temperature
        self.temperature = temperature
        self.train_loss = 0
        self.step_size = step_size
        self.temperature = temperature
        self.w = w
        self.minY = np.zeros((1, 1))
        self.maxY = np.zeros((1, 1))
        self.minlim_param = minlim_param
        self.maxlim_param = maxlim_param 
        self.path = path

        # self.outres1.write("Langevin probability used " + str(self.l_prob))
        # self.outres1.write('\n')

        # ----------------

    @staticmethod
    def likelihood_func(cae, data, w, tau_sq, temp):
        #print("Inside likelihood_func")
        # y = data

        #tau_sq =  tau_sq.detach().numpy()
        
        #data = data.detach().numpy()

        #print(data.shape , ' data shape in lhood ')

        fx = cae.evaluate_proposal(data, w) 
        
        n = data.shape[0] * data.shape[1]
        
        #fx_ = fx.detach().numpy()
 

        mse = np.mean(np.square(fx -  data) ) 
 
        #loss = torch.sum(torch.as_tensor((-0.5 * np.log(2 * math.pi * tau_sq) - 0.5 * np.square(fx) / tau_sq)))


        px = -(n/2) * np.log(2 * math.pi * tau_sq)

        loss =  px - 0.5 *tau_sq * np.sum(np.square(fx - data) )

  

        return [loss / temp, fx, mse]

    def prior_likelihood(self, sigma_squared, w_list, tausq, nu_1, nu_2):
        # print("inside prior likelihood") 
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def accuracy(self, data): # i dont get the point of this (if MESE then we already get it from lhood)

        '''loss_ans = []
        output = self.cae(data)
        loss = self.criterion(output, data) 
        loss_ans.append(loss)'''
 
        #loss = 0 #torch.mean(torch.Tensor(loss_ans))
        return 0

    def run(self):
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
        weight_array4 = np.zeros(samples)
        weight_array5 = np.zeros(samples)
        weight_array6 = np.zeros(samples)
        weight_array7 = np.zeros(samples)
        weight_array8 = np.zeros(samples)
        weight_array9 = np.zeros(samples)
        weight_array10 = np.zeros(samples)
        weight_array11 = np.zeros(samples)
        weight_array12 = np.zeros(samples)
        sum_value_array = np.zeros(samples)
        u_value = np.zeros(samples)

        train = self.traindata  # data_load(data='train')
        pred_train = cae.evaluate_proposal(train, w)
 

        


        #pred_test = cae.evaluate_proposal(self.testdata, w)
        #pred_train = torch.tensor(pred_train)


        eta = np.log(np.var(pred_train - train))


        tau_pro = np.exp(eta)
 
        step_eta = 0.2
        # eta = 0 # size of compressed neural network (300)
        w_proposal_ = np.random.randn(w_size)
 
        w_proposal = cae.dictfromlist(w_proposal_)
 

        step_w = self.step_size
        train = self.traindata  # data_load(data='train')
        test = self.testdata  # data_load(data= 'test')
        sigma_squared = 25
        nu_1 = 0
        nu_2 = 3 
        prior_current = self.prior_likelihood(sigma_squared, w_proposal_, tau_pro, nu_1, nu_2)

        [likelihood, pred_train, msetrain] = self.likelihood_func(cae, train, w_proposal, tau_pro, self.adapttemp)
        #[_,  , msetest] = self.likelihood_func(cae, test, w, tau_pro, self.adapttemp)

        print(prior_current, likelihood, '    ** init lhood ')

        print('begin   * * ') #Beginning Sampling using MCMC RANDOMWALK
 

        num_accepted = 0
        langevin_count = 0
        init_count = 0 
        weight_array[0] = 0
        weight_array1[0] = 0
        weight_array2[0] = 0
        weight_array3[0] = 0
        weight_array4[0] = 0
        weight_array5[0] = 0
        weight_array6[0] = 0
        weight_array7[0] = 0
        weight_array8[0] = 0
        weight_array9[0] = 0
        weight_array10[0] = 0
        weight_array11[0] = 0
        weight_array12[0] = 0

        sum_value_array[0] = 0

        # pytorch_total_params = sum(p.numel() for p in cae.parameters() if p.requires_grad)
        # print(pytorch_total_params)
        # acc_train[0] = 50.0
        # acc_test[0] = 50.0

        # print('i and samples')
        for i in range(
                samples):  # Begin sampling --------------------------------------------------------------------------
            # print("Sampling")
            ratio = ((samples - i) / (samples * 1.0))
            if i < pt_samples:
                self.adapttemp = self.temperature  # T1=T/log(k+1);
            if i == pt_samples and init_count == 0:  # Move to canonical MCMC
                self.adapttemp = 1
                [likelihood, pred_train, msetrain] = self.likelihood_func(cae, train, w_proposal, 1, self.adapttemp)
                [_, pred_test, msetest] = self.likelihood_func(cae, test, w_proposal, 1, self.adapttemp)

                init_count = 1

            lx = np.random.uniform(0, 1, 1)
            old_w = cae.state_dict()
            # and (lx < self.l_prob)
            if  (self.use_langevin_gradients is True) and (lx < self.l_prob):
                w_gd = cae.langevin_gradient(train)  # Eq 8
                w_proposal = cae.addnoiseandcopy(0, step_w)  # np.random.normal(w_gd, step_w, w_size) # Eq 7
                w_prop_gd = cae.langevin_gradient(train)
                wc_delta = (cae.getparameters(w) - cae.getparameters(w_prop_gd))
                wp_delta = (cae.getparameters(w_proposal) - cae.getparameters(w_gd))
                sigma_sq = step_w * step_w
                # print(wc_delta)
                # print(wp_delta)
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                # print('first', first)
                # print('second', second)
                diff_prop = first - second
                diff_prop = diff_prop
                langevin_count = langevin_count + 1
                # print('langevin')
            else:
                diff_prop = 0
                w_proposal = cae.addnoiseandcopy(0, step_w)  # np.random.normal(w, step_w, w_size)
                # print('random')

            eta_pro = eta + np.random.normal(0, step_eta, 1)
 

            tau_pro = np.exp(eta_pro)
            tau_pro = tau_pro[0]
 

            
            [likelihood_proposal, pred_train, msetrain] = self.likelihood_func(cae, train, w_proposal, tau_pro, self.adapttemp)
            
            [likelihood_ignore, pred_test, msetest] = self.likelihood_func(cae, test, w_proposal, tau_pro, self.adapttemp)

            prior_prop = self.prior_likelihood(sigma_squared, cae.getparameters(w_proposal), tau_pro, nu_1, nu_2)  # takes care of the gradients
             

            diff_likelihood = likelihood_proposal - likelihood
            # diff_likelihood = diff_likelihood*-1
            prior_prop = torch.tensor(prior_prop)
            #priot_current = torch.tensor(prior_current)
            diff_prior = prior_prop - prior_current

            likelihood_proposal_array[i] = likelihood_proposal
            likelihood_array[i] = likelihood
            diff_likelihood_array[i] = diff_likelihood

        
            sum_value = diff_likelihood + diff_prior + diff_prop
             
            u = np.log(random.uniform(0, 1))
            u_value[i] = u
            # print(u)

            sum_value_array[i] = sum_value
 

            if u < sum_value:
            #if 1:
                num_accepted = num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)  # cae.getparameters(w_proposal)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                # print('like',diff_likelihood)
                # print('prior',diff_prior)
                # print('prop',diff_prop)
                print(i, num_accepted, msetrain, msetest, acc_train1, acc_test1, likelihood_proposal, prior_prop, diff_prop, 'accepted')
                # print(sum_value)
                mse_train[i] = msetrain
                mse_test[i] = msetest
                acc_train[i,] = acc_train1
                acc_test[i,] = acc_test1
                eta = eta_pro

            else:
                w = old_w
                cae.loadparameters(w)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                # print('like',diff_likelihood)
                # print('prior',diff_prior)
                # print('prop',diff_prop)
                #print(i, msetrain, msetest, acc_train1, acc_test1, 'rejected')
                # print(sum_value)
                # mse_train[i] = msetrain
                # mse_test[i] = msetest
                # acc_train[i,] = acc_train1
                # acc_test[i,] = acc_test1
                mse_train[i,] = mse_train[i - 1,]
                mse_test[i,] = mse_test[i - 1,]
                acc_train[i,] = acc_train[i - 1,]
                acc_test[i,] = acc_test[i - 1,]

            ll = cae.getparameters()
            # print(len(ll))
            # print(ll[0])

            weight_array[i] = ll[0]
            if len(ll) >= 100:
                weight_array1[i] = ll[100]
            if len(ll) >= 5000:
                weight_array3[i] = ll[5000]
            if len(ll) >= 10000:
                weight_array2[i] = ll[10000]
            if len(ll) >= 2000:
                weight_array4[i] = ll[2000]
            if len(ll) >= 3000:
                weight_array5[i] = ll[3000]
            if len(ll) >= 4000:
                weight_array6[i] = ll[4000]
            if len(ll) >= 5000:
                weight_array7[i] = ll[5000]
            if len(ll) >= 6000:
                weight_array8[i] = ll[6000]
            if len(ll) >= 7000:
                weight_array9[i] = ll[7000]
            if len(ll) >= 8000:
                weight_array10[i] = ll[8000]
            if len(ll) >= 9000:
                weight_array11[i] = ll[9000]
            if len(ll) >= 11000:
                weight_array12[i] = ll[11000]

            if (i + 1) % self.swap_interval == 0:
                param = np.concatenate([np.asarray([cae.getparameters(w)]).reshape(-1), np.asarray([eta]).reshape(-1),
                                        np.asarray([likelihood]), np.asarray([self.adapttemp]), np.asarray([i])])
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                self.event.wait()
                result = self.parameter_queue.get()
                w = cae.dictfromlist(result[0:w_size])
                eta = result[w_size]

            # if i % 100 == 0:
            # print(i, msetrain, msetest, 'Iteration Number and MSE Train & Test')

            # """
            # big_data=data_load1()
            # final_test_acc=self.accuracy(big_data)
            # print(final_test_acc)
            # """
        param = np.concatenate(
            [np.asarray([cae.getparameters(w)]).reshape(-1), np.asarray([eta]).reshape(-1),
             np.asarray([likelihood]),
             np.asarray([self.adapttemp]), np.asarray([i])])
        # print('SWAPPED PARAM',self.temperature,param)
        # self.parameter_queue.put(param)
        self.signal_main.set()
        # param = np.concatenate([s_pos_w[i-self.surrogate_interval:i,:],lhood_list[i-self.surrogate_interval:i,:]],axis=1)
        # self.surrogate_parameterqueue.put(param)

        print((num_accepted * 100 / (samples * 1.0)), '% was Accepted')
        accept_ratio = num_accepted / (samples * 1.0) * 100

        print((langevin_count * 100 / (samples * 1.0)), '% was Langevin')
        langevin_ratio = langevin_count / (samples * 1.0) * 100

        print('Exiting the Thread', self.temperature)

        file_name = self.path + '/sum_value_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, sum_value_array, fmt='%1.2f')

        file_name = self.path + '/likelihood_value_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, likelihood_array, fmt='%1.4f')

        file_name = self.path + '/likelihood_value_proposal' + str(self.temperature) + '.txt'
        np.savetxt(file_name, likelihood_proposal_array, fmt='%1.4f')

        file_name = self.path + '/weight[0]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array, fmt='%1.2f')

        file_name = self.path + '/weight[100]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array1, fmt='%1.2f')

        file_name = self.path + '/weight[10000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array2, fmt='%1.2f')

        file_name = self.path + '/weight[5000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array3, fmt='%1.2f')

        file_name = self.path + '/weight[2000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array4, fmt='%1.2f')

        file_name = self.path + '/weight[3000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array5, fmt='%1.2f')

        file_name = self.path + '/weight[4000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array6, fmt='%1.2f')

        file_name = self.path + '/weight[5000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array7, fmt='%1.2f')

        file_name = self.path + '/weight[6000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array8, fmt='%1.2f')

        file_name = self.path + '/weight[7000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array9, fmt='%1.2f')

        file_name = self.path + '/weight[8000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array10, fmt='%1.2f')

        file_name = self.path + '/weight[9000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array11, fmt='%1.2f')

        file_name = self.path + '/weight[11000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array12, fmt='%1.2f')

        file_name = self.path + '/mse_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, mse_test, fmt='%1.2f')

        file_name = self.path + '/mse_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, mse_train, fmt='%1.2f')

        file_name = self.path + '/acc_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, acc_test, fmt='%1.2f')

        file_name = self.path + '/acc_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, acc_train, fmt='%1.2f')

        file_name = self.path + '/u_value_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, u_value, fmt='%1.4f')

        file_name = self.path + '/accept_percentage' + str(self.temperature) + '.txt'
        with open(file_name, 'w') as f:
            f.write('%d' % accept_ratio)

        # print(len(ll))
        print((num_accepted * 100 / (samples * 1.0)), '% was Accepted')
        temp = num_accepted * 100 / samples * 1.0
        # self.outres1.write(str(temp) + ' % was Accepted')
        # self.outres1.write('\n')

        print((langevin_count * 100 / (samples * 1.0)), '% was Langevin')
        temp = langevin_count * 100 / samples * 1.0
        # self.outres1.write(str(temp) + ' % was Langevin')
        # self.outres1.write('\n')

        ###################################Classification##########################################################################################################################
        if use_dataset == 2:
            global madelon_train_sample
            madelon_train_sample = StandardScaler().fit_transform(madelon_train_sample)
            madelon_train_sample = torch.from_numpy(madelon_train_sample).to(device)
            madelon_train_sample = copy.deepcopy(cae.encode(madelon_train_sample).detach())
            madelon_train_sample = madelon_train_sample.data
            mad_X_train, mad_X_test, mad_y_train, mad_y_test = train_test_split(madelon_train_sample, \
                                                                                madelon_train_sample_label)
            # using out of the box default parameters provided in scikit learn library
            names_of_classifiers = ['LogisticRegression', 'KNeighbors', 'DecisionTree', 'SVClassifier']

            classifiers = [
                LogisticRegression(n_jobs=-1, random_state=42, max_iter=200),
                KNeighborsClassifier(n_jobs=-1),
                DecisionTreeClassifier(random_state=42),
                SVC(random_state=42)]

            mad_raw_test_scores = {}
            mad_raw_train_scores = {}
            mad_raw_y_preds = {}

            for name, clfr in zip(names_of_classifiers, classifiers):
                clfr.fit(mad_X_train, mad_y_train)

                train_score = clfr.score(mad_X_train, mad_y_train)
                test_score = clfr.score(mad_X_test, mad_y_test)
                y_pred = clfr.predict(mad_X_test)

                mad_raw_train_scores[name] = train_score
                mad_raw_test_scores[name] = test_score
                mad_raw_y_preds[name] = y_pred

            print('Test', mad_raw_test_scores)
            # print('Train',mad_raw_train_scores)

        elif use_dataset == 3:
            X, color = make_swiss_roll(n_samples=2500)
            X = MinMaxScaler().fit_transform(X)
            X = torch.from_numpy(X).to(device)
            # X_r = copy.deepcopy(cae.encode(X).detach())
            X_r = copy.deepcopy(cae.forward(X).detach())
            # X_r= cae.forward(X)
            '''
            fig = plt.figure(figsize=(15,6))
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.jet)
            #ax.set_title("Original data")
            plt.savefig(self.path + 'Swiss_Roll_Original')
            plt.clf()
            fig = plt.figure(figsize=(15, 6))
            ax = fig.add_subplot()
            ax.scatter(X[:, 0], X[:, 1], c=color, cmap=plt.cm.jet)
            plt.savefig(self.path + 'Swiss_Roll_Original_2D')
            plt.clf()
            '''

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=color, cmap=plt.cm.jet)
            plt.axis('tight')
            plt.xticks(fontsize=10), plt.yticks(fontsize=10)
            plt.savefig(self.path + '/Swiss_Roll_Reconstructed.png')
            plt.clf()

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.jet)
            plt.axis('tight')
            plt.xticks(fontsize=10), plt.yticks(fontsize=10)
            plt.savefig(self.path + 'Swiss_Roll_Re_2D')
            plt.clf()

        elif use_dataset == 1:
            #global madelon_train_sample
            coil_train_sample = pd.read_csv('data\Ticeval2000.txt', sep="\t", header=None)
            coil_train_sample_label = pd.read_csv('data\Tictgts2000.txt', sep="\t", header=None)
            coil_train_sample= coil_train_sample.to_numpy()
            coil_train_sample_label= coil_train_sample_label.to_numpy()
            coil_train_sample = MinMaxScaler().fit_transform(coil_train_sample)
            coil_train_sample = torch.from_numpy(coil_train_sample).to(device)
            coil_train_sample = copy.deepcopy(cae.encode(coil_train_sample).detach())
            coil_train_sample = coil_train_sample.data
            c_X_train, c_X_test, c_y_train, c_y_test = train_test_split(coil_train_sample, \
                                                                                coil_train_sample_label)
            # using out of the box default parameters provided in scikit learn library
            names_of_classifiers = ['LogisticRegression', 'KNeighbors', 'DecisionTree', 'SVClassifier']

            classifiers = [
                LogisticRegression(n_jobs=-1, random_state=42, max_iter=200),
                KNeighborsClassifier(n_jobs=-1),
                DecisionTreeClassifier(random_state=42),
                SVC(random_state=42)]

            c_raw_test_scores = {}
            c_raw_train_scores = {}
            c_raw_y_preds = {}

            for name, clfr in zip(names_of_classifiers, classifiers):
                clfr.fit(c_X_train, c_y_train)

                train_score = clfr.score(c_X_train, c_y_train)
                test_score = clfr.score(c_X_test, c_y_test)
                y_pred = clfr.predict(c_X_test)

                c_raw_train_scores[name] = train_score
                c_raw_test_scores[name] = test_score
                c_raw_y_preds[name] = y_pred

            print('Test', c_raw_test_scores)
            # print('Train',mad_raw_train_scores)


        ##################################################################################################################################################################################
        # torch.save(cae.state_dict(), PATH)
        return acc_train, acc_test, mse_train, mse_test, sum_value_array, weight_array, weight_array1, weight_array2, weight_array3


# Manages the parallel tempering, initialises and executes the parallel chains
class ParallelTempering:
    def __init__(self, use_langevin_gradients, num_chains, maxtemp, NumSample, swap_interval,
                 path, batch_size, bi, step_size ):
        cae = Model()
        # self.outres = outres
        self.cae = cae
        self.traindata = data_load(data='train')
        self.testdata = data_load(data='test')
        self.num_param = len(cae.getparameters(
            cae.state_dict()))  # (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        # Parallel Tempering variables
        self.swap_interval = swap_interval
        self.path = path
        self.maxtemp = maxtemp
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.NumSamples = int(NumSample / self.num_chains)
        self.sub_sample_size = max(1, int(0.05 * self.NumSamples))
        # create queues for transfer of parameters between process chain
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()
        self.wait_chain = [multiprocessing.Event() for i in range(self.num_chains)]
        self.event = [multiprocessing.Event() for i in range(self.num_chains)]
        self.all_param = None
        self.geometric = True  # True (geometric)  False (Linear)
        self.minlim_param = 0.0
        self.maxlim_param = 0.0
        self.minY = np.zeros((1, 1))
        self.maxY = np.ones((1, 1))
        self.model_signature = 0.0
        self.learn_rate = lrate
        self.use_langevin_gradients = use_langevin_gradients
        self.batch_size = batch_size
        self.masternumsample = NumSample
        self.burni = bi
        self.step_size = step_size 

    def default_beta_ladder(self, ndim, ntemps,
                            Tmax):   

        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')
 

        maxtemp = Tmax
        numchain = ntemps
        b = []
        b.append(maxtemp)
        last = maxtemp
        for i in range(maxtemp):
            last = last * (numchain ** (-1 / (numchain - 1)))
            b.append(last)
        tstep = np.array(b)

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0 * np.sqrt(np.log(4.0)) / np.sqrt(ndim)
        else:
            tstep = tstep[ndim - 1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                 'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas 

    def assign_temperatures(self):
        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)
            for i in range(0, self.num_chains):
                self.temperatures.append(np.inf if betas[i] == 0 else 1.0 / betas[i])
                # print (self.temperatures[i])
        else:

            tmpr_rate = (self.maxtemp / self.num_chains)
            temp = 1
            for i in range(0, self.num_chains):
                self.temperatures.append(temp)
                temp += tmpr_rate
                # print(self.temperatures[i])

    def initialize_chains(self, burn_in):
        self.burn_in = burn_in
        self.assign_temperatures()
        self.minlim_param = np.repeat([-100], self.num_param)  # priors for nn weights
        self.maxlim_param = np.repeat([100], self.num_param)
        for i in range(0, self.num_chains):
            w = np.ones(self.num_param)
            w = w * i
            # w = np.random.randn(self.num_param)
            # r1= -1
            # r2= 1
            # w = w[torch.randperm(w.size()[0])]
            w = self.cae.dictfromlist(w)
            self.chains.append(
                ptReplica(self.use_langevin_gradients, self.learn_rate, w, self.minlim_param, self.maxlim_param,
                          self.NumSamples, self.traindata, self.testdata, self.burn_in,
                          self.temperatures[i], self.swap_interval, self.path, self.parameter_queue[i],
                          self.wait_chain[i], self.event[i], self.batch_size, self.step_size ))

    def surr_procedure(self, queue):
        if queue.empty() is False:
            return queue.get()
        else:
            return

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        #        if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
        param1 = parameter_queue_1.get()
        param2 = parameter_queue_2.get()
        w1 = param1[0:self.num_param]
        w1 = self.cae.dictfromlist(w1)
        eta1 = param1[self.num_param]
        lhood1 = param1[self.num_param + 1]
        T1 = param1[self.num_param + 2]
        w2 = param2[0:self.num_param]
        w2 = self.cae.dictfromlist(w2)
        eta2 = param2[self.num_param]
        lhood2 = param2[self.num_param + 1]
        T2 = param2[self.num_param + 2]

        
        # SWAPPING PROBABILITIES
        #[lhood12, dump1, dump2] = ptReplica.likelihood_func(self.cae, self.traindata, w1, np.exp(eta1), T2)
        #[lhood21, dump1, dump2] = ptReplica.likelihood_func(self.cae, self.traindata, w2, np.exp(eta2), T1)

        lhood12 = 0
        lhood21 = 0
        try:
            swap_proposal = min(1, np.exp((lhood12 - lhood1) + (lhood21 - lhood2)))
        except OverflowError:
            swap_proposal = 1
        u = np.random.uniform(0, 1)
        if u < swap_proposal:
            swapped = True
            self.total_swap_proposals += 1
            self.num_swap += 1
            param_temp = param1
            param1 = param2
            param2 = param_temp
            param1[self.num_param + 1] = lhood21
            param1[self.num_param + 2] = T2
            param2[self.num_param + 1] = lhood12
            param2[self.num_param + 2] = T1

            print('  swap ')
        else:
            swapped = False
            self.total_swap_proposals += 1
        return param1, param2, swapped

    def run_chains(self):
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        # swap_proposal = np.ones(self.num_chains-1)
        # create parameter holders for paramaters that will be swapped
        # replica_param = np.zeros((self.num_chains, self.num_param))
        # lhood = np.zeros(self.num_chains)
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples - 1
        # number_exchange = np.zeros(self.num_chains)
        # filen = open(self.path + '/num_exchange.txt', 'a')
        # RUN MCMC CHAINS
        for l in range(0, self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        for j in range(0, self.num_chains):
            self.wait_chain[j].clear()
            self.event[j].clear()
            self.chains[j].start()
        # SWAP PROCEDURE
        swaps_affected_main = 0
        total_swaps = 0
        for i in range(int(self.NumSamples / self.swap_interval)):
            # print(i,int(self.NumSamples/self.swap_interval), 'Counting')
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count += 1
                    self.wait_chain[index].set()
                    # print(str(self.chains[index].temperature) + " Dead" + str(index))

            if count == self.num_chains:
                break
            # print(count,'Is the Count')
            timeout_count = 0
            for index in range(0, self.num_chains):
                # print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait()
                if flag:
                    # print("Signal from chain: {}".format(index+1))
                    timeout_count += 1

            if timeout_count != self.num_chains:
                # print("Skipping the Swap!")
                continue
            # print("Event Occured")
            for index in range(0, self.num_chains - 1):
                # print('Starting Swap')
                swapped = False
                param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],
                                                                self.parameter_queue[index + 1])
                self.parameter_queue[index].put(param_1)
                self.parameter_queue[index + 1].put(param_2)
                if index == 0:
                    if swapped:
                        swaps_affected_main += 1
                    total_swaps += 1
            for index in range(self.num_chains):
                self.wait_chain[index].clear()
                self.event[index].set()

        print("Joining Processes")

        # JOIN THEM TO MAIN PROCESS
        for index in range(0, self.num_chains):
            print('Waiting to Join ', index, self.num_chains)
            print(self.chains[index].is_alive())
            self.chains[index].join()
            print(index, 'Chain Joined')
        self.chain_queue.join()
        # pos_w, fx_train, fx_test, mse_train, mse_test, acc_train, acc_test, likelihood_vec, accept_vec, accept = self.show_results()
        mse_train, mse_test, acc_train, acc_test, apal = self.show_results()
        print("NUMBER OF SWAPS = ", self.num_swap)
        if self.total_swap_proposals == 0:
            self.total_swap_proposals = 1
        swap_perc = self.num_swap * 100 / self.total_swap_proposals
        # return pos_w, fx_train, fx_test, mse_train, mse_test, acc_train, acc_test, likelihood_vec, swap_perc, accept_vec, accept
        return mse_train, mse_test, acc_train, acc_test, apal, swap_perc

    def show_results(self):
        burnin = int(self.NumSamples * self.burn_in)
        mcmc_samples = int(self.NumSamples * 0.25)
        # likelihood_rep = np.zeros((self.num_chains, self.NumSamples - burnin,2))  # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        # accept_percent = np.zeros((self.num_chains, 1))
        # accept_list = np.zeros((self.num_chains, self.NumSamples))
        # pos_w = np.zeros((self.num_chains, self.NumSamples - burnin, self.num_param))
        # fx_train_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.traindata)))
        mse_train = np.zeros((self.num_chains, self.NumSamples))
        acc_train = np.zeros((self.num_chains, self.NumSamples))

        # fx_test_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.testdata)))
        mse_test = np.zeros((self.num_chains, self.NumSamples))
        acc_test = np.zeros((self.num_chains, self.NumSamples))
        sum_val_array = np.zeros((self.num_chains, self.NumSamples))
        likelihood_val_array = np.zeros((self.num_chains, self.NumSamples))
        likelihood_proposal_val_array = np.zeros((self.num_chains, self.NumSamples))
        u_value_all = np.zeros((self.num_chains, self.NumSamples))

        weight_ar = np.zeros((self.num_chains, self.NumSamples))
        weight_ar1 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar2 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar3 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar4 = np.zeros((self.num_chains, self.NumSamples))

        accept_percentage_all_chains = np.zeros(self.num_chains)

        for i in range(self.num_chains):
            # file_name = self.path + '/posterior/pos_w/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            # print(self.path)
            # print(file_name)
            # dat = np.loadtxt(file_name)
            # pos_w[i, :, :] = dat[burnin:, :]

            # file_name = self.path + '/posterior/pos_likelihood/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            # dat = np.loadtxt(file_name)
            # likelihood_rep[i, :] = dat[burnin:]

            # file_name = self.path + '/posterior/accept_list/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            # dat = np.loadtxt(file_name)
            # accept_list[i, :] = dat

            file_name = self.path + '/mse_test_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            mse_test[i, :] = dat

            file_name = self.path + '/mse_train_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            mse_train[i, :] = dat

            file_name = self.path + '/acc_test_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            acc_test[i, :] = dat

            file_name = self.path + '/acc_train_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            acc_train[i, :] = dat

            file_name = self.path + '/sum_value_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            sum_val_array[i, :] = dat

            file_name = self.path + '/likelihood_value_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            likelihood_val_array[i, :] = dat

            file_name = self.path + '/likelihood_value_proposal' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            likelihood_proposal_val_array[i, :] = dat

            file_name = self.path + '/u_value_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            u_value_all[i, :] = dat

            file_name = self.path + '/weight[0]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar[i, :] = dat

            file_name = self.path + '/weight[100]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar1[i, :] = dat

            file_name = self.path + '/weight[10000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar2[i, :] = dat

            file_name = self.path + '/weight[5000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar3[i, :] = dat

            file_name = self.path + '/accept_percentage' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            accept_percentage_all_chains[i] = dat

        mse_train_single_chain_plot = mse_train[0, :]
        mse_test_single_chain_plot = mse_test[0, :]
        acc_train_single_chain_plot = acc_train[0, :]
        acc_test_single_chain_plot = acc_test[0, :]
        sum_val_array_single_chain_plot = sum_val_array[0]
        likelihood_val_array_single_chain_plot = likelihood_val_array[0]
        likelihood_proposal_val_array_single_chain_plot = likelihood_proposal_val_array[0]
        u_value_single = u_value_all[0]

        # path = 'cifar_torch/CAE/graphs'

        x2 = np.linspace(0, self.NumSamples, num=self.NumSamples)

       

        plt.plot(x2, likelihood_val_array_single_chain_plot, label='Likelihood Value')
        # plt.legend(loc='upper right')
        plt.title("Likelihood Value Single Chain")
        plt.savefig(self.path + '/likelihood_value_single_chain.png')
        plt.clf()

        plt.plot(x2, likelihood_proposal_val_array_single_chain_plot, label='Proposed Likelihood Value')
        plt.title("Proposed Likelihood Value Single Chain")
        plt.savefig(self.path + '/likelihood_proposal_value_single_chain.png')
        plt.clf()
  

        color = 'tab:red'
        plt.plot(x2, acc_train_single_chain_plot, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x2, acc_test_single_chain_plot, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        # plt.legend()
        plt.savefig(self.path + '/superimposed_acc_single_chain.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x2, mse_train_single_chain_plot, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x2, mse_test_single_chain_plot, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('mse')
        # plt.legend()
        plt.savefig(self.path + '/superimposed_mse_single_chain.png')
        plt.clf()

        #print(mse_train, '  mse train')
       
       
        burn = int(self.NumSamples * self.burni)

        #print(mse_train[:,burn:], ' is burn ')

        mse_train = mse_train[:,burn:]
        mse_test = mse_test[:,burn:]
        acc_train = acc_train[:,burn:]
        acc_test = acc_test[:,burn:]


 
 

        '''sum_val_array = sum_val_array.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar = weight_ar.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar1 = weight_ar1.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar2 = weight_ar2.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar3 = weight_ar3.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar4 = weight_ar4.reshape((self.num_chains * self.NumSamples), 1)

        x = np.linspace(0, int(self.masternumsample - self.masternumsample * self.burni),
                        num=int(self.masternumsample - self.masternumsample * self.burni))
        x1 = np.linspace(0, self.masternumsample, num=self.masternumsample)

        plt.plot(x1, weight_ar, label='Weight[0]')
        plt.legend(loc='upper right')
        plt.title("Weight[0] Trace")
        plt.savefig(self.path + '/weight[0]_samples.png')
        plt.clf()

        plt.hist(weight_ar, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(self.path + '/weight[0]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar1, label='Weight[100]')
        plt.legend(loc='upper right')
        plt.title("Weight[100] Trace")
        plt.savefig(self.path + '/weight[100]_samples.png')
        plt.clf()

        plt.hist(weight_ar1, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(self.path + '/weight[100]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar2, label='Weight[10000]')
        plt.legend(loc='upper right')
        plt.title("Weight[10000] Trace")
        plt.savefig(self.path + '/weight[10000]_samples.png')
        plt.clf()

        plt.hist(weight_ar2, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(self.path + '/weight[10000]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar3, label='Weight[5000]')
        plt.legend(loc='upper right')
        plt.title("Weight[5000] Trace")
        plt.savefig(self.path + '/weight[5000]_samples.png')
        plt.clf()

        plt.hist(weight_ar3, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(self.path + '/weight[5000]_hist.png')
        plt.clf()
  

        color = 'tab:red'
        plt.plot(x1, acc_train, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x1, acc_test, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.path + '/superimposed_acc.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x1, mse_train, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x1, mse_test, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.path + '/superimposed_mse.png')
        plt.clf()'''
 
        # accept_vec = accept_list

        # accept = np.sum(accept_percent) / self.num_chains

        # np.savetxt(self.path + '/pos_param.txt', posterior.T)  # tcoment to save space

        # np.savetxt(self.path + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

        # np.savetxt(self.path + '/accept_list.txt', accept_list, fmt='%1.2f')

        # np.savetxt(self.path + '/acceptpercent.txt', [accept], fmt='%1.2f')

        # return posterior, fx_train_all, fx_test_all, mse_train, mse_test, acc_train, acc_test, likelihood_vec.T, accept_vec, accept
        return mse_train, mse_test, acc_train, acc_test, accept_percentage_all_chains


def main():
    numSamples = 48000 # int(input("Enter no of samples: "))
    # swap_interval = int(swap_ratio * numSamples / num_chains)
    problemfolder = 'results/Paper_Revision'
    description = ' swap interval ' + str(swap_interval)
    global shape

    if use_dataset == 1:
        shape = 28
        problemfolder += '/' + str(exp) + '_coil_  ' + str(numSamples) + str(description)
        PATH = 'saved_model' + 'SR.pt'
    elif use_dataset == 2:
        shape = 96
        problemfolder += '/autoencoder_' + str(exp) + '_Madelon_' + str(numSamples) + str(description)
        PATH = 'saved_model' + 'Madelon.pt'
    elif use_dataset == 3:
        problemfolder += '/autoencoder_' + str(exp) + '_Swiss Roll_' + str(numSamples) + str(description)

    os.makedirs(problemfolder)
    global outres
    outres = open(problemfolder + '/results.txt', 'w')

    time1 = 0 #time.now()

    pt = ParallelTempering(ulg, num_chains, maxtemp, numSamples,
                           swap_interval, problemfolder, batch_size, burnin, step_size)  # declare class
    pt.initialize_chains(burnin)
    mse_train, mse_test, acc_train, acc_test, accept_percent_all, sp = pt.run_chains()



    # acc_train, acc_test, mse_train, mse_test, sva, wa, wa1, wa2, wa3 = mcmc.sampler()

    time2 = 0 #time.now()

    time2 = (time2 - time1)
    outres.write("Time for sampling: " + str(time2))
    outres.write('\n')
    outres.write('\n')



  

    # sva = sva[int(numSamples * burnin):]
    # print(lpa)

    #x = np.linspace(0, int(numSamples - numSamples * burnin), num=int(numSamples - numSamples * burnin))
    #x1 = np.linspace(0, numSamples, num=numSamples)
 

    mse_tr = np.mean(mse_train)
    msetr_std = np.std(mse_train)
    msetr_max = np.amin(mse_train)

    

    mse_tes = np.mean(mse_test)
    msetest_std = np.std(mse_test)
    msetes_max = np.amin(mse_test)
 

    accept_percent_mean = np.mean(accept_percent_all)

    print("Train mse (Mean, Max, Std)")
    print(mse_tr, msetr_max, msetr_std)
    print("\n")
    print("Test mse (Mean, Max, Std)")
    print(mse_tes, msetes_max, msetest_std)
    print("\n")
    print("Acceptance Percentage Mean")
    print(accept_percent_mean)
    print("\n")


    '''acc_tr = np.mean(acc_train[int(numSamples * burnin):])
    acctr_std = np.std(acc_train[int(numSamples * burnin):])

    
    print(acc_train.shape, ' acc train   **************')
    acctr_max = np.amax(acc_train[int(numSamples * burnin):])

    acc_tes = np.mean(acc_test[int(numSamples * burnin):])
    acctest_std = np.std(acc_test[int(numSamples * burnin):])
    acctes_max = np.amax(acc_test[int(numSamples * burnin):])


    # outres = open(path+'/result.txt', "a+")
    # outres_db = open(path_db+'/result.txt', "a+")
    # resultingfile = open(problemfolder+'/master_result_file.txt','a+')
    # resultingfile_db = open( problemfolder_db+'/master_result_file.txt','a+')
    # xv = name+'_'+ str(run_nb)
    print("\n\n\n\n")
    print("Train Acc (Mean, Max, Std)")
    print(acc_tr, acctr_max, acctr_std)
    print("\n")
    print("Test Acc (Mean, Max, Std)")
    print(acc_tes, acctes_max, acctest_std)
    print("\n")
    print("Train mse (Mean, Max, Std)")
    print(mse_tr, msetr_max, msetr_std)
    print("\n")
    print("Test mse (Mean, Max, Std)")
    print(mse_tes, msetes_max, msetest_std)
    print("\n")
    print("Acceptance Percentage Mean")
    print(accept_percent_mean)
    print("\n")
    print("Swap Percentage")
    print(sp)
    print("\n")
    print("Time")
    print(time2)'''


if __name__ == "__main__": main()
