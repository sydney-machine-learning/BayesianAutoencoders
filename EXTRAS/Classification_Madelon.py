#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mahir
python 3.6 --  working great
cae with mcmc in torch
"""

#  %reset
#  %reset -sf


#import torchvision
#import torch.nn.functional as F
#import torchvision.transforms as transforms
#from torchvision import datasets
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
import times
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

device= 'cpu'

train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
madelon_train_sample = np.loadtxt(urllib2.urlopen(train_data_url))
madelon_train_sample = StandardScaler().fit_transform(madelon_train_sample)
madelon_train_sample = torch.from_numpy(madelon_train_sample).to(device)



train_data_labels_url= 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
madelon_train_sample_label = np.loadtxt(urllib2.urlopen(train_data_labels_url))

mad_X_train, mad_X_test, mad_y_train, mad_y_test = train_test_split(madelon_train_sample, \
                                                                            madelon_train_sample_label)
#using out of the box default parameters provided in scikit learn library
names_of_classifiers = ['LogisticRegression', 'KNeighbors', 'DecisionTree', 'SVClassifier']

classifiers = [
    LogisticRegression(n_jobs=-1, random_state=42),
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
print('Train', mad_raw_train_scores)
