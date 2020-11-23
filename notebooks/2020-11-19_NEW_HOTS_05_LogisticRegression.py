#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import time
#
# https://en.wikipedia.org/wiki/ISO_8601
import datetime
timestr = datetime.datetime.now().date().isoformat()
timestr = '2020-11-03'
tau = 9e-4

import sys
sys.path.append('..')
from HOTS.Event import conv2eve
from HOTS.Tools import SaveObject, LoadObject

ds = 10
ds = 4
NbClusteringData = 1600//ds
NbTrainingData = 1600//ds
NbTestingData = 1600//ds

records_path = '../Records'

def get_nmnist(timestr, NbTrainingData, NbTestingData, NbClusteringData, DataPath='../Data/testsetnmnist.p', verbose=False):
    fname_event_nmnist = f'{records_path}/EXP_03_NMNIST/{timestr}_{NbTrainingData}_{NbTestingData}_{NbClusteringData}_hots_event_nmnist.pkl'
    # print(help(LoadNMNIST))
    try:
        dataset = LoadObject(fname_event_nmnist)
        if verbose: print('loading the events from file', fname_event_nmnist)
    except:
        from HOTS.Event import LoadNMNIST
        dataset = LoadNMNIST(NbTrainingData, NbTestingData, NbClusteringData,
                             Path=DataPath, verbose=0)
        SaveObject(dataset, fname_event_nmnist)
        if verbose: print('saving the events to file', fname_event_nmnist)

    events_train, events_test, event_cluster, labels_train, labels_test = dataset
    return events_train, events_test, event_cluster, labels_train, labels_test


def get_events(timestr, NbTrainingData=NbTrainingData, NbTestingData=NbTestingData, NbClusteringData=NbClusteringData,
               tau=tau, N_layer=3, # -> tau=1ms, si on prend 10 ms on est à 1s pour la dernière couche et les vidéos font 0.3s en moyenne
               homeo=True, homrun = False, verbose=False):
    R = 2
    filthr = 2
    nbkNN = 3
    algo = 'lagorce'
    decay = 'exponential'
    krnlinit = 'rdn'
    #nb_cluster = [4, 8, 16]
    nb_cluster = [4*(2**i_layer) for i_layer in range(N_layer)]

    fname_ = f'{records_path}/EXP_03_NMNIST/{timestr}_hots_{tau*1000}'

    label = '_homeo' if homeo else ''

    fname_model = fname_ + 'ms_' + algo + label + '.pkl'
    fname_event0_o = fname_ + '_event_out_' + algo + label + '.pkl'

    if not os.path.isfile(fname_event0_o):
        if verbose: print('creating the events in file', fname_event0_o)
        from HOTS.Event import Event, LoadNMNIST

        events_train, events_test, event_cluster, labels_train, labels_test = get_nmnist(timestr, NbTrainingData, NbTestingData, NbClusteringData)

        from HOTS.Layer import ClusteringLayer
        opts_layer = dict(verbose=0, ThrFilter=filthr, LearningAlgo=algo, kernel=decay, homeo=homeo, init=krnlinit)
        layers = []
        for i_layer in range(N_layer):
            layers.append(ClusteringLayer(tau=(10**i_layer)*tau, R=(2**i_layer)*R, **opts_layer))
        #L1 = ClusteringLayer(tau=tau, R=R, **opts_layer)
        #L2 = ClusteringLayer(tau=10 * tau, R=2 * R, , **opts_layer)
        #L3 = ClusteringLayer(tau=10 * 10 * tau, R=2 * 2 * R, **opts_layer)

        from HOTS.Network import Network
        Net = Network(layers)#[L1, L2, L3])

        if not os.path.isfile(fname_model):
            if verbose: print('learning model', fname_model)
            ClusterLayer, event_output = Net.TrainCluster(
                    event=event_cluster, NbClusterList=nb_cluster, to_record=True, NbCycle=1
                )
            SaveObject(ClusterLayer, fname_model)
        else:
            if verbose: print('loading model from file', fname_model)

            ClusterLayer, Classif0 = LoadObject(fname_model)

        if verbose: print('run the events through the network')

        events_train_o = Net.RunNetwork(events_train, NbClusterList=ClusterLayer, homrun=homrun)
        events_test_o = Net.RunNetwork(events_test, NbClusterList=ClusterLayer, homrun=homrun)
        SaveObject([events_train, events_test], fname_event0_o)
    else:
        if verbose: print('loading the events from file', fname_event0_o)
        events_train_o, events_test_o = LoadObject(fname_event0_o)

    return events_train_o, events_test_o


# ### Building matrix for logistic regression

# In[6]:


def gather_data(events_in, labels_in,
                tau_cla=.150, # characteristic time of a digit
                sample_events=200, sample_space = 1,
                verbose=False, debug=False):

    c_int = lambda n, d : ((n - 1) // d) + 1

    n_events = events_in.time.shape[0]

    data = np.zeros((c_int(events_in.ImageSize[0], sample_space),
                     c_int(events_in.ImageSize[1], sample_space),
                     len(events_in.ListPolarities))) #tmp data

    X = np.zeros((c_int(n_events, sample_events), len(data.ravel())))
    y = np.zeros((c_int(n_events, sample_events), ))

    for i_event in range(1, n_events):

        data *= np.exp(-(events_in.time[i_event]-events_in.time[i_event-1])/tau_cla)

        x_pos = events_in.address[i_event, 0]//sample_space
        y_pos = events_in.address[i_event, 1]//sample_space
        p = events_in.polarity[i_event]
        data[x_pos, y_pos, p] = 1.

        if i_event % sample_events == sample_events//2 :
            if debug:
                print(f'DEBUG {i_event=} {i_event//sample_events=} ')
                print(f'DEBUG {y[i_event//sample_events]=}   ')
                print(f'DEBUG  {labels_in[i_event]=} ')
            X[i_event//sample_events, :] = data.ravel()
            y[i_event//sample_events] = labels_in[i_event]


    if verbose: print('Number of events: ' + str(X.shape[0])+' - Number of features: ' + str(X.shape[1]))

    return X, y


# ### Performing logistic regression
#
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
#

# In[7]:


from sklearn.linear_model import LogisticRegression as LR
# from sklearn.linear_model import LogisticRegressionCV as LR
# from sklearn.model_selection import train_test_split

opts_LR = dict(max_iter=400//ds, # random_state=0,
               n_jobs=32, class_weight='balanced', verbose=2)
#opts_LR['Cs'] = 5
#opts_LR['Cs'] = 32


# ### Performing logistic regression on raw input

# In[8]:


def tic():
    global ttic
    ttic = time.time()
def toc():
    print(f'Done in {time.time() - ttic:.3f} s')


verbose=True


#for homeo in [False, True]:
for homeo in [True, False]:
    print(40*'-')
    print(f'{homeo=}')
    print(40*'-')
    events_train_o, events_test_o = get_events(timestr, tau=tau, homeo=homeo, verbose=verbose)

    X, y = gather_data(events_train_o, labels_train, verbose=verbose)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    lr = LR(**opts_LR).fit(X_train, y_train)
    print(f'Classification score for {homeo=} is {lr.score(X_train, y_train):.3f}')
    print(f'Classification score for {homeo=} is {lr.score(X_test, y_test):.3f}')
    X_test, y_test = gather_data(events_test_o, labels_test, verbose=verbose)
    print(f'Classification score for {homeo=} is {lr.score(X_test, y_test):.3f}')

    
tauz = np.array([1e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,1.5e-3,2e-3,2.5e-3,3e-3,4e-3,5e-3,1e-2,2e-2])
for homeo in [False, True]:
    print(40*'-')
    print(f'{homeo=}')
    print(40*'-')
    for tau_ in tauz:
        events_train_o, events_test_o = get_events(timestr, tau=tau_, homeo=homeo, verbose=verbose)    