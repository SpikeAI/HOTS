from mix_Network import network, histoscore
from Tools import netparam
name = 'homhots'
sigma = None
pooling = False
homeinv = False
jitonic = [None,None] #[temporal, spatial]
jitter = False
tau = 5
nblay = 3
nbclust = 4
filt = 2

nethots, homeotest = netparam(name, filt, tau, nblay, nbclust, sigma, homeinv, jitter)

import numpy as np
jit_s = np.arange(0,6,0.2)
jit_t = np.arange(0,300,10)
jit_s, jit_t = jit_s**2, jit_t**2
#jit_s = [0]
#jit_t = [0]
nb_test = 10000
nb_train = 60000

import pickle
timestr = '2021-03-01'
for tau in range(1,10):
    for name in ['hots','homhots', 'onlyonline']:
        nethots, homeotest = netparam(name, filt, tau, nblay, nbclust, sigma, homeinv, jitter)
        trainhistomap = nethots.running(homeotest=homeotest, nb_digit = nb_train)
        score_S, score_T = runjit(nethots, jit_s, jit_t, homeotest, trainhistomap, nb_test)
        f_name = f'../Records/EXP_03_NMNIST/{timestr}_results_jitter_histo_{name}.pkl'
        with open(f_name, 'wb') as file:
            pickle.dump([score_T, jit_t, score_S, jit_s], file, pickle.HIGHEST_PROTOCOL)
