import numpy as np
import sys
sys.path.append('../HOTS')
from Tools import fit_data, predict_data, classification_results, netparam, knn
import pickle
from os.path import isfile

#_________NETWORK_PARAMETERS___________________
name = 'homhots'
sigma = None
pooling = False
homeinv = False
jitonic = [None,None] #[temporal, spatial]
jitter = False
tau = 5
R = 2
nbclust = [4, 8, 16]
filt = 2
#______________________________________________

#_______________JITTER_________________________
std_jit_t = np.logspace(3,7,20)
nb_class = 10
#______________________________________________

#_______________NB_OF_DIGITS___________________
dataset = 'nmnist'
nb_test = 10000
nb_train = 60000
ds = 10
nb_test = nb_test//ds
#nb_train = nb_train//ds
print(f'training set size: {nb_train} - testing set: {nb_test}')
subset_size = nb_test
nb_trials = 10
#______________________________________________

#_______________LR_PARAMETERS__________________
num_workers = 0
learning_rate = 0.005
beta1, beta2 = 0.9, 0.999
betas = (beta1, beta2)
num_epochs = 2 ** 5 + 1
#num_epochs = 2 ** 9 + 1
print(f'number of epochs: {num_epochs}')
#______________________________________________
path = '../Records/EXP_03_NMNIST/'
timestr = '2021-03-29'
thres = None
tau_cla = 150000

for name in ['homhots', 'hots']:
    jitonic = [None, None]
    ds_ev = 10
    hotshom, homeotest = netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R)
    for trial in [1,2,3]:
        timestr_trial = '2021-03-29_'+str(trial)
        hotshom.date = timestr_trial
        for id_jit, jit_t in enumerate(std_jit_t):
            jit_t = int(round(jit_t,0))
            jitonic = [jit_t,None]
            if jit_t==0:
                jitonic = [None,None]
            testhistomap = hotshom.running(homeotest = homeotest, train=False, nb_digit=nb_test, jitonic=jitonic, outstyle = 'LR', subset_size = nb_test, verbose = True)
