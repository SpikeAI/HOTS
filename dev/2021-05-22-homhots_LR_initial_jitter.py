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
jitonic = [40000, None] #[temporal, spatial]
jitter = False
tau = 5
R = 2
nbclust = [4, 8, 16]
filt = 2
#______________________________________________

#_______________JITTER_________________________
std_jit_t = np.logspace(3,7,20)
std_jit_s = np.arange(0,10,0.5)
var_jit_s = std_jit_s**2
nb_class = 10
#______________________________________________

#_______________NB_OF_DIGITS___________________
dataset = 'nmnist'
nb_test = 10000
nb_train = 60000
ds = 10
nb_test = nb_test//ds
nb_train = nb_train//ds
print(f'training set size: {nb_train} - testing set: {nb_test}')
subset_size = nb_test
nb_trials = 1
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
timestr = '2021-05-27'
thres = None
tau_cla = 150000

for name in ['homhots', 'hots']:
    jitonic = [40000,None]
    ds_ev = 10
    model, loss  = fit_data(name,timestr,path,filt,tau,R,nbclust,sigma,homeinv,jitter,dataset,nb_train, ds_ev,learning_rate,num_epochs,betas, tau_cla, jitonic=jitonic, subset_size=nb_train, num_workers=num_workers, verbose=False)
    ds_ev = 1
    results_t, results_t_last = np.zeros([2, nb_trials, len(std_jit_t)])
    results_s, results_s_last = np.zeros([2, nb_trials, len(std_jit_s)])
    for trial in range(nb_trials):
        timestr_trial = '2021-05-27_'+str(trial)
        for id_jit, jit_t in enumerate(std_jit_t):
            jit_t = int(round(jit_t,0))
            jitonic = [jit_t,None]
            if jit_t==0:
                jitonic = [None,None]
            likelihood, true_target, timescale = predict_data(model,name,timestr_trial,path,filt,tau,R,nbclust,sigma, homeinv, jitter,dataset,nb_test,ds_ev,tau_cla,jitonic=jitonic,subset_size=nb_test,num_workers=num_workers, verbose=False)
            meanac, onlinac, lastac, truepos, falsepos = classification_results(likelihood, true_target, thres, nb_test, 1/nb_class)
            results_t[trial,id_jit] = meanac
            results_t_last[trial,id_jit] = lastac
            print(jitonic, meanac, lastac)
            
    f_name = f'{path}{timestr}_LR_results_jitter_{name}_{nb_train}_{nb_test}_{ds_ev}_{thres}_initjitter40ms.pkl'
    with open(f_name, 'wb') as file:
        pickle.dump([results_t, results_t_last, std_jit_t], file, pickle.HIGHEST_PROTOCOL)
