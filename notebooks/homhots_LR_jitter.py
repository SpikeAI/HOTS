import numpy as np
import sys
sys.path.append('../HOTS')
from Tools import tic, toc, get_loader, fit_data, predict_data, classification_results, netparam
import pickle
from os.path import isfile

if __name__ == '__main__':
    #_________NETWORK_PARAMETERS______________________
    #______________________________________________
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
    #_______________JITTER_________________________
    jit_s = np.arange(0,40,1)
    jit_t = np.arange(0,9000,100)
    #______________________________________________

    #_______________NB_OF_DIGITS___________________
    dataset = 'nmnist'
    nb_test = 10000
    nb_train = 60000
    ds = 1000
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


    timestr = '2021-03-29'
    path = '../Records/EXP_03_NMNIST/'
    
    thres = None
    ds_ev = 10
    tau_cla = 150000
    
    for name in ['homhots']:
        f_name = f'{path}{timestr}_LR_results_jitter_{name}_{nbclust}_{nb_train}_{nb_test}_{ds_ev}_{thres}.pkl'
        if isfile(f_name):
            with open(f_name, 'rb') as file:
                likelihood, true_target = pickle.load(file)
        else:
            print(f'LR fit for {name}...')
            model, loss  = fit_data(name,timestr,path,filt,tau,R,nbclust,sigma,homeinv,jitter,dataset,nb_train, ds_ev,learning_rate,num_epochs,betas, tau_cla,jitonic=jitonic,subset_size=nb_train,num_workers=num_workers,verbose=False)
            
            ds_ev = 1
            results_s, results_s_last = np.zeros([2, nb_trials, len(jit_s)])
            results_t, results_t_last = np.zeros([2, nb_trials, len(jit_t)])
            for trial in range(nb_trials):
                timestr = '2021-03-29_'+str(trial)
                id_jit = 0
                for id_jit, i in enumerate(jit_s):
                    i = round(i,2)
                    jitonic = [None,i]
                    if i==0:
                        jitonic = [None,None]
                    likelihood, true_target, timescale = predict_data(model,name,timestr,path,filt,tau,R,nbclust,sigma, homeinv, jitter,dataset,nb_test,ds_ev,tau_cla,jitonic=jitonic,subset_size=nb_test,num_workers=num_workers, verbose=False)
                    meanac, onlinac, lastac, truepos, falsepos = classification_results(likelihood, true_target, thres, nb_test)
                    results_s[trial,id_jit] = meanac
                    results_s_last[trial,id_jit] = lastac
                    print(jitonic, meanac, lastac)
                    
                for id_jit, j in enumerate(jit_t):
                    j = round(j,0)
                    jitonic = [j,None]
                    if j==0:
                        jitonic = [None,None]
                    likelihood, true_target, timescale = predict_data(model,name,timestr,path,filt,tau,R,nbclust,sigma, homeinv, jitter,dataset,nb_test,ds_ev,tau_cla,jitonic=jitonic,subset_size=nb_test,num_workers=num_workers, verbose=False)
                    meanac, onlinac, lastac, truepos, falsepos = classification_results(likelihood, true_target, thres, nb_test)
                    results_t[trial,id_jit] = meanac
                    results_t_last[trial,id_jit] = lastac
                    print(jitonic, meanac, lastac)
            with open(f_name, 'wb') as file:
                pickle.dump([results_s, results_t, results_s_last, results_t_last], file, pickle.HIGHEST_PROTOCOL)