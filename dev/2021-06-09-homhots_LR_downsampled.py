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
    subset_size = None

    #_______________NB_OF_DIGITS___________________
    dataset = 'nmnist'
    nb_test = 10000
    nb_train = 60000
    ds = 1
    nb_test = nb_test//ds
    nb_train = nb_train//ds
    print(f'training set size: {nb_train} - testing set: {nb_test}')
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
    ds_ev_train = 1
    ds_ev_test = 1
    tau_cla = 150000

    timestr = '2021-03-29'
    path = '../Records/EXP_03_NMNIST/'

    for sample_space in [2, 4, 8, 16]:
        for name in ['homhots']:
            f_name = f'{path}{timestr}_LR_results_{name}_{nbclust}_{nb_train}_{nb_test}_{ds_ev_test}_{sample_space}_timescale.pkl'
            if isfile(f_name):
                with open(f_name, 'rb') as file:
                    likelihood, true_target, time_scale = pickle.load(file)
            else:
                ds_ev = ds_ev_train
                print(f'LR fit for {name}...')
                model, loss  = fit_data(name,timestr,path,filt,tau,R,nbclust,sigma,homeinv,jitter,dataset,nb_train, ds_ev,learning_rate,num_epochs,betas,tau_cla,sample_space=sample_space,jitonic=jitonic,subset_size=nb_train,num_workers=num_workers,verbose=False)
                ds_ev = ds_ev_test
                print(f'prediction for {name}...')
                likelihood, true_target, time_scale = predict_data(model,name,timestr,path,filt,tau,R,nbclust,sigma, homeinv,jitter,dataset,nb_test,ds_ev,tau_cla,sample_space=sample_space,jitonic=jitonic,subset_size=nb_test,num_workers=num_workers, verbose=False)
                with open(f_name, 'wb') as file:
                    pickle.dump([likelihood, true_target, time_scale], file, pickle.HIGHEST_PROTOCOL)
    #return likelihood, true_target, time_scale