import numpy as np
import sys
sys.path.append('../HOTS')
from Tools import tic, toc, get_loader, fit_data, predict_data, classification_results, netparam
import pickle
from os.path import isfile

if __name__ == '__main__':
    #_________NETWORK_PARAMETERS______________________
    #______________________________________________
    sigma = None
    pooling = False
    homeinv = False
    jitonic = [None,None] #[temporal, spatial]
    jitter = False
    tau = 5
    R = 2
    filt = 2
    nbclust = [16,8,16]

    #_______________NB_OF_DIGITS___________________
    dataset = 'cars'
    nb_learn = 50
    nb_test = 4396 + 4211
    nb_train = 7940 + 7482
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
    ds_ev_test = 1
    tau_cla = 50000

    timestr = '2021-03-29'
    record_path = '../Records/EXP_04_NCARS/'

    for name in ['homhots','raw']:
        f_name = f'{record_path}{timestr}_LR_results_{name}_{nbclust}_{nb_train}_{nb_test}_{ds_ev_test}_timescale.pkl'
        if isfile(f_name):
            with open(f_name, 'rb') as file:
                likelihood, true_target, timescale = pickle.load(file)
        
        else:
            ds_ev = 10
            print(f'LR fit for {name}...')
            subset_size = nb_train
            model, loss  = fit_data(name,timestr,record_path,filt,tau,R,nbclust,sigma,homeinv,jitter,dataset,nb_train, ds_ev,learning_rate,num_epochs,betas,tau_cla,jitonic=jitonic,subset_size=subset_size,num_workers=num_workers, nb_learn = nb_learn,verbose=False)
            ds_ev = ds_ev_test
            subset_size = nb_test
            print(f'prediction for {name}...')
            likelihood, true_target, time_scale = predict_data(model,name,timestr,record_path,filt,tau,R,nbclust,sigma, homeinv, jitter,dataset,nb_test,ds_ev,tau_cla,jitonic=jitonic,subset_size=subset_size,num_workers=num_workers, verbose=False)
            with open(f_name, 'wb') as file:
                pickle.dump([likelihood, true_target, time_scale], file, pickle.HIGHEST_PROTOCOL)

        meanac, onlinac, lastac, truepos, falsepos = classification_results(likelihood, true_target, None, nb_test, 1/2)
        print(f'Mean accuracy for {name}:{meanac}')
        print(f'Accuracy for {name} at the last event:{lastac}')
    #return likelihood, true_target, time_scale