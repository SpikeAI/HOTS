import numpy as np
import sys
sys.path.append('../HOTS')
from Tools import tic,toc, get_loader, fit_data, predict_data, classification_results, netparam
import pickle

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
    jit_s = np.arange(0,10,0.5)
    jit_t = np.arange(0,300,10)
    jit_s, jit_t = jit_s**2, jit_t**2
    #______________________________________________

    #_______________NB_OF_DIGITS___________________
    dataset = 'nmnist'
    nb_test = 10000
    nb_train = 60000
    ds = 10
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
    
    timestr = '2021-03-29'
    ds_ev_output = 1
    record_path = '../Records/EXP_03_NMNIST/models/'
    
    nb_trials = 10
    results_s = np.zeros([nb_trials, len(jit_s)])
    results_t = np.zeros([nb_trials, len(jit_t)])
    

    for name in ['homhots']:
        #learn_set, nb_pola, name_net = get_loader(name, record_path, nb_train, True, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, ds_ev = ds_ev_output)
        param = [0.25, 1]
        tau_nm = [5.0, 50.0, 500.0]
        R_nm = [2, 4, 8]
        name_net = f'../Records/EXP_03_NMNIST/models/2021-03-29_lagorce_rdn_None_True_{param}_{nbclust}_{tau_nm}_{R_nm}_False_LR_60000.pkl'
        with open(name_net, 'rb') as file:
            model, loss = pickle.load(file)
        
        #model, loss = fit_data(name_net, learn_set, nb_train, nb_pola, learning_rate, num_epochs, betas, num_workers=num_workers, verbose=True)

        for trial in [1]:
            timestr = '2021-03-29_'+str(trial)
            id_jit = 0
            for i in [jit_s[-1]]:
                i = round(i,2)
                jitonic = [None,i]
                test_set, nb_pola, name_net = get_loader(name, record_path, nb_test, False, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, subset_size = nb_test, jitonic = jitonic, ds_ev = ds_ev_output)
                print(f'prediction for {name}...')
                pred_target, true_target = predict_data(test_set, model, nb_test, num_workers=num_workers)
                meanac, onlinac, lastac = classification_results(pred_target, true_target, nb_test)
                print(f'Classification performance for {name}: {meanac}')
                results_s[trial,id_jit] = meanac
                id_jit += 1
                
            id_jit = 0
            for j in jit_t:
                j = round(j,0)
                jitonic = [j,None]
                test_set, nb_pola, name_net = get_loader(name, record_path, nb_test, False, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, subset_size = nb_test, jitonic = jitonic, ds_ev = ds_ev_output)
                print(f'prediction for {name}...')
                pred_target, true_target = predict_data(test_set, model, nb_test, num_workers=num_workers)
                meanac, onlinac, lastac = classification_results(pred_target, true_target, nb_test)
                print(f'Classification performance for {name}: {meanac}')
                results_t[trial,id_jit] = meanac

    path = '../Records/EXP_03_NMNIST/'
    f_name = f'{path}{timestr}_LR_results_jitter_{nbclust}_{nb_train}_{nb_test}_{ds_ev_output}.pkl'

    with open(f_name, 'wb') as file:
        pickle.dump([results_s, results_t], file, pickle.HIGHEST_PROTOCOL)
