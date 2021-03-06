import numpy as np
import sys
sys.path.append('../HOTS')
from Tools import tic,toc, get_loader, fit_data, predict_data, classification_results

#_________NETWORK_PARAMETERS______________________
#______________________________________________
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

#______________________________________________
#______________________________________________

#_______________JITTER_________________________
jit_s = np.arange(0,6,0.2)
jit_t = np.arange(0,300,10)
jit_s, jit_t = jit_s**2, jit_t**2
#______________________________________________

#_______________NB_OF_DIGITS___________________
nb_test = 10000
nb_train = 60000
ds = 1200
#ds = 1200
nb_test = nb_test//ds
nb_train = nb_train//ds
print(f'training set size: {nb_train} - testing set: {nb_test}')
#______________________________________________

#_______________LR_PARAMETERS__________________
learning_rate = 0.005
beta1, beta2 = 0.9, 0.999
betas = (beta1, beta2)
num_epochs = 2 ** 5 + 1
#num_epochs = 2 ** 9 + 1
print(f'number of epochs: {num_epochs}')
#______________________________________________

timestr = '2021-02-16'
record_path = '../Records/EXP_03_NMNIST/models/'

for name in ['raw','hots','homhots']:
    tic()
    learn_set, nb_pola, name_net = get_loader(name, record_path, nb_train, True, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr)
    model, loss = fit_data(name_net, learn_set, nb_train, nb_pola, learning_rate, num_epochs, betas, verbose=True)
    test_set, nb_pola, name_net = get_loader(name, record_path, nb_test, False, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr)
    pred_target, true_target = predict_data(test_set, model, nb_test)
    mean_acc, online_acc = classification_results(pred_target, true_target, nb_test)
    toc()
    print(f'Classification performance for {name}: {mean_acc}')
    
homhots_jit_s = []
hots_jit_s = []
homhots_jit_t = []
hots_jit_t = []
for name in ['homhots', 'hots']:
    learn_set, nb_pola, name_net = get_loader(name, record_path, nb_train, True, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr)
    model, loss = fit_data(name_net, learn_set, nb_train, nb_pola, learning_rate, num_epochs, betas, verbose=True)
    for j_s in jit_s:
        jitonic = [None,j_s]
        test_set, nb_pola, name_net = get_loader(name, record_path, nb_test, False, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr, jitonic = jitonic)
        pred_target, true_target = predict_data(test_set, model, nb_test)
        mean_acc, online_acc = classification_results(pred_target, true_target, nb_test)
        if name=='homhots':
            homhots_jit_s.append(mean_acc)
        else:
            hots_jit_s.append(mean_acc)
            
    for j_t in jit_t:
        jitonic = [j_t, None]
        test_set, nb_pola, name_net = get_loader(name, record_path, nb_test, False, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr, jitonic = jitonic)
        pred_target, true_target = predict_data(test_set, model, nb_test)
        mean_acc, online_acc = classification_results(pred_target, true_target, nb_test)
        if name=='homhots':
            homhots_jit_t.append(mean_acc)
        else:
            hots_jit_t.append(mean_acc)

path = '../Records/EXP_03_NMNIST/'
f_name = f'{path}{hotshom.get_fname()}_jitter_LR_{nb_train}_{nb_test}.pkl'

with open(f_name, 'wb') as file:
    pickle.dump([jit_t, homhots_jit_t, hots_jit_t, jit_s, homhots_jit_s, hots_jit_s], file, pickle.HIGHEST_PROTOCOL)