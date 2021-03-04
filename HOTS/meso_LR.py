from Tools import tic,toc, get_loader, fit_data, predict_data

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
timestr = '2021-03-01'

#______________________________________________
#______________________________________________

#_______________JITTER_________________________
#jit_s = np.arange(0,6,0.2)
#jit_t = np.arange(0,300,10)
#jit_s, jit_t = jit_s**2, jit_t**2
#______________________________________________

#_______________NB_OF_DIGITS___________________
nb_test = 10000
nb_train = 60000
ds = 120
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

for name in ['raw','hots','homhots']:
    tic()
    learn_set, nb_pola, name_net = get_loader(name, nb_train, True, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr)
    model, loss = fit_data(name_net, learn_set, nb_train,nb_pola,learning_rate,num_epochs,betas,verbose=True)
    test_set, nb_pola, name_net = get_loader(name, nb_test, False, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr)
    pred_target, true_target = predict_data(test_set, model)
    mean_acc, online_acc = classification_results(pred_target, true_target, nb_test)
    toc()
    print(f'Classification performance for {name}: {mean_acc}')