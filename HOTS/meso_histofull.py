from Tools import runjit
import numpy as np

#_________NETWORK_PARAMETERS___________________
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
jit_s = np.arange(0,6,0.2)
jit_t = np.arange(0,300,10)
jit_s, jit_t = jit_s**2, jit_t**2
#______________________________________________

#_______________NB_OF_DIGITS___________________
nb_test = 10000
nb_train = 60000
ds = 120
nb_test = nb_test//ds
nb_train = nb_train//ds
print(f'training set size: {nb_train} - testing set: {nb_test})
#______________________________________________

timestr = '2021-02-16'

score_T, jit_t, score_S, jit_s = runjit(timestr, 
                                        name, 
                                        filt, 
                                        tau, 
                                        nblay, 
                                        nbclust, 
                                        sigma, 
                                        homeinv, 
                                        jitter, 
                                        jit_s, 
                                        jit_t, 
                                        nb_train, 
                                        nb_test, 
                                        verbose=True
                                       )