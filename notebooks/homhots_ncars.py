import sys
sys.path.append('../HOTS')
from Tools import netparam, histoscore
import numpy as np

#_________NETWORK_PARAMETERS___________________
#______________________________________________
sigma = None
pooling = False
homeinv = False
jitonic = [None,None] #[temporal, spatial]
jitter = False
tau = 5
R = 2
filt = 2
nbclust = [4,8,16]
#______________________________________________
#______________________________________________

#_______________JITTER_________________________
#jit_s = np.arange(0,6,0.2)
#jit_t = np.array([0.0])
#jit_t = np.arange(0,300,10)
#jit_s, jit_t = jit_s**2, jit_t**2
#______________________________________________

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

timestr = '2021-03-29'
record_path = '../Records/EXP_04_NCARS/'

print('classic HOTS and homeoHOTS')
for name in ['homhots', 'hots']:
    print(f'{name} clustering...')
    hotshom, homeotest = netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, nb_learn=nb_learn)
    print(f'{name} training...')
    trainhistomap = hotshom.running(homeotest=homeotest, nb_digit=nb_train, outstyle='histo', dataset=dataset)
    print(f'{name} testing...')
    testhistomap = hotshom.running(homeotest=homeotest, train=False, nb_digit=nb_test, outstyle='histo', dataset=dataset)
    print('KFolds')
    kfold = 20
    nb_test = 4396 + 4211
    nb_test = nb_test//kfold
    for kfold_ind in range(kfold):
        testhistomap = hotshom.running(homeotest=homeotest, train=False, nb_digit=nb_test, outstyle='histo', dataset=dataset, kfold = kfold, kfold_ind = kfold_ind)