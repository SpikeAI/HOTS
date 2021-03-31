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
dataset = 'gesture'

nb_test = 264
nb_train = 1077
ds = 10
max_nbevents = 50000
nb_test = nb_test//ds
nb_train = nb_train//ds
print(f'training set size: {nb_train} - testing set: {nb_test}')
#______________________________________________

timestr = '2021-03-29'
record_path = '../Records/EXP_06_DVSGESTURE/'

print('classic HOTS and homeoHOTS')
#for name in ['homhots', 'hots']:
name = 'homhots'
meanscore = []
torange = [0.1, 1, 2, 5, 10, 20]
for tau in torange:
    print('clustering...')
    hotshom, homeotest = netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, nb_learn=5, maxevts  = max_nbevents)
    print('training...')
    #trainhistomap = hotshom.running(homeotest=homeotest, nb_digit = nb_train, outstyle='LR')
    trainhistomap = hotshom.running(homeotest=homeotest, nb_digit = nb_train, outstyle='histo', dataset=dataset, maxevts  = max_nbevents)
    print('testing...')
    testhistomap = hotshom.running(homeotest = homeotest, train=False, nb_digit=nb_test, jitonic=jitonic, dataset=dataset, maxevts  = max_nbevents)
    score = histoscore(trainhistomap,testhistomap, verbose = True)
    meanscore.append(np.mean(score))
    
ind_tmax = np.argmax(meanscore)

tau = torange(ind_tmax)
for R in [1, 2, 5, 10]:
    print('clustering...')
    hotshom, homeotest = netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, nb_learn=5, maxevts  = max_nbevents)
    print('training...')
    #trainhistomap = hotshom.running(homeotest=homeotest, nb_digit = nb_train, outstyle='LR')
    trainhistomap = hotshom.running(homeotest=homeotest, nb_digit = nb_train, outstyle='histo', dataset=dataset, maxevts  = max_nbevents)
    print('testing...')
    testhistomap = hotshom.running(homeotest = homeotest, train=False, nb_digit=nb_test, jitonic=jitonic, dataset=dataset, maxevts  = max_nbevents)
    score = histoscore(trainhistomap,testhistomap, verbose = True)
    meanscore.append(np.mean(score))
