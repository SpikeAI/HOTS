import sys
sys.path.append('../HOTS')
from Tools import netparam, histoscore, histoscore_lagorce, knn
import numpy as np
import pickle

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
kNN = 12
#______________________________________________
#______________________________________________

#_______________JITTER_________________________
jit_s = np.arange(0,10,0.5)
jit_t = np.arange(0,100000,5000)
jit_s = jit_s**2
#______________________________________________

#_______________NB_OF_DIGITS___________________
dataset = 'nmnist'
nb_test = 10000
nb_train = 60000
ds = 10
nb_test = nb_test//ds
nb_trials = 10
#nb_train = nb_train//ds
print(f'training set size: {nb_train} - testing set: {nb_test}')
#______________________________________________

timestr = '2021-03-29'
record_path = '../Records/EXP_03_NMNIST/'

print('classic HOTS and homeoHOTS')
for name in [ 'hots', 'homhots']:
    
    score_s1 = np.zeros([nb_trials, len(jit_s)])
    score_t1 = np.zeros([nb_trials, len(jit_t)])
    score_s12 = np.zeros([nb_trials, len(jit_s)])
    score_t12 = np.zeros([nb_trials, len(jit_t)])
    
    
    f_name = f'{record_path}{timestr}_results_jitter_{nb_test}_histo_{name}_std.pkl'
    print(f'{name} clustering...')
    hotshom, homeotest = netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R)
    print(f'{name} training...')
    trainhistomap = hotshom.running(homeotest=homeotest, nb_digit=nb_train, outstyle='histo')
    print(f'{name} testing...')

    for trial in range(nb_trials):
        hotshom.date = '2021-03-29_'+str(trial)
        id_jit = 0
        for i in jit_s:
            i = round(i,2)
            jitonic = [None,i]
            if i==0:
                jitonic = [None,None]
            testhistomap = hotshom.running(homeotest = homeotest, train=False, nb_digit=nb_test, jitonic=jitonic, outstyle = 'histo', subset_size = nb_test)
            
            kNN12_score = knn(trainhistomap,testhistomap, k = kNN, weights = 'distance')
            kNN1_score = knn(trainhistomap,testhistomap, k = 1, weights = 'distance')
            score_s1[trial,id_jit] = kNN1_score
            score_s12[trial,id_jit] = kNN12_score
            id_jit+=1
            
        id_jit = 0
        for j in jit_t:
            j = round(j,0)
            jitonic = [j,None]
            if j==0:
                jitonic = [None,None]
            testhistomap = hotshom.running(homeotest = homeotest, train=False, nb_digit=nb_test, jitonic=jitonic, outstyle = 'histo', subset_size = nb_test)
            
            kNN12_score = knn(trainhistomap,testhistomap, k = kNN, weights = 'distance')
            kNN1_score = knn(trainhistomap,testhistomap, k = 1, weights = 'distance')
            score_t1[trial,id_jit] = kNN1_score
            score_t12[trial,id_jit] = kNN12_score
            id_jit+=1
            
    with open(f_name, 'wb') as file:
        pickle.dump([score_t1, score_t12, jit_t, score_s1, score_s12, jit_s], file, pickle.HIGHEST_PROTOCOL)
            
            
    #trainhistomap = hotshom.running(homeotest=homeotest, nb_digit = nb_train, outstyle='histo')
    #JS_score = histoscore(trainhistomap,testhistomap, verbose = True)
    #trainhistomap = hotshom.running(homeotest=homeotest, nb_digit = nb_train, outstyle='histav')
    #JS_score = histoscore_lagorce(trainhistomap,testhistomap, verbose = True)
