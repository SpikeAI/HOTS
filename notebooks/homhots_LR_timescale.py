from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import pickle
import numpy as np

record_path = '../Records/EXP_03_NMNIST/'
timestr = '2021-03-29'

nbclust = [4, 8, 16]
nb_train = 60000
nb_test = 10000
ds_ev = 1


timesteps = np.arange(500,100000,100)

results = [timesteps]

namelist = ['raw', 'hot', 'homhots']
for namnum, name in enumerate(namelist):
    f_name = f'{record_path}{timestr}_LR_results_{name}_{nbclust}_{nb_train}_{nb_test}_{ds_ev}_timescale.pkl'
    with open(f_name, 'rb') as file:
        likelihood, true_target, time_scale = pickle.load(file)
    nb_classes = 4
    y_true = np.zeros([len(true_target)])
    y_score = np.zeros([len(true_target),nb_classes])
    proba_timestep = np.zeros([len(timesteps),len(true_target),nb_classes])
    i = 0
    for likelihood_, true_target_, time_scale_ in zip(likelihood, true_target, time_scale):
        time_scale_ -= time_scale_[0]
        previous_ind = 0
        for idx, step in enumerate(timesteps):
            ind = np.where(time_scale_<step)[0][-1]
            proba = np.mean(likelihood_[previous_ind:ind,:], axis=0)
            if np.isnan(proba[0]):
                print(step)
            proba_timestep[idx,i,:] = proba
        i+=1
    AUC = np.zeros([len(timesteps)])
    for idx, step in enumerate(timesteps):
        #print(proba_timestep[idx,:,:].shape, np.array(true_timestep[idx]))
        AUC[idx] = roc_auc_score(LabelBinarizer().fit_transform(np.array(true_target)),proba_timestep[idx,:,:], multi_class='ovr')
    results.append(AUC)

f_name = f'{path}{timestr}_LR_results_{nbclust}_{nb_train}_{nb_test}_{ds_ev}_AUC.pkl'

with open(f_name, 'wb') as file:
    pickle.dump([results], file, pickle.HIGHEST_PROTOCOL)