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


timesteps = np.arange(0,290*1e3,100)
nb_classes = 10

results = [timesteps]

namelist = ['homhots', 'raw']
for namnum, name in enumerate(namelist):
    f_name = f'{record_path}{timestr}_LR_results_{name}_{nbclust}_{nb_train}_{nb_test}_{ds_ev}_timescale.pkl'
    print(f_name)
    with open(f_name, 'rb') as file:
        likelihood, true_target, time_scale = pickle.load(file)

    y_true = np.zeros([len(true_target)])
    y_score = np.zeros([len(true_target),nb_classes])
    proba_timestep = np.zeros([len(timesteps),len(true_target),nb_classes])
    i = 0
    accuracy = np.zeros([len(timesteps)])
    for likelihood_, true_target_, time_scale_ in zip(likelihood, true_target, time_scale):
        time_scale_ -= time_scale_[0]
        previous_ind = 0
        for idx, step in enumerate(timesteps):
            ind = np.where(time_scale_<step)[0][-1]
            proba = np.mean(likelihood_[previous_ind:ind,:], axis=0)
            proba_timestep[idx,i,:] = proba
            previous_ind = ind
            prediction = np.argmax(proba)
            if prediction==true_target_:
                accuracy[idx] += 1
        i+=1
    accuracy/=len(true_target)

    f_name = f'{record_path}{timestr}_LR_results_{name}_{nbclust}_{nb_train}_{nb_test}_{ds_ev}_timescale_acc.pkl'
    with open(f_name, 'wb') as file:
        pickle.dump([accuracy, proba_timestep, timesteps], file, pickle.HIGHEST_PROTOCOL)