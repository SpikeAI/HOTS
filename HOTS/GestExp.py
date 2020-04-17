import HOTS.libUnpackAtis as ua
import HOTS.libDataHelper as dh
from HOTS.STS import STS
import numpy as np
from HOTS.Event import LoadGestureDB
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier


class GestExp(object):
    def __init__(self, SettingsFile, nb_dico, R, tau, OutOnePolarity=True,
                 verbose=0):

        self.verbose = verbose
        self.specs = dh.superLoadHOTSNetworkSettingsFromFile(
            SettingsFile, _verbose=self.verbose)
        self.dbspecs = dh.superReadDB(self.specs.db_file, self.specs.db_path)
        self.nb_dico = nb_dico
        self.R = R
        self.tau = tau
        self.OutOnePolarity = OutOnePolarity
        if self.OutOnePolarity == True:
            self.nb_polarities = 1
        else:
            self.nb_polaritues = 2

    def learnMaro(self, nb_file=None):
        SpTe_Layer1 = STS(tau=self.tau, R=self.R, initial_time=1)
        self.res_list = list()
        #eta = 0.1
        if nb_file is None:
            nb_file = np.sum(np.array(self.dbspecs.dblabels) == 1)
        if self.verbose != 0:
            print('Training on {0} files'.format(nb_file))
        self.Prototype = np.random.rand(
            self.nb_dico, self.nb_polarities * SpTe_Layer1.RF_diam*SpTe_Layer1.RF_diam)
        self.total_activation = np.zeros(self.nb_dico)
        idx_train = 1
        for idf, dblabel in enumerate(self.dbspecs.dblabels):
            if dblabel == 1:

                if (self.verbose != 0) and (idx_train % 10 == 0):
                    print('learned file {0}/{1}'.format(idx_train, nb_file))

                filepath = self.dbspecs.path + self.dbspecs.filenames[idf]
                event = LoadGestureDB(
                    filepath, OutOnePolarity=self.OutOnePolarity)
                Surface_Layer1 = SpTe_Layer1.create(
                    event=event, kernel='linear')

                filt = np.sum(Surface_Layer1, axis=1) > 2 * self.R
                Surface_Layer2 = Surface_Layer1[filt, :]
                res = np.zeros((Surface_Layer2.shape[0]))
                for idx, Si in enumerate(Surface_Layer2):
                    Distance_to_proto = np.linalg.norm(
                        Si - self.Prototype, ord=2, axis=1)
                    closest_proto_idx = np.argmin(Distance_to_proto)
                    Ck = self.Prototype[closest_proto_idx, :].copy()
                    pk = self.total_activation[closest_proto_idx]
                    alpha = 1/(1+pk)
                    beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))
                                           * np.sqrt(np.dot(Ck, Ck)))
                    Ck_t = Ck + alpha*beta*(Si-Ck)
                    res[idx] = np.linalg.norm(Si-Ck_t, ord=2)
                    self.total_activation[closest_proto_idx] += 1
                    self.Prototype[closest_proto_idx, :] = Ck_t

                self.res_list.append(np.mean(res))
                if idx_train == nb_file:
                    break
                idx_train += 1
        return self.Prototype

    def GenerateHistogramMaro(self, train=True, nb_file=None):
        SpTe_Layer1 = STS(tau=self.tau, R=self.R, initial_time=1)
        if train == True and nb_file == None:
            nb_file = np.sum(np.array(self.dbspecs.dblabels) == 1)
        if train == False and nb_file == None:
            nb_file = np.sum(np.array(self.dbspecs.dblabels) == 2)
        if train == True:
            mode = 1
            if self.verbose != 0:
                print('coding on training db')
        else:
            if self.verbose != 0:
                print('coding on testing db')
            mode = 2

        all_histo = np.zeros((nb_file, self.nb_dico))
        labels = np.zeros(nb_file)
        idx_train = 1
        for idf, dblabel in enumerate(self.dbspecs.dblabels):
            if dblabel == mode:
                if (self.verbose != 0) and (idx_train % 10 == 0):
                    print('coded file {0}/{1}'.format(idx_train, nb_file))

                histo = np.zeros(self.nb_dico).astype(int)

                filepath = self.dbspecs.path + self.dbspecs.filenames[idf]
                event = LoadGestureDB(
                    filepath, OutOnePolarity=self.OutOnePolarity)
                Surface_Layer1 = SpTe_Layer1.create(
                    event=event, kernel='linear')
                filt = np.sum(Surface_Layer1, axis=1) > 2*self.R
                Surface_Layer2 = Surface_Layer1[filt, :]

                for idx, Si in enumerate(Surface_Layer2):
                    Distance_to_proto = np.linalg.norm(
                        Si - self.Prototype, ord=2, axis=1)
                    closest_proto_idx = np.argmin(Distance_to_proto)
                    histo[closest_proto_idx] += 1

                all_histo[idx_train-1, :] = histo/np.sum(histo)
                labels[idx_train-1] = self.dbspecs.labelids[idf]
                if idx_train == nb_file:
                    break
                idx_train += 1
        if train == True:
            self.training_data = (all_histo, labels)
        else:
            self.testing_data = (all_histo, labels)
        return (all_histo, labels)

    def learnHomeo(self, nb_file=None, eta=0.01, eta_homeo=0.01):
        SpTe_Layer1 = STS(tau=self.tau, R=self.R, initial_time=1)
        self.res_list = list()
        if nb_file is None:
            nb_file = np.sum(np.array(self.dbspecs.dblabels) == 1)
        if self.verbose != 0:
            print('Training on {0} files'.format(nb_file))
        self.Prototype = np.random.rand(
            self.nb_dico, self.nb_polarities * SpTe_Layer1.RF_diam*SpTe_Layer1.RF_diam)
        self.Prototype /= np.linalg.norm(self.Prototype,
                                         ord=2, axis=1)[:, None]
        Modulation = np.ones(self.nb_dico)
        idx_train = 1
        self.total_activation = np.zeros(self.nb_dico)
        for idf, dblabel in enumerate(self.dbspecs.dblabels):
            if dblabel == 1:

                if (self.verbose != 0) and (idx_train % 10 == 0):
                    print('learned file {0}/{1}'.format(idx_train, nb_file))

                filepath = self.dbspecs.path + self.dbspecs.filenames[idf]
                event = LoadGestureDB(
                    filepath, OutOnePolarity=self.OutOnePolarity)
                Surface_Layer1 = SpTe_Layer1.create(
                    event=event, kernel='linear')
                filt = np.sum(Surface_Layer1, axis=1) > 2*self.R
                Surface_Layer2 = Surface_Layer1[filt, :]
                Surface_Layer2 /= np.linalg.norm(Surface_Layer2,
                                                 ord=2, axis=1)[:, None]

                n_batches = Surface_Layer2.shape[0] // 500

                batched_data = np.array_split(Surface_Layer2, n_batches)
                nb_proto = np.zeros(self.nb_dico)

                for each_batches in batched_data:

                    corr = Modulation*(each_batches @ self.Prototype.T)

                    for idx_sample, Si in enumerate(each_batches):
                        res = np.zeros(each_batches.shape[0])
                        c = corr[idx_sample, :].copy()
                        ind = np.argmax(c)
                        nb_proto[ind] += 1
                        Ck = self.Prototype[ind, :]
                        beta = np.dot(
                            Ck, Si)/(np.sqrt(np.dot(Si, Si))*np.sqrt(np.dot(Ck, Ck)))
                        self.Prototype[ind, :] = Ck + beta*eta*(Si - Ck)
                        res[idx_sample] = np.linalg.norm(
                            Si-self.Prototype[ind, :], ord=2)

                    self.Prototype /= np.linalg.norm(
                        self.Prototype, ord=2, axis=1)[:, None]

                    self.res_list.append(np.mean(res))

                    self.total_activation += nb_proto
                    target = np.mean(self.total_activation)
                    tau = - (np.max(self.total_activation)-target)/np.log(0.2)
                    if eta_homeo is not None:
                        Modulation = np.exp(
                            (1-eta_homeo)*np.log(Modulation) - eta_homeo*((self.total_activation-target)/tau))

                if idx_train == nb_file:
                    break
                idx_train += 1
        return self.Prototype

    def learnHomeo2(self, nb_file=None, eta_homeo=0.01):
        SpTe_Layer1 = STS(tau=self.tau, R=self.R, initial_time=1)
        self.res_list = list()
        #eta = 0.1
        if nb_file is None:
            nb_file = np.sum(np.array(self.dbspecs.dblabels) == 1)
        if self.verbose != 0:
            print('Training on {0} files'.format(nb_file))
        self.Prototype = np.random.rand(
            self.nb_dico, self.nb_polarities * SpTe_Layer1.RF_diam*SpTe_Layer1.RF_diam)
        Modulation = np.ones(self.nb_dico)
        self.total_activation = np.zeros(self.nb_dico)
        idx_train = 1
        for idf, dblabel in enumerate(self.dbspecs.dblabels):
            if dblabel == 1:

                if (self.verbose != 0) and (idx_train % 10 == 0):
                    print('learned file {0}/{1}'.format(idx_train, nb_file))

                filepath = self.dbspecs.path + self.dbspecs.filenames[idf]
                event = LoadGestureDB(
                    filepath, OutOnePolarity=self.OutOnePolarity)
                Surface_Layer1 = SpTe_Layer1.create(
                    event=event, kernel='linear')

                filt = np.sum(Surface_Layer1, axis=1) > 2 * self.R
                Surface_Layer2 = Surface_Layer1[filt, :]
                res = np.zeros((Surface_Layer2.shape[0]))
                #nb_proto = np.zeros(self.nb_dico)
                for idx, Si in enumerate(Surface_Layer2):
                    Distance_to_proto = np.linalg.norm(
                        Modulation[:, None]*(Si - self.Prototype), ord=2, axis=1)
                    closest_proto_idx = np.argmin(Distance_to_proto)
                    Ck = self.Prototype[closest_proto_idx, :].copy()
                    pk = self.total_activation[closest_proto_idx]
                    alpha = 1/(1+pk)
                    beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))
                                           * np.sqrt(np.dot(Ck, Ck)))
                    Ck_t = Ck + alpha*beta*(Si-Ck)
                    res[idx] = np.linalg.norm(Si-Ck_t, ord=2)
                    self.total_activation[closest_proto_idx] += 1
                    self.Prototype[closest_proto_idx, :] = Ck_t
                    #nb_proto[closest_proto_idx] += 1
                to_print = (self.total_activation /
                            np.sum(self.total_activation))*100
                #print('proba activation {0}'.format(to_print))
                #print('Modulation {0}'.format(Modulation))
                target = np.mean(self.total_activation)
                tau = - (np.max(self.total_activation)-target)/np.log(0.2)
                if eta_homeo is not None:
                    Modulation = np.exp(
                        (1-eta_homeo)*np.log(Modulation) - eta_homeo*((self.total_activation-target)/tau))

                self.res_list.append(np.mean(res))
                if idx_train == nb_file:
                    break
                idx_train += 1
        return self.Prototype

    def GenerateHistogramHomeo(self, train=True, nb_file=None):
        SpTe_Layer1 = STS(tau=self.tau, R=self.R, initial_time=1)
        if train == True and nb_file == None:
            nb_file = np.sum(np.array(self.dbspecs.dblabels) == 1)
        if train == False and nb_file == None:
            nb_file = np.sum(np.array(self.dbspecs.dblabels) == 2)
        if train == True:
            mode = 1
            if self.verbose != 0:
                print('coding on training db')
        else:
            if self.verbose != 0:
                print('coding on testing db')
            mode = 2

        #nb_dico = dico.shape[0]

        all_histo = np.zeros((nb_file, self.nb_dico))
        labels = np.zeros(nb_file)
        idx_train = 1
        for idf, dblabel in enumerate(self.dbspecs.dblabels):
            if dblabel == mode:
                if (self.verbose != 0) and (idx_train % 10 == 0):
                    print('coded file {0}/{1}'.format(idx_train, nb_file))

                histo = np.zeros(self.nb_dico).astype(int)

                filepath = self.dbspecs.path + self.dbspecs.filenames[idf]
                event = LoadGestureDB(
                    filepath, OutOnePolarity=self.OutOnePolarity)
                Surface_Layer1 = SpTe_Layer1.create(
                    event=event, kernel='linear')
                filt = np.sum(Surface_Layer1, axis=1) > 2*self.R
                Surface_Layer2 = Surface_Layer1[filt, :]
                Surface_Layer2 /= np.linalg.norm(Surface_Layer2,
                                                 ord=2, axis=1)[:, None]
                corr = Surface_Layer2 @ self.Prototype.T
                for idx, Si in enumerate(Surface_Layer2):
                    c = corr[idx, :].copy()
                    closest_proto_idx = np.argmax(c)
                    #Distance_to_proto = np.linalg.norm(Si - Prototype1,ord=2,axis=1)
                    #closest_proto_idx = np.argmin(Distance_to_proto)
                    histo[closest_proto_idx] += 1

                all_histo[idx_train-1, :] = histo/np.sum(histo)
                labels[idx_train-1] = self.dbspecs.labelids[idf]
                if idx_train == nb_file:
                    break
                idx_train += 1
        if train == True:
            self.training_data = (all_histo, labels)
        else:
            self.testing_data = (all_histo, labels)
        return (all_histo, labels)


def save_object(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
    print('file saved')


def load_object(filename):
    with open(filename, 'rb') as file:
        Exp = pickle.load(file)
    return Exp


def MonitorHisto(histo):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    nb_dico = histo.shape[0]
    to_plot = ax.bar(np.arange(nb_dico), histo,
                     width=np.diff(np.arange(nb_dico+1)), ec="k", align="edge")


def Classify(training_data, testing_data, n_neighbors=3):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    (histo_train, label_train) = training_data
    (histo_test, label_test) = testing_data
    neigh.fit(histo_train, label_train)
    return neigh.score(histo_test, label_test)
