__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import time
import numpy as np
from KmeansCluster import Cluster

class KmeansMaro(Cluster):
    '''
    Clustering algorithm as defined in the second HOTS paper (Maro et al 2017)
    INPUT :
        + nb_cluster : (<int>) number of cluster centers
        + to_record : (<boolean>) parameter to activate the monitoring of the learning
        + verbose : (<int>) control the verbosity
        + eta : (<float>) could be use to define a learning rate
    '''

    def __init__(self, nb_cluster, homeo, to_record=True, verbose=0, eta=None):
        Cluster.__init__(self, nb_cluster, to_record, verbose)
        self.homeo = homeo
        if eta is None:
            self.eta = 1.
        else:
            self.eta = eta

    def fit(self, STS, init=None, NbCycle=1):
        '''
        Methods to learn prototypes fitting data
        INPUT :
            + STS : (<STS object>) Stream of SpatioTemporal Surface to fit
            + init : (<string>) Method to initialize the prototype ('rdn' or None)
            + NbCycle : (<int>) Number of time the stream is going to be browse.
        OUTPUT :
            + prototype : (<np.array>) matrix of size (nb_cluster,nb_polarity*((2*R+1)*(2*R+1)))
                representing the centers of clusters
        '''
        tic = time.time()

        surface = STS.Surface.copy()
        if self.to_record == True:
            self.record_each = surface.shape[0]//1000
        if init is None:
            self.prototype = surface[:self.nb_cluster, :]
        elif init == 'rdn':
            idx = np.random.permutation(np.arange(surface.shape[0]))[:self.nb_cluster]
            self.prototype = surface[idx, :]
        else:
            self.prototype = init

        last_time_activated = np.zeros((self.nb_cluster)).astype(int)
        self.idx_global = 0
        for each_cycle in range(NbCycle):
            nb_proto = np.zeros((self.nb_cluster))
            for idx, Si in enumerate(surface):
                # find the closest prototype
                #Distance_to_proto = EuclidianNorm(Si, self.prototype)
                Distance_to_proto = np.linalg.norm(
                    Si - self.prototype, ord=2, axis=1)
                if self.homeo==True:
                    gain = np.exp(STS.R*(nb_proto/max(self.idx_global,1)-1/self.nb_cluster))
                    closest_proto_idx = np.argmin(Distance_to_proto*gain)
                else:
                    closest_proto_idx = np.argmin(Distance_to_proto)
                Ck = self.prototype[closest_proto_idx, :]
                last_time_activated[closest_proto_idx] = idx
                # Updating the prototype
                pk = nb_proto[closest_proto_idx]
                alpha = 1/(1+pk)
                beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))
                                       * np.sqrt(np.dot(Ck, Ck)))
                Ck_t = Ck + self.eta*alpha*beta*(Si-Ck)

                # Updating the number of selection
                nb_proto[closest_proto_idx] += 1
                self.prototype[closest_proto_idx, :] = Ck_t

                critere = (idx-last_time_activated) > 10000
                critere2 = nb_proto < 25000
                if np.any(critere2*critere):
                    cri = nb_proto[critere] < 25000
                    idx_critere = np.arange(0, self.nb_cluster)[critere][cri]
                    for idx_c in idx_critere:
                        Ck = self.prototype[idx_c, :]
                        beta = np.dot(
                            Ck, Si)/(np.sqrt(np.dot(Si, Si))*np.sqrt(np.dot(Ck, Ck)))
                        Ck_t = Ck + 0.2*beta*(Si-Ck)
                        self.prototype[idx_c, :] = Ck_t

                if self.to_record == True:
                    if self.idx_global % int(self.record_each) == 0:
                        self.monitor(surface, self.idx_global,
                                     SurfaceFilter=1000, R = STS.R)
                self.idx_global += 1

        tac = time.time()

        self.nb_proto = nb_proto
        if self.verbose > 0:
            print(
                'Clustering SpatioTemporal Surface in ------ {0:.2f} s'.format(tac-tic))

        return self.prototype
