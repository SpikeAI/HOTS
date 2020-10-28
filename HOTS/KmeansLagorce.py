__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import time
import numpy as np
from HOTS.Tools import EuclidianNorm, prediction
from HOTS.KmeansCluster import Cluster

class KmeansLagorce(Cluster):
    '''
    Clustering algorithm as defined in the HOTS paper (Lagorce et al 2017)
    INPUT :
        + nb_cluster : (<int>) number of centriods to cluster
        + to_record : (<boolean>) parameter to activate the monitoring of the learning
        + verbose : (<int>) control the verbosity
    '''

    def __init__(self, nb_cluster, homeo, to_record=True, verbose=0):
        Cluster.__init__(self, nb_cluster, homeo, to_record, verbose)
        self.homeo = homeo

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
        
        valmin = 0.7
        valmax = 1.3*2
        N = self.nb_cluster
        b = np.log(valmax)*np.log(valmin)/((1-N)*np.log(valmin)-np.log(valmax))
        d = -b/np.log(valmin)
        a = -b*N

        surface = STS.Surface.copy()
        if self.to_record == True:
            self.record_each = surface.shape[0]//1000
        if init is None:
            self.prototype = surface[:self.nb_cluster, :]
        elif init == 'rdn':
            idx = np.random.permutation(np.arange(surface.shape[0]))[
                :self.nb_cluster]
            self.prototype = surface[idx, :]
        else:
            raise NameError('argument '+str(init) +
                            ' is not valid. Only None or rdn are valid')
        self.idx_global = 0
        nb_proto = np.zeros((self.nb_cluster))
        for each_cycle in range(NbCycle):
            for idx, Si in enumerate(surface):
                # find the closest prototype
                Distance_to_proto = np.linalg.norm(Si - self.prototype, ord=2, axis=1)
                #Adding homeostasis rule
                if self.homeo==True:
                    #gain = np.exp((a*nb_proto/max(self.idx_global,1)+b)/(nb_proto/max(self.idx_global,1)-d))
                    gain = np.exp(STS.R*(nb_proto/max(self.idx_global,1)-1/self.nb_cluster))
                    #gain = np.log(1/self.nb_cluster)/np.log(nb_proto/self.idx_global)
                    #print('fit', gain, Distance_to_proto)
                    closest_proto_idx = np.argmin(Distance_to_proto*gain)
                else:
                    closest_proto_idx = np.argmin(Distance_to_proto)
                pk = nb_proto[closest_proto_idx]
                Ck = self.prototype[closest_proto_idx, :]
                alpha = 0.01/(1+pk/20000)
                beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))
                                       * np.sqrt(np.dot(Ck, Ck)))
                Ck_t = Ck + alpha*(Si - beta*Ck)
                #Ck_t = Ck + alpha*beta*(Si - Ck)

                # Updating the number of selection
                nb_proto[closest_proto_idx] += 1
                self.prototype[closest_proto_idx, :] = Ck_t

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

    def fitcosine(self, STS, init=None, NbCycle=1):
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
            idx = np.random.permutation(np.arange(surface.shape[0]))[
                :self.nb_cluster]
            self.prototype = surface[idx, :]
        else:
            raise NameError('argument '+str(init) +
                            ' is not valid. Only None or rdn are valid')
        self.idx_global = self.nb_cluster
        nb_proto = np.ones((self.nb_cluster))
        for each_cycle in range(NbCycle):
            for idx, Si in enumerate(surface):
                # find the closest prototype
                Distance_to_proto = np.dot(self.prototype,Si)/(np.sqrt(np.sum(Si**2))*np.sqrt(np.sum(self.prototype**2, axis=1)))
                #Adding homeostasis rule
                if self.homeo==True:
                    gain = np.log(nb_proto/self.idx_global)/np.log(1/self.nb_cluster)
                    #print('fit', gain, Distance_to_proto)
                    closest_proto_idx = np.argmax(Distance_to_proto*gain)
                else:
                    closest_proto_idx = np.argmax(Distance_to_proto)
                pk = nb_proto[closest_proto_idx]
                Ck = self.prototype[closest_proto_idx, :]
                alpha = 0.01/(1+pk/20000)
                beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))
                                       * np.sqrt(np.dot(Ck, Ck)))
                Ck_t = Ck + alpha*(Si - beta*Ck)
                #Ck_t = Ck + alpha*beta*(Si - Ck)

                # Updating the number of selection
                nb_proto[closest_proto_idx] += 1
                self.prototype[closest_proto_idx, :] = Ck_t

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