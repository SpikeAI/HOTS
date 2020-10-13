__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import numpy as np
from HOTS.STS import STS
from HOTS.KmeansLagorce import KmeansLagorce
from HOTS.KmeansMaro import KmeansMaro
from HOTS.KmeansCompare import KmeansCompare
from HOTS.KmeansHomeo import KmeansHomeo

class Layer(object):
    '''
    Layer is the mother class.

    INPUT :
        + verbose : (<int>) controls verbosity
        + type : (<str>) describes layer type

    '''

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.type = 'void'


class FilterNHBD(Layer):
    '''
    Filtering layer that keeps the event if the number of event in a neighbour
    of size [2*neighbourhood+1,2*neighbourhood+1] around the event's position
    is over the threshold value

    INPUT
        + threshold : (int), specify the minimum number of neighbour
        + neighbourhood : (int), specify the size of the neighbourhood to take into account
        + verbose : (<int>) control the verbosity

    TODO: is that function redundant with FilterRecent ?
    '''

    def __init__(self, threshold, neighbourhood, verbose=0):
        Layer.__init__(self, verbose)
        self.type = 'Filter'
        self.threshold = threshold
        self.neighbourhood = neighbourhood

    def RunLayer(self, event):
        '''

        OUTPUT
            + event : (event object) the filtered event
        '''
        self.input = event
        filt = np.zeros(self.input.address.shape[0]).astype(bool)
        idx_old = 0
        accumulated_image = np.zeros(
            (self.input.ImageSize[0]+2*self.neighbourhood, self.input.ImageSize[1]+2*self.neighbourhood))
        X_p, Y_p = np.meshgrid(np.arange(-self.neighbourhood, self.neighbourhood+1),
                               np.arange(-self.neighbourhood, self.neighbourhood+1), indexing='ij')
        for idx, (each_address, each_pola, each_time) in enumerate(zip(self.input.address, self.input.polarity, self.input.time)):
            # if self.input.event_nb[idx_old]>self.input.event_nb[idx] :
            if self.input.time[idx_old] > self.input.time[idx]:
                accumulated_image = np.zeros(
                    (self.input.ImageSize[0]+2*self.neighbourhood, self.input.ImageSize[1]+2*self.neighbourhood))
            x_translated = each_address[0] + self.neighbourhood
            y_translated = each_address[1] + self.neighbourhood

            accumulated_image[x_translated, y_translated] = 1
            nb_voisin = np.sum(
                accumulated_image[x_translated+X_p, y_translated+Y_p])
            if nb_voisin > self.threshold:
                filt[idx] = True
            idx_old = idx
        self.output = self.input.filter(filt)
        return self.output


class ClusteringLayer(Layer):
    '''
    Class of a layer associating SpatioTemporal surface from input event to a Cluster, and then outputing another event
    INPUT :
        + tau : (<int>), the time constant of the spatiotemporal surface
        + R : (<int>), the size of the neighbourhood taken into consideration in the time surface
        + ThrFilter :
        + LearningAlgo : (<string>)
        + kernel : (<string>)
        + eta : (<float>) learning rate.
        + eta_homeo : (<float>) learning rate for homeostasis. Set to zero to not use homeostasis.
        + sigma : (<float>) parameter of filtering in the isotropic filter. If None, there is no filter applied
        + verbose : (<int>) controls the verbosity
    '''

    def __init__(self, tau, R, ThrFilter=0, LearningAlgo='lagorce', kernel='exponential',
                 eta=None, homeo=False, eta_homeo=None, C=None, sigma=None, l0_sparseness=5, verbose=0):
        Layer.__init__(self, verbose)
        LearningAlgos = ['homeo', 'maro', 'lagorce', 'comp']
        self.type = 'Layer'
        self.homeo = homeo
        self.tau = tau
        self.R = R
        self.ThrFilter = ThrFilter
        self.LearningAlgo = LearningAlgo
        if self.LearningAlgo not in LearningAlgos:
            raise KeyError(
                'LearningAlgo should be in ' + str(LearningAlgos))
        self.kernel = kernel
        if self.kernel not in ['linear', 'exponential']:
            raise KeyError('[linear,exponential]')
        self.eta = eta
        self.eta_homeo = eta_homeo
        self.C = C
        self.sigma = sigma
        self.l0_sparseness = l0_sparseness
        if self.LearningAlgo == 'lagorce':
            self.ClusterLayer = KmeansLagorce(
                nb_cluster=0, homeo=self.homeo, verbose=self.verbose, to_record=False)
        elif self.LearningAlgo == 'maro':
            self.ClusterLayer = KmeansMaro(nb_cluster=0, homeo=self.homeo, verbose=self.verbose, to_record=False,
                                           eta=self.eta)
        elif self.LearningAlgo == 'homeo':
            self.ClusterLayer = KmeansLagorceHomeo(
                nb_cluster=0, verbose=self.verbose, to_record=False)
        elif self.LearningAlgo == 'comp':
            self.ClusterLayer = KmeansHomeo(nb_cluster=0, verbose=self.verbose,
                                            to_record=False,
                                            eta=self.eta, eta_homeo=None, C=self.C,
                                            l0_sparseness=self.l0_sparseness,  Norm_Type='standard')

    def RunLayer(self, event, Cluster):
        '''
        Associate each polarity of the event input to the prototype of the cluster
        INPUT :
            + event (<object Event>) : input event
            + Cluster (<object Cluster>) : Cluster, previously trained,
        OUTPUT :
            + self.output : (<object Event>) : Event with new polarities corresponding to the closest cluster center
        '''
        self.input = event
        self.SpTe_Layer = STS(tau=self.tau, R=self.R,
                              verbose=self.verbose, sigma=self.sigma)
        Surface_Layer = self.SpTe_Layer.create(event=self.input, kernel=self.kernel)

        # Check that THRFilter=0 is equivalent to no Filter
        event_filtered, _ = self.SpTe_Layer.FilterRecent(event=self.input, threshold=self.ThrFilter)

        self.output, _ = Cluster.predict(
            Surface=self.SpTe_Layer.Surface, event=event_filtered, R = self.R)

        return self.output

    def TrainLayer(self, event, nb_cluster, to_record=False, NbCycle=1):
        '''
        Train the ClusteringLayer

        INPUT :
            + event (<object event>) : input event
            + nb_cluster (<int>) : nb of centers
            + record_each (<int>) : record the convergence error each reach_each
            + NbCycle (<int>) : number of time we repeat the learning process. Need to be used when not enought training data to reach the convergence

        OUTPUT :
            + output (<object event>) : Event with new polarities corresponding to the closest cluster center
            + ClusterLayer (<object Cluster>) : Learnt cluster
        '''
        self.input = event
        self.SpTe_Layer = STS(tau=self.tau, R=self.R,
                              verbose=self.verbose, sigma=self.sigma)
        Surface_Layer = self.SpTe_Layer.create(
            event=self.input, kernel=self.kernel)

        event_filtered, _ = self.SpTe_Layer.FilterRecent(event=self.input, threshold=self.ThrFilter)

        self.ClusterLayer.nb_cluster, self.ClusterLayer.to_record = nb_cluster, to_record
        Prototype = self.ClusterLayer.fit(self.SpTe_Layer, NbCycle=NbCycle)
        self.output, _ = self.ClusterLayer.predict(
            Surface=self.SpTe_Layer.Surface, event=event_filtered, R = self.R)

        return self.output, self.ClusterLayer
