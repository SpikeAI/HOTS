__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import time
import numpy as np
import pandas as pd
import HOTS.Tools as Tools

class Cluster(object):
    '''
    Cluster is the master class gathering all the clustering algorithms.
    INPUT :
        + nb_cluster : (<int>) number of cluster centers
        + record_each : (<int>) used to monitor the learning, it records errors and histogram
        + verbose : (<int>) control the verbosity
    '''

    def __init__(self, nb_cluster, homeo=False, to_record=False, verbose=0):
        self.nb_cluster = nb_cluster
        self.verbose = verbose
        self.homeo = homeo
        self.prototype = np.zeros(0)
        self.to_record = to_record
        self.record = pd.DataFrame([], columns=['error', 'histo'])
        self.idx_global = 0

    def predict(self, Surface, R, event=None, SurfaceFilter=None):
        '''
        Methods to predict the closest prototype from a stream a STS
        INPUT :
            + Surface : (<np.array>) array of size (nb_of_event,nb_polarity*(2*R+1)*(2*R+1))
                representing the spatiotemporal surface to cluster.
            + event : (<event object>) event associated to the STS. return another
                event stream with new polarity
        OUTPUT :
            + output_distance : (<np.array>)
            + either: event_output : (<event.object>)
            + or : polarity : (<np.array>)
        '''

        if self.prototype is None:
            raise ValueError('Train the Cluster before doing prediction')

        output_distance, polarity = Tools.prediction(Surface, self.prototype, self.homeo, R)

        if event is not None:
            event_output = event.copy()
            event_output.polarity = polarity
            event_output.ListPolarities = list(np.arange(self.nb_cluster))
            return event_output, output_distance
        else:
            return polarity, output_distance

    def monitor(self, Surface, idx_global, SurfaceFilter, R):
        '''
        Methods to record error and activation histogram during the training
        INPUT :
            + STS : (<STS object>) Stream of SpatioTemporal Surface to fit
            + idx_global (<int>) number of iteration where this methods is called
            + SurfaceFilter : (<int>) To predict only on a small subsample of Surface of size (SurfaceFilter)
        '''
        if SurfaceFilter == None:
            to_predict = Surface
        else:
            random_selection = np.random.permutation(
                np.arange(Surface.shape[0]))[:SurfaceFilter]
            to_predict = Surface[random_selection]

        pol, output_distance, = self.predict(to_predict, R)
        error = np.mean(output_distance)
        active_probe = np.histogram(pol, bins=np.arange(self.nb_cluster+1))[0]

        self.record.loc[idx_global] = {'error': error,
                                     'histo': active_probe}
