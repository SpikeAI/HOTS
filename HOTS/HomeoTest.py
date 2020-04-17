__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS"

import time
import numpy as np
import pandas as pd

from HOTS.Tools import EuclidianNorm
from HOTS.KmeansCluster import Cluster
import itertools


class KmeansWithoutHomeo(Cluster):
    '''
    Clustering algorithm using homeostasis with histogram equalization
    INPUT :
        + nb_cluster : (<int>) number of cluster centers
        + to_record : (<boolean>) parameter to activate the monitoring of the learning
            'reach_each' steps
        + verbose : (<int>) control the verbosity
        + nb_quant : (<int>) controlling the number of quantization of histogram for the
            histogram equalization
        + C : (<int>) rescale parameter
        + NormType : (<string>) indicate the type of normalization we want to use, could
            be 'max' or 'standard'
        + eta : (<float>) learning paramter for the prototype updates
        + eta_homeo : (<float>) learning parameter for the histogram equalization
    '''

    def __init__(self, nb_cluster, to_record=False, verbose=0, nb_quant=100,
                 C=6, Norm_Type='max', eta=0.000005, eta_homeo=0.0005, l0_sparseness=5):
        Cluster.__init__(self, nb_cluster, to_record, verbose)
        self.nb_quant = nb_quant
        self.C = C
        self.Norm_Type = Norm_Type
        self.do_sym = False

        if eta is None:
            self.eta = 0.000005
        else:
            self.eta = eta

        if eta_homeo is None:
            self.eta_homeo = 0.0005
        else:
            self.eta_homeo = eta_homeo

        self.l0_sparseness = l0_sparseness
        self.verbose = verbose

    def fit(self, STS, batch_size=100, NbCycle=1, record_num_batches=10000):
        '''
        Methods to learn prototypes fitting data
        INPUT :
            + STS : (<STS object>) Stream of SpatioTemporal Surface to fit
            + init : (<string>) Method to initialize the prototype ('rdn' or None)
            + NbCycle : (<int>) Number of time the stream is going to be browse.
            + batch_size : (<int>) size of the batch used to update histogram equalization
                algortihm
            + record_num_batches : (<int>) number of data used at each step of monitoring
        OUTPUT :
            + prototype : (<np.array>) matrix of size (nb_cluster,nb_polarity*((2*R+1)*(2*R+1)))
                representing the centers of clusters
        '''

        X = STS.Surface
        if self.to_record == True:
            self.record_each = X.shape[0]//100
        n_samples, n_pixels = X.shape
        X_train = X.copy()
        norm = self.norm(X_train, Norm_Type=self.Norm_Type)
        X_train /= norm[:, np.newaxis]
        prototype = X_train[:self.nb_cluster, :].copy()
        self.P_cum = np.linspace(0, 1, self.nb_quant, endpoint=True)[
            np.newaxis, :] * np.ones((self.nb_cluster, 1))
        n_batches = n_samples // batch_size

        np.random.shuffle(X_train)
        batches = np.array_split(X_train, n_batches)
        import itertools
        # Return elements from list of batches until it is exhausted. Then repeat the sequence indefinitely.
        #
        batches = itertools.cycle(batches)
        n_iter = int(NbCycle * n_samples)
        for ii, this_X in zip(range(n_iter), batches):
            if this_X.ndim == 1:
                this_X = this_X[:, np.newaxis]

            n_samples, n_pixels = this_X.shape
            #n_dictionary, n_pixels = dictionary.shape
            sparse_code = np.zeros((n_samples, self.nb_cluster))

            if not self.P_cum is None:
                #nb_quant = P_cum.shape[1]
                stick = np.arange(self.nb_cluster)*self.nb_quant

            sparse_code = self.code(this_X, prototype)
            residual = this_X - sparse_code @ prototype
            residual /= self.nb_cluster
            prototype += self.eta * sparse_code.T @ residual

            norm = self.norm(prototype, Norm_Type=self.Norm_Type)

            prototype /= norm[:, np.newaxis]

            if self.verbose > 0 and ii % 5000 == 0:
                print('{0} / {1}'.format(ii, n_iter))

            if self.to_record == True:
                if ii % int(self.record_each) == 0:
                    from scipy.stats import kurtosis
                    indx = np.random.permutation(X_train.shape[0])[
                        :record_num_batches]
                    #polarity = self.code(X_train[indx, :],prototype,self.P_cum,sparse=True)
                    polarity = self.code(X_train[indx, :], prototype)
                    error = np.linalg.norm(
                        X_train[indx, :] - polarity @ prototype)/record_num_batches
                    active_probe = np.sum(polarity > 0, axis=0)
                    record_one = pd.DataFrame([{'error': error,
                                                'histo': active_probe,
                                                'var': np.var(active_probe)}],
                                              index=[ii])
                    self.record = pd.concat([self.record, record_one])

            #self.P_cum = self.update_Pcum(self.P_cum, sparse_code)
        self.prototype = prototype
        return prototype

    def norm(self, to_normalize, Norm_Type):
        if Norm_Type == 'standard':
            if to_normalize.ndim > 1:
                norm = np.sqrt(np.sum(to_normalize**2, axis=1))
            else:
                norm = np.sqrt(np.sum(to_normalize**2))
        elif Norm_Type == 'max':
            if to_normalize.ndim > 1:
                norm = np.amax(to_normalize, axis=1)
            else:
                norm = np.amax(to_normalize)

        return norm

    def update_Pcum(self, P_cum, code):
        '''
        Update the estimated modulation function in place.
        INPUT :
            + P_cum: (<np.array>) matrix of shape (n_samples, nb_quant) Value of the modulation
                function at the previous iteration.
            + code: (<np.array>) matrix of shape (n_samples, nbcluster) encoded data
        OUTPUT :
            + P_cum: (<np.array>) matrix of shape (n_samples, nb_quant) updated value
                of the modulation function.
        '''
        if self.eta_homeo > 0.:
            P_cum_ = self.get_Pcum(code)
            P_cum = (1 - self.eta_homeo)*P_cum + self.eta_homeo * P_cum_
        return P_cum

    def code(self, X, dictionary):
        '''
        Warning : This function this could be the same than the predict function in the Cluster mother class !
        code the data
        INPUT :
            + X : (<np.array>) matrix od size (nb_samples, nb_polarity*(2*R+1)*(2*R+1)) of spatiotemporal
                surface we want to code
            + dictionary : (<np.array>) matrix of size(nb_cluster, nb_polarity*(2*R+1)*(2*R+1)) of the
                prototype as they are at this step of the learning
            + P_cum: (<np.array>) matrix of shape (n_samples, nb_quant) Value of the modulation
                function at the previous iteration.
            + sparse : (<boolean>) parameter to define the format of the coding matrix. If sparse is True, the
                coding matrix is of size (nb_sample,nb_cluster) with zeros evrywhere and 1 on the matching prototype.
                If sparse is False, the coding matrix is of size (nb_sample), indicating the index of the
                matching prototype for each sample
        OUTPUT :
            + polarity : (<np.array>) of size depending of the sparse parameter, represeting a vector a index of closest
                prototype to each sample
        '''
        n_samples = X.shape[0]
        corr = (X @ dictionary.T)
        Xcorr = (dictionary @ dictionary.T)
        sparse_code = np.zeros((n_samples, self.nb_cluster))
        for i_sample in range(n_samples):
            c = corr[i_sample, :].copy()
            list_of_ind = list()
            list_of_c = list()
            for i_l0 in range(int(self.l0_sparseness)):
                ind = np.argmax(c)
                c_ind = c[ind] / Xcorr[ind, ind]
                sparse_code[i_sample, ind] += c_ind
                c -= c_ind * Xcorr[ind, :]
        return sparse_code

    def get_Pcum(self, code):
        '''
        calculate quantized histogram
        INPUT :
            + code: (<np.array>) matrix of shape (n_samples, nbcluster) encoded data
        OUTPUT :
            + P_cum: (<np.array>) matrix of shape (n_samples, nb_quant) value
                of the modulation function, corresponding to the quantized cumulative function
        '''
        n_samples = code.shape[0]
        P_cum = np.zeros((self.nb_cluster, self.nb_quant))
        for i in range(self.nb_cluster):
            p, bins = np.histogram(self.rescaling(code[:, i]), bins=np.linspace(
                0., 1, self.nb_quant, endpoint=True), density=True)
            p /= p.sum()
            P_cum[i, :] = np.hstack((0, np.cumsum(p)))
        return P_cum

    def rescaling(self, code):
        '''
        scale the code with decreasing exponential
        INPUT :
            + code: (<np.array>) matrix of shape (n_samples, nbcluster) representing encoded data
        OUTPUT :
            + (<np.array>) matrix of shape (n_samples, nbcluster), scaled version of the code matrix
        '''
        if self.do_sym:
            return 1.-np.exp(-np.abs(code)/self.C)
        else:
            return (1.-np.exp(-code/self.C))*(code > 0)

    def quantile(self, Pcum, p_c, stick):
        '''
        scale the code with decreasing exponential
        INPUT :
            + P_cum : (<np.array>) matrix of shape (n_samples, nb_quant) value
                of the modulation function, corresponding to the quantized cumulative function
            + p_c : (<np.array>) matrix of shape (n_samples, nbcluster), scaled version of the code matrix
            + stick : (<np.array>) matrix of shape (nb_quant), representing the index of quantization
        OUTPUT :
            + quantized value of the cumulative probability distribution
        '''
        return Pcum.ravel()[(p_c*Pcum.shape[1] - (p_c == 1)).astype(np.int) + stick]
