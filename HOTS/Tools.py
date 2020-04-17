__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS"

import numpy as np
import pickle
from numba import jit


@jit(nopython=True)
def jitted_prediction(to_predict, prototype):
    '''
    jitted fonction to fast predict polarities
    INPUT :
        + to_predict : (<np.array>) array of size (nb_of_event,nb_polarity*(2*R+1)*(2*R+1)) representing the
            spatiotemporal surface to cluster
        + prototype : (<np.array>)  array of size (nb_cluster,nb_polarity*(2*R+1)*(2*R+1)) representing the
            learnt prototype
    OUTPUT :
        + output_distance : (<np.array>) vector representing the euclidian distance from each surface to the closest
            prototype
        + polarity : (<np.array>) vector representing the polarity of the closest prototype (argmin)
    '''

    polarity, output_distance = np.zeros(
        to_predict.shape[0]), np.zeros(to_predict.shape[0])
    for idx in range(to_predict.shape[0]):
        Euclidian_distance = np.sqrt(
            np.sum((to_predict[idx] - prototype)**2, axis=1))
        polarity[idx] = np.argmin(Euclidian_distance)
        output_distance[idx] = np.amin(Euclidian_distance)
    return output_distance, polarity


def Norm(Hist, Histo_proto, method):
    '''
    One function to pack all the norm
    INPUT :
        + Hist : (<np.array>) matrix of size (nb_sample,nb_polarity) representing the histogram for each sample
        + Histo_proto : (<np.array>) matrix of size (nb_cluster,nb_polarity) representing the histogram for each
            prototype
    OUTPUT :
        + to_return : (<np.array>)  of size (nb_sample,nb_Cluster) representing the euclidian distance from the samples histogram
            to the prototype histo
    '''
    if method == 'euclidian':
        to_return = EuclidianNorm(Hist, Histo_proto)
    elif method == 'normalized':
        to_return = NormalizedNorm(Hist, Histo_proto)
    elif method == 'battacha':
        to_return = BattachaNorm(Hist, Histo_proto)
    return to_return


def EuclidianNorm(Hist, Histo_proto):
    return np.sqrt(np.sum((Hist - Histo_proto)**2, axis=1))


def NormalizedNorm(Hist, Histo_proto):
    summation = np.sum(Histo_proto, axis=1)
    return np.sqrt(np.sum((Hist/np.sum(Hist) - Histo_proto/summation[:, None])**2, axis=1))


def BattachaNorm(Hist, Histo_proto):
    summation = np.sum(Histo_proto, axis=1)
    return -np.log(np.sum(np.sqrt(np.multiply(Histo_proto/summation[:, None], Hist/np.sum(Hist))), axis=1))


def SaveObject(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def LoadObject(filename):
    with open(filename, 'rb') as file:
        Clust = pickle.load(file)
    return Clust
# def Load(filename):


def GenerateHistogram(event):
    '''
    Generate an histogram for each sample.
    INPUT :
        + event (<object event>) stream on event on which we want to create an histogram
    OUTPUT :
        + freq = (<np.array>) of size (nb_samples,nb_clusters) representing the histrogram of cluster
            activation for each sample
        + pola = (<np.array>) of size (nb_sample,nb_clusters) representing the index of cluster activation
    '''
    last_change = 0
    for idx, each_change in enumerate(event.ChangeIdx):
        freq, pola = np.histogram(
            event.polarity[last_change:each_change+1], bins=len(event.ListPolarities))
        if idx != 0:
            freq_mat = np.vstack((freq_mat, freq))
        else:
            freq_mat = freq
        last_change = each_change
    return freq_mat, pola
