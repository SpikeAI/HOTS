__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import numpy as np
import pickle

def prediction(to_predict, prototype, homeo, R):
    '''
    function to predict polarities
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
    valmin = 0.7
    valmax = 1.3*2
    N = prototype.shape[0]
    b = np.log(valmax)*np.log(valmin)/((1-N)*np.log(valmin)-np.log(valmax))
    d = -b/np.log(valmin)
    a = -b*N

    if homeo==True:
        nb_proto = np.zeros(prototype.shape[0])
        idx_global = 0
    polarity, output_distance = np.zeros(
        to_predict.shape[0]), np.zeros(to_predict.shape[0])
    for idx in range(to_predict.shape[0]):
        Euclidian_distance = np.sqrt(
            np.sum((to_predict[idx] - prototype)**2, axis=1))
        if homeo==True and idx_global>0:
            #gain = np.exp((a*nb_proto/max(idx_global,1)+b)/(nb_proto/max(idx_global,1)-d))
            gain = np.exp(R*(nb_proto/max(idx_global,1)-1/prototype.shape[0]))
            #gain = np.log(1/prototype.shape[0])/np.log(nb_proto/idx_global)
            #print('predict', gain, Euclidian_distance)
            polarity[idx] = np.argmin(Euclidian_distance*gain)
            output_distance[idx] = Euclidian_distance[int(polarity[idx])]
            nb_proto[int(polarity[idx])] += 1
            idx_global += 1
        else:
            polarity[idx] = np.argmin(Euclidian_distance)
            output_distance[idx] = np.amin(Euclidian_distance)
    return output_distance, polarity.astype(int)

def predictioncosine(to_predict, prototype, homeo, R):
    '''
    function to predict polarities
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

    if homeo==True:
        nb_proto = np.ones(prototype.shape[0])
        idx_global = prototype.shape[0]
    polarity, output_distance = np.zeros(
        to_predict.shape[0]), np.zeros(to_predict.shape[0])
    for idx in range(to_predict.shape[0]):
        Euclidian_distance = np.dot(prototype, to_predict[idx])/(np.sqrt(np.sum(to_predict**2))*np.sqrt(np.sum(prototype**2, axis=1)))
        if homeo==True:
            #gain = np.exp((a*nb_proto/max(idx_global,1)+b)/(nb_proto/max(idx_global,1)-d))
            #gain = np.exp(R*(nb_proto/max(idx_global,1)-1/prototype.shape[0]))
            gain = np.log(nb_proto/idx_global)/np.log(1/prototype.shape[0])
            #print('predict', gain, Euclidian_distance)
            polarity[idx] = np.argmax(Euclidian_distance*gain)
            output_distance[idx] = Euclidian_distance[int(polarity[idx])]
            nb_proto[int(polarity[idx])] += 1
            idx_global += 1
        else:
            polarity[idx] = np.argmax(Euclidian_distance)
            output_distance[idx] = np.amax(Euclidian_distance)
    return output_distance, polarity.astype(int)

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
