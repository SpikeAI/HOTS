__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import numpy as np

from HOTS.Tools import Norm
from HOTS.Tools import GenerateHistogram


class Classifier(object):
    '''
    class made to classify events:
    INPUT :
        + event_train : (<object event>) the events used to generate prototype histogram
        + event_test : (<object event>) the events we want to classify
        + TrainingLabel : (<np.array>) vector of string of size (nb_of_training_sample,1) describing the training set.
            It has to be in the same order than event_train
        + GroundTruth : (<np.array>) vector of string of size (nb_of_testing_sample,1) describing the testing set.
            It has to be in the same order than event_test. It will be used as a groundthruth to calculate accuracy
    '''

    def __init__(self, event_train, event_test, TrainingLabel, GroundTruth):
        self.event_train = event_train
        self.event_test = event_test
        self.TrainingLabel = TrainingLabel
        self.GroundTruth = GroundTruth

    def Accuracy(self, list_of_classified):
        '''
        method to classify using histogram distance between prototypes and :
        INPUT :
            + list_of_classified : (<list>) of np.array of size (nb_of_methods) representing the prediction made by each method
        OUTPUT :
            + list_of_accuracy : (<list>) of float of size (nb_of_methods) representing the accuracy on the testing_sample
        '''
        if type(list_of_classified) is not list:
            list_of_classified = [list_of_classified]

        nb_sample = list_of_classified[0].shape[0]
        list_of_accuracy = list()
        for idx, each_norm in enumerate(list_of_classified):
            list_of_accuracy.append((np.sum(list_of_classified[idx] == self.GroundTruth)/nb_sample))

        return list_of_accuracy

    # (histo_to_classify,proto_histo,proto_label_list=None):
    def HistogramDistance(self, methods=['euclidian', 'normalized', 'battacha'], to_print=False):
        '''
        method to classify using histogram distance between prototypes and :
        INPUT :
            + methods : (<list>) of (<string>) to inform which norm to use to calculate histogram distances
                should be euclidian, normalized or battacha
            + to_print : (<boolean>) to indicate if we want to print to resulting accuracy
        OUTPUT :
            + prediction : (<list>) of (<np.array>) of size (nb_of_testing_sample,1) representing the predicted output
                for each method
            + accu : (<list>) of (<float>) of size (nb_of_methods) representing the accuracy on the testing_sample for each method
            + methods : (<list>) of (<string>) of size (nb_of_methods) representing the name of the method used to calculate distance
        '''
        histo_train, pola_train = GenerateHistogram(self.event_train)
        histo_test, pola_test = GenerateHistogram(self.event_test)
        prediction = list()
        allmethod = list()
        for each_method in methods:
            output = np.zeros((histo_test.shape[0], 1)).astype(np.str_)
            for idx, each_histo in enumerate(histo_test):

                distance = Norm(each_histo, histo_train, each_method)
                min_dist = np.argmin(distance)
                output[idx, 0] = self.TrainingLabel[min_dist][0]

            prediction.append(output)
            allmethod.append(each_method)
        accu = self.Accuracy(prediction)
        if to_print == True:
            to_write = ''
            for each_accu, each_method in zip(accu, allmethod):
                to_write = str(each_method) + ':' + str(each_accu*100) + '% ### ' + to_write
            print(to_write)

        return prediction, accu, allmethod
