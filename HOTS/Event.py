__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import scipy.io
import numpy as np
from random import shuffle
from os import listdir
from HOTS.Tools import LoadObject
import numba
import HOTS.libUnpackAtis as ua


class Event(object):
    '''
    Events is a class representing an event with all his attribute
    ATTRIBUTE
        + polarity : np.array of shape [nb_event] with the polarity number of each event
        + address : np array of shape [nb_event, 2] with the x and y of each event
        + time : np.array of shape [nb_event] with the time stamp of each event
        + ImageSize : (tuple) representing the maximum window where an event could appear
        + ListPolarities : (list) list of the polarity we want to keep
        + ChangeIdx : (list) list composed by the last index of event of each event
        + OutOnePolarity : (bool), transform all polarities into 1 polarity
    '''

    def __init__(self, ImageSize, ListPolarities=None, OutOnePolarity=False):
        self.polarity = np.zeros(1)
        self.address = np.zeros(1)
        self.time = np.zeros(1)
        self.ImageSize = ImageSize
        #self.event_nb = np.zeros(1)
        self.ListPolarities = ListPolarities
        self.ChangeIdx = list()
        self.type = 'event'
        self.OutOnePolarity = OutOnePolarity
        # Idée, faire un mécanisme pour vérifier qu'il n'y a pas d'adresse en dehors de l'image

    def LoadFromMat(self, path, image_number, verbose=0):
        '''
        Load Events from a .mat file. Only the events contained in ListPolarities are kept:
        INPUT
            + path : a string which is the path of the .mat file (ex : './data_cache/ExtractedStabilized.mat')
            + image_number : list with all the numbers of image to load
        '''
        obj = scipy.io.loadmat(path)
        ROI = obj['ROI'][0]

        if type(image_number) is int:
            image_number = [image_number]
        elif type(image_number) is not list:
            raise TypeError(
                'the type of argument image_number should be int or list')
        if verbose > 0:
            print("chargement des images {0}".format(image_number))
        Total_size = 0
        for idx, each_image in enumerate(image_number):
            image = ROI[each_image][0, 0]

            Total_size += image[1].shape[1]

        self.address = np.zeros((Total_size, 2)).astype(int)
        self.time = np.zeros((Total_size))
        self.polarity = np.zeros((Total_size))
        first_idx = 0

        for idx, each_image in enumerate(image_number):
            image = ROI[each_image][0, 0]
            last_idx = first_idx + image[0].shape[1]
            self.address[first_idx:last_idx, 0] = (image[1] - 1).astype(int)
            self.address[first_idx:last_idx, 1] = (image[0] - 1).astype(int)
            self.time[first_idx:last_idx] = (image[3] * 1e-6)
            self.polarity[first_idx:last_idx] = image[2].astype(int)
            first_idx = last_idx

        self.polarity[self.polarity.T == -1] = 0
        self.polarity = self.polarity.astype(int)
        # Filter only the wanted polarity
        self.ListPolarities = np.unique(self.polarity)
        filt = np.in1d(self.polarity, np.array(self.ListPolarities))
        self.filter(filt, mode='itself')
        if self.OutOnePolarity == True:
            self.polarity = np.zeros_like(self.polarity)
            self.ListPolarities = [0]

    def LoadFromBin(self, PathList, verbose=0):
        '''
        Load Events from a .bin file. Only the events contained in ListPolarities are kept:
        INPUT
            + PathList : a list of string representing the path each of the .bin file
        '''
        if type(PathList) is str:
            PathList = [PathList]
        elif type(PathList) not in [list, np.ndarray]:
            raise TypeError(
                'the type of argument image_number should be int or list')

        Total_size = 0
        for idx_path, path in enumerate(PathList):
            with open(path, 'rb') as f:
                a = np.fromfile(f, dtype=np.uint8)
            raw_data = np.uint32(a)
            x = raw_data[0::5]
            Total_size += x.shape[0]

        self.address = np.zeros((Total_size, 2)).astype(int)
        self.time = np.zeros((Total_size))
        self.polarity = np.zeros((Total_size))
        first_idx = 0
        for idx_path, path in enumerate(PathList):
            with open(path, 'rb') as f:
                a = np.fromfile(f, dtype=np.uint8)
            raw_data = np.uint32(a)
            x, y = raw_data[0::5], raw_data[1::5]
            p = (raw_data[2::5] & 128) >> 7
            t = ((raw_data[2::5] & 127) << 16) + \
                ((raw_data[3::5]) << 8) + raw_data[4::5]
            #each_address = np.vstack((y,x)).astype(int).T
            #each_time = (t * 1e-6).T
            each_polarity = p.copy().astype(int)
            each_polarity[each_polarity == 0] = -1
            each_polarity.T
            last_idx = first_idx + x.shape[0]
            self.address[first_idx:last_idx, 0] = y.astype(int).T
            self.address[first_idx:last_idx, 1] = x.astype(int).T
            self.time[first_idx:last_idx] = (t * 1e-6).T
            self.polarity[first_idx:last_idx] = each_polarity.T

            first_idx = last_idx

        # Filter only the wanted polarity

        filt = np.in1d(self.polarity, np.array(self.ListPolarities))
        self.filter(filt, mode='itself')

        if self.OutOnePolarity == True:
            self.polarity = np.zeros_like(self.polarity)
            self.ListPolarities = [0]

    # @numba.jit(nopython=True)
    def SeparateEachImage(self):
        '''
        find the separation event index if more than one image is represented, and store it into
        self.ChangeIDX

        '''

        add2 = self.time[1:]
        add1 = self.time[:-1]
        comp = add1 > add2
        self.ChangeIdx = np.zeros(np.sum(comp)+1).astype(int)
        self.ChangeIdx[:-1] = np.arange(0, comp.shape[0])[comp]
        self.ChangeIdx[-1] = comp.shape[0]

    # @numba.jit(nopython=True)
    def copy(self):
        '''
        copy the address, polarity, timing, and event_nb to another event
        OUTPUT :
            + event_output = event object which is the copy of self
        '''
        event_output = Event(self.ImageSize, self.ListPolarities)
        event_output.address = self.address.copy()
        event_output.polarity = self.polarity.copy()
        event_output.time = self.time.copy()
        event_output.ChangeIdx = self.ChangeIdx
        event_output.type = self.type
        event_output.OutOnePolarity = self.OutOnePolarity

        return event_output

    # @numba.jit(nopython=True)
    def filter(self, filt, mode=None):
        '''
        filter the event is mode is 'itself', or output another event else
        INPUT :
            + filt : np.array of boolean having the same dimension than self.polarity
        OUTPUT :
            + event_output : return an event, which is the filter version of self, only if mode
                is not 'itself'
        '''
        if mode == 'itself':
            self.address = self.address[filt]
            self.time = self.time[filt]
            self.polarity = self.polarity[filt]
            self.SeparateEachImage()
        else:
            event_output = self.copy()
            event_output.address = self.address[filt]
            event_output.time = self.time[filt]
            event_output.polarity = self.polarity[filt]
            event_output.SeparateEachImage()
            return event_output


def SimpleAlphabet(NbTrainingData, NbTestingData, Path=None, LabelPath=None, ClusteringData=None, OutOnePolarity=False, ListPolarities=None, verbose=0):
    '''
    Extract the Data from the SimpleAlphabet DataBase :
    INPUT :
        + NbTrainingData : (int) Number of Training Data
        + NbTestingData : (int) Number of Testing Data
        + Path : (str) Path of the .mat file. If the path is None, the path is ../database/SimpleAlphabet/ExtractedStabilized.mat
        + LabelPath : (str) Path of the .pkl label path. If the path is None, the path is  ../database/SimpleAlphabet/alphabet_label.pkl
        + ClusteringData : (list) a list of int indicating the image used to train the cluster. If None, the image used to train the
            the cluster are the trainingData
        + OutOnePolarity : (bool), transform all polarities into 1 polarity
        + ListPolarities : (list), list of the polarity we want to keep
    OUTPUT :
        + event_tr : (<object event>)
        + event_te : (<object event>)
        + event_cl : (<object event>)
        + label_tr :
        + label_te :
    '''
    if Path is None:
        Path = '../database/SimpleAlphabet/ExtractedStabilized.mat'

    if LabelPath is None:
        label_list = LoadObject(
            '../database/SimpleAlphabet/alphabet_label.pkl')
    else:
        label_list = LoadObject(LabelPath)

    if NbTrainingData+NbTestingData > 76:
        raise NameError('Overlaping between TrainingData and Testing Data')
    event_tr = Event(ImageSize=(
        32, 32), ListPolarities=ListPolarities, OutOnePolarity=OutOnePolarity)
    event_te = Event(ImageSize=(
        32, 32), ListPolarities=ListPolarities, OutOnePolarity=OutOnePolarity)
    event_cl = Event(ImageSize=(
        32, 32), ListPolarities=ListPolarities, OutOnePolarity=OutOnePolarity)
    event_tr.LoadFromMat(Path, image_number=list(
        np.arange(0, NbTrainingData)), verbose=verbose)
    event_te.LoadFromMat(Path, image_number=list(
        np.arange(NbTrainingData, NbTrainingData+NbTestingData)), verbose=verbose)

    if ClusteringData is None:
        event_cl = event_tr
    else:
        event_cl.LoadFromMat(
            Path, image_number=ClusteringData, verbose=verbose)

    # Generate Groud Truth Label
    for idx, img in enumerate(np.arange(0, NbTrainingData)):
        if idx != 0:
            label_tr = np.vstack((label_tr, label_list[img][0]))
        else:
            label_tr = label_list[img][0]

    for idx, img in enumerate(np.arange(NbTrainingData, NbTrainingData+NbTestingData)):
        if idx != 0:
            label_te = np.vstack((label_te, label_list[img][0]))
        else:
            label_te = label_list[img][0]

    return event_tr, event_te, event_cl, label_tr, label_te


def ExtractMNIST(NbTrainingSerie, NbTestingSerie, PathTraining=None, PathTesting=None,
                 ClusteringData=None, OutOnePolarity=False, ListPolarities=[-1, 1], verbose=0):
    '''
    Extract the Data from the SimpleAlphabet DataBase :
    INPUT :
        + NbTrainingSerie : (int) Number of Training Data
        + NbTestingSerie : (int) Number of Testing Data
        + Path : (str) Path of the .mat file. If the path is None, the path is ../database/SimpleAlphabet/ExtractedStabilized.mat
        + LabelPath : (str) Path of the .pkl label path. If the path is None, the path is  ../database/SimpleAlphabet/alphabet_label.pkl
        + ClusteringData : (list) a list of int indicating the image used to train the cluster. If None, the image used to train the
            the cluster are the trainingData
        + OutOnePolarity : (bool), transform all polarities into 1 polarity
        + ListPolarities : (list), list of the polarity we want to keep
    OUTPUT :
        + event_tr : (<object event>)
        + event_te : (<object event>)
        + event_cl : (<object event>)
        + label_tr :
        + label_te :
    '''

    if PathTraining is None:
        PathTraining = '../database/MNIST_Train/'
    if PathTesting is None:
        PathTesting = '../database/MNIST_Test/'

    event_tr = Event(ImageSize=(
        34, 34), ListPolarities=ListPolarities, OutOnePolarity=OutOnePolarity)
    event_te = Event(ImageSize=(
        34, 34), ListPolarities=ListPolarities, OutOnePolarity=OutOnePolarity)
    event_cl = Event(ImageSize=(
        34, 34), ListPolarities=ListPolarities, OutOnePolarity=OutOnePolarity)

    # generate TestingList
    FileListTesting = load_file_list(PathTesting)
    TestingData = np.zeros((NbTestingSerie*10, 2)).astype('<U35')
    # generate Traininglist
    FileListTraining = load_file_list(PathTraining)
    TrainingData = np.zeros((NbTrainingSerie*10, 2)).astype('<U35')

    idx_Tr = 0
    for i in np.arange(10):
        to_pick = FileListTraining[FileListTraining[:, 1] == str(i), 0]
        for EachCycle in range(NbTrainingSerie):
            TrainingData[idx_Tr, 0] = to_pick[EachCycle]
            TrainingData[idx_Tr, 1] = str(i)
            idx_Tr += 1
    np.random.shuffle(TrainingData)
    label_tr = TrainingData[:, 1]

    idx_Te = 0

    for i in np.arange(10):
        to_pick = FileListTesting[FileListTesting[:, 1] == str(i), 0]
        for EachCycle in range(NbTestingSerie):
            TestingData[idx_Te, 0] = to_pick[EachCycle]
            TestingData[idx_Te, 1] = str(i)
            idx_Te += 1
    np.random.shuffle(TestingData)
    label_te = TestingData[:, 1]

    if ClusteringData is None:
        ClusteringData = TrainingData

    event_tr.LoadFromBin(TrainingData[:, 0], verbose=verbose)
    event_te.LoadFromBin(TestingData[:, 0], verbose=verbose)
    event_cl.LoadFromBin(ClusteringData[:, 0], verbose=verbose)

    return event_tr, event_te, event_cl, label_tr, label_te


def load_file_list(mainpath='./Train/'):
    nb_list = list(np.arange(10))
    file_name = list()
    for idx_sample, nb in enumerate(nb_list):
        subdir = mainpath + str(nb) + '/'
        for bin_file in listdir(subdir):
            if bin_file != '.DS_Store':
                tot_dir = subdir + bin_file
                file_name.append((tot_dir, nb))
    shuffle(file_name)
    file_name = np.asarray(file_name)
    return file_name


def GenerateRandomSquare(x, y, t, p, seed=33, image_size=(30, 30)):
    np.random.seed(seed=seed)
    event = Event(ImageSize=image_size)
    event.address = np.zeros((8, 2), dtype=int)
    event.address[0, :] = (x - 1, y - 1)
    event.address[1, :] = (x - 1, y)
    event.address[2, :] = (x - 1, y + 1)
    event.address[3, :] = (x, y - 1)
    event.address[4, :] = (x, y + 1)
    event.address[5, :] = (x + 1, y - 1)
    event.address[6, :] = (x + 1, y)
    event.address[7, :] = (x + 1, y + 1)
    event.address = event.address[np.random.permutation(8), :].T
    event.time = np.zeros(8, dtype=int)
    for idt in np.arange(event.time.size):
        event.time[idt] = t
        t += np.random.randint(0, 10)
    event.polarity = np.ones(8, dtype=int) * p
    return event


def LoadGestureDB(filepath, OutOnePolarity=False):
    ts, c, p, removed_events = ua.readATIS_td(filepath, orig_at_zero=True,
                                              drop_negative_dt=True, verbose=False)
    event = Event(ImageSize=(304, 240))
    # print(p.shape)
    event.time = ts * 1e-6
    if OutOnePolarity == False:
        event.polarity = p
    else:
        event.polarity = np.zeros_like(p)
    event.ListPolarities = np.unique(event.polarity)
    event.address = c
    return event
