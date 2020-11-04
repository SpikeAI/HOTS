__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

class Network(object):
    def __init__(self, Layers, verbose=0):
        '''
        Class to define a hierarchical network from a list of Layers
            INPUT :
                + Layers : (<list>) list of (<object Layer>) stacking in the order of excecution all the Layers
                    of the Network
                + verbose : (<int>) control the verbosity
        '''
        self.Layers = Layers
        self.verbose = verbose

    def TrainCluster(self, event, NbClusterList, to_record=False, NbCycle=1):
        '''
        Method to train all the layers of the network
        INPUT :
            + event : (<obect event>) the input event of the network
            + NbClusterList : (<list>) of int, stacking the number of cluster of each Clustering Layers
            + to_record :(<boolean>) parameter to record error and histogram during the training
            + NbCycle : <int> number of cycle to perform during the training
        OUTPUT :
            + ClusterList : (<list>) of (<object Cluster>) learned during the training
            + event_o : (<object event>) the output event of the network
        '''
        event_i = event
        idx_Layer = 0
        ClusterList = list()
        for idx, each_Layer in enumerate(self.Layers):
            if each_Layer.type == 'void':
                print('problem !!')
            elif each_Layer.type == 'Filter':
                event_o = each_Layer.RunLayer(event_i)
            elif each_Layer.type == 'Layer':
                event_o, Cluster = each_Layer.TrainLayer(event_i, NbClusterList[idx_Layer], to_record=to_record, NbCycle=NbCycle)
                ClusterList.append(Cluster)
                idx_Layer = idx_Layer + 1
            else:
                print(type(each_Layer))
            event_i = event_o
        return ClusterList, event_o
    
    def RunNetwork(self, event, NbClusterList, homrun=False):
        '''
        Method to run the network
        INPUT :
            + event : (<obect event>) the input event of the network
            + NbClusterList : (<list>) of int, stacking the number of cluster of each Clustering Layers
        OUTPUT :
            + event_o : (<object event>) the output event of the network
        '''
        event_i = event
        idx_Layer = 0
        for idx, each_Layer in enumerate(self.Layers):
            if each_Layer.type == 'void':
                print('problem !!')
            elif each_Layer.type == 'Filter':
                event_o = each_Layer.RunLayer(event_i)
            elif each_Layer.type == 'Layer':
                event_o = each_Layer.RunLayer(event_i, Cluster=NbClusterList[idx_Layer], homrun=homrun)
                idx_Layer = idx_Layer + 1
            else:
                print(type(each_Layer))
            event_i = event_o
        return event_o

    def RunNetwork2(self, event, NbClusterList, homrun=False):
        '''
        Method to run the network
        INPUT :
            + event : (<obect event>) the input event of the network
            + NbClusterList : (<list>) of int, stacking the number of cluster of each Clustering Layers
        OUTPUT :
            + event_o : (<object event>) the output event of the network
        '''
        event_i = event
        event_olz = []
        idx_Layer = 0
        for idx, each_Layer in enumerate(self.Layers):
            if each_Layer.type == 'void':
                print('problem !!')
            elif each_Layer.type == 'Filter':
                event_o = each_Layer.RunLayer(event_i)
            elif each_Layer.type == 'Layer':
                event_o = each_Layer.RunLayer(event_i, Cluster=NbClusterList[idx_Layer], homrun=homrun)
                idx_Layer = idx_Layer + 1
            else:
                print(type(each_Layer))
            event_i = event_o
            event_olz.append(event_o)
        return event_o, event_olz
