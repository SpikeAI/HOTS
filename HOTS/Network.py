import numpy as np
import matplotlib.pyplot as plt
from Layer import layer
from TimeSurface import TimeSurface
from Stats import stats
from tqdm import tqdm
import tonic
import os
import pickle
from torch import Generator
from torch.utils.data import SubsetRandomSampler

class network(object):
    """network is an Hierarchical network described in Lagorce et al. 2017 (HOTS). It loads event stream with the tonic package.
    METHODS: .load -> loads datasets thanks to tonic package built on Pytorch.
             .learning1by1 -> makes the unsupervised clustering of the different layers 1 layer after the other
             .learningall -> makes the online unsupervised clustering of the different layers
             .running -> run the network and output either an averaged histogram for each class (train=True), either an histogram for each digit/video as input (train=False) either the stream of events as output of the last layer (LR=True)
             .run -> computes the run of an event as input of the network
             .get_fname -> returns the name of the network depending on its parameters
             .plotlayer -> plots the histogram of activation of the different layers ad associated kernels
             .plotconv -> plots the convergence of the layers during learning phase
             .plotactiv -> plots the activation map of each layer
             """

    def __init__(self,  timestr = None,
                        # architecture of the network (default=Lagorce2017)
                        nbclust = [4, 8, 16],
                        # parameters of time-surfaces and datasets
                        tau = 10, #timestamp en millisec/
                        K_tau = 10,
                        decay = 'exponential', # among ['exponential', 'linear']
                        nbpolcam = 2,
                        R = 2,
                        K_R = 2,
                        camsize = (34, 34),
                        # functional parameters of the network
                        algo = 'lagorce', # among ['lagorce', 'maro', 'mpursuit']
                        krnlinit = 'rdn',
                        hout = False, #works only with mpursuit
                        homeo = False,
                        homparam = [.25, 1],
                        pola = True,
                        to_record = True,
                        filt = 2,
                        sigma = None,
                        jitter = False,
                        homeinv = False,
                ):
        self.jitter = jitter # != from jitonic, this jitter is added at the layer output, creating an average pooling
        self.onbon = False
        self.name = 'hots'
        self.date = timestr
        tau *= 1e3 # to enter tau in ms
        nblay = len(nbclust)
        if to_record:
            self.stats = [[]]*nblay
        self.TS = [[]]*nblay
        self.L = [[]]*nblay
        for lay in range(nblay):
            if lay == 0:
                self.TS[lay] = TimeSurface(R, tau, camsize, nbpolcam, pola, filt, sigma)
                self.L[lay] = layer(R, nbclust[lay], pola, nbpolcam, homeo, homparam, homeinv, algo, hout, krnlinit, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust[lay], camsize)
            else:
                self.TS[lay] = TimeSurface(R*(K_R**lay), tau*(K_tau**lay), camsize, nbclust[lay-1], pola, filt, sigma)
                #self.L[lay] = layer(R*(K_R**lay), nbclust*(K_clust**lay), pola, nbclust*(K_clust**(lay-1)), homeo, homparam, homeinv, algo, hout, krnlinit, to_record)
                self.L[lay] = layer(R*(K_R**lay), nbclust[lay], pola, nbclust[lay-1], homeo, homparam, homeinv, algo, hout, krnlinit, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust[lay], camsize)

##___________________________________________________________________________________________

    def load(self, dataset, trainset=True, jitonic=[None,None], subset_size = None, kfold = None, kfold_ind = None):

        if jitonic[1] is not None:
            print(f'spatial jitter -> var = {jitonic[1]}')
            transform = tonic.transforms.Compose([tonic.transforms.SpatialJitter(variance_x=jitonic[1], variance_y=jitonic[1], sigma_x_y=0, integer_coordinates=True, clip_outliers=True)])

        if jitonic[0] is not None:
            print(f'time jitter -> var = {jitonic[0]}')
            transform = tonic.transforms.Compose([tonic.transforms.TimeJitter(variance=jitonic[0], integer_timestamps=False, clip_negative=True, sort_timestamps=True)])

        if jitonic == [None,None]:
            print('no jitter')
            transform = None

        download=False
        path = '../Data/'
        if dataset == 'nmnist':
            if trainset:
                path+='Train/'
            else:
                path+='Test/'
            if not os.path.exists(path):
                download=True

            eventset = tonic.datasets.NMNIST(save_to='../Data/',
                                train=trainset, download=download,
                                transform=transform)
        elif dataset == 'poker':
            if trainset:
                path+='pips_train/'
            else:
                path+='pips_test/'
            if not os.path.exists(path):
                download=True
            eventset = tonic.datasets.POKERDVS(save_to='../Data/',
                                train=trainset, download=download,
                                transform=transform)
        elif dataset == 'gesture':
            if trainset:
                path+='ibmGestureTrain/'
            else:
                path+='ibmGestureTest/'
            if not os.path.exists(path):
                download=True
            eventset = tonic.datasets.DVSGesture(save_to='../Data/',
                                train=trainset, download=download,
                                transform=transform)
        elif dataset == 'cars':
            if trainset:
                path+='ncars-train/'
            else:
                path+='ncars-test/'
            if not os.path.exists(path):
                download=True
            eventset = tonic.datasets.NCARS(save_to='../Data/',
                                train=trainset, download=download,
                                transform=transform)
        elif dataset == 'ncaltech':
            eventset = tonic.datasets.NCALTECH101(save_to='../Data/',
                                train=trainset, download=download,
                                transform=transform)
        else: print('incorrect dataset')

        if subset_size is not None:
            subset_indices = []
            for i in range(len(eventset.classes)):
                all_ind = np.where(np.array(eventset.targets)==i)[0]
                subset_indices += all_ind[:subset_size//len(eventset.classes)].tolist()
            g_cpu = Generator()
            subsampler = SubsetRandomSampler(subset_indices, g_cpu)
            loader = tonic.datasets.DataLoader(eventset, batch_size=1, shuffle=False, sampler=subsampler)
        elif kfold is not None:
            subset_indices = []
            subset_size = len(testset)//kfold
            for i in range(len(testset.classes)):
                all_ind = np.where(np.array(testset.targets)==i)[0]
                subset_indices += all_ind[kfold_ind*subset_size//len(testset.classes):
                            min((kfold_ind+1)*subset_size//len(testset.classes), len(testset)-1)].tolist()
            g_cpu = Generator()
            subsampler = SubsetRandomSampler(subset_indices, g_cpu)
            loader = tonic.datasets.DataLoader(testset, batch_size=1, shuffle=False, sampler=subsampler)
        else:
            loader = tonic.datasets.DataLoader(eventset, shuffle=True)

        if eventset.sensor_size!=self.TS[0].camsize:
            print('sensor formatting...')
            self.sensformat(eventset.sensor_size)

        return loader, eventset.ordering, eventset.classes

    def sensformat(self,sensor_size):
        for i in range(1,len(self.TS)):
            self.TS[i].camsize = sensor_size
            self.TS[i].spatpmat = np.zeros((self.L[i-1].kernel.shape[1],sensor_size[0]+1,sensor_size[1]+1))
            self.stats[i].actmap = np.zeros((self.L[i-1].kernel.shape[1],sensor_size[0]+1,sensor_size[1]+1))
        self.TS[0].camsize = sensor_size
        self.TS[0].spatpmat = np.zeros((2,sensor_size[0]+1,sensor_size[1]+1))
        self.stats[0].actmap = np.zeros((2,sensor_size[0]+1,sensor_size[1]+1))

    def learning1by1(self, nb_digit=10, dataset='nmnist', diginit=True, filtering=None, jitonic=[None,None], maxevts=None, subset_size = None, kfold = None, kfold_ind = None, ds_ev = None, verbose=True):
        self.onbon = True
        model = self.load_model(dataset, verbose)
        if model:
            return model
        else:
            loader, ordering, classes = self.load(dataset, jitonic=jitonic, subset_size=subset_size)
            nbclass = len(classes)
            #eventslist = [next(iter(loader))[0] for i in range(nb_digit)]
            eventslist = []
            nbloadz = np.zeros([nbclass])
            while np.sum(nbloadz)<nb_digit*nbclass:
                loadev, loadtar = next(iter(loader))
                if nbloadz[loadtar]<nb_digit:
                    eventslist.append(loadev)
                    nbloadz[loadtar]+=1

            for n in range(len(self.L)):
                pbar = tqdm(total=nb_digit*nbclass)
                for idig in range(nb_digit*nbclass):
                    pbar.update(1)
                    events = eventslist[idig]

                    if dataset=='cars':
                        size_x = max(events[0,:,ordering.find("x")])-min(events[0,:,ordering.find("x")])
                        size_y = max(events[0,:,ordering.find("y")])-min(events[0,:,ordering.find("y")])
                        self.sensformat((int(size_x.item()),int(size_y.item())))
                        events[0,:,ordering.find("x")] -= min(events[0,:,ordering.find("x")]).numpy()
                        events[0,:,ordering.find("y")] -= min(events[0,:,ordering.find("y")]).numpy()
                    if diginit:
                        for l in range(n+1):
                            self.TS[l].spatpmat[:] = 0
                            self.TS[l].iev = 0
                    if ds_ev is not None:
                        events = events[:,::ds_ev,:]
                    if maxevts is not None:
                        N_max = min(maxevts, events.shape[1])
                    else:
                        N_max = events.shape[1]

                    for iev in range(N_max):
                        x,y,t,p =   events[0,iev,ordering.find("x")].item(), \
                                    events[0,iev,ordering.find("y")].item(), \
                                    events[0,iev,ordering.find("t")].item(), \
                                    events[0,iev,ordering.find("p")].item()
                        lay=0
                        while lay < n+1:
                            if lay==n:
                                learn=True
                            else:
                                learn=False
                            timesurf, activ = self.TS[lay].addevent(x, y, t, p)
                            if lay==0 or filtering=='all':
                                activ2=activ
                            if activ2 and np.sum(timesurf)>0:
                            #if activ==True:
                                p, dist = self.L[lay].run(timesurf, learn)
                                if learn:
                                    self.stats[lay].update(p, self.L[lay].kernel, timesurf, dist)
                                if self.jitter:
                                    x,y = spatial_jitter(x,y,self.TS[0].camsize)
                                lay += 1
                            else:
                                lay = n+1
                pbar.close()
            for l in range(len(self.L)):
                self.stats[l].histo = self.L[l].cumhisto.copy()
            self.save_model(dataset)
            return self

    def learningall(self, nb_digit=10, dataset='nmnist', diginit=True, jitonic=[None,None], maxevts = None, subset_size=None, kfold = None, kfold_ind = None, ds_ev = None, verbose=True):

        self.onbon = False
        model = self.load_model(dataset, verbose)
        if model:
            return model
        else:
            loader, ordering, classes = self.load(dataset, jitonic=jitonic, subset_size=subset_size)
            nbclass = len(classes)
            pbar = tqdm(total=nb_digit*nbclass)
            nbloadz = np.zeros([nbclass])
            while np.sum(nbloadz)<nb_digit*nbclass:
                if diginit:
                    for i in range(len(self.L)):
                        self.TS[i].spatpmat[:] = 0
                        self.TS[i].iev = 0
                events, target = next(iter(loader))

                if nbloadz[target]<nb_digit:
                    nbloadz[target]+=1
                    pbar.update(1)
                    if ds_ev is not None:
                        events = events[:,::ds_ev,:]
                    if maxevts is not None:
                        N_max = min(maxevts, events.shape[1])
                    else:
                        N_max = events.shape[1]
                    if dataset=='cars':
                        size_x = max(events[0,:,ordering.find("x")])-min(events[0,:,ordering.find("x")])+1
                        size_y = max(events[0,:,ordering.find("y")])-min(events[0,:,ordering.find("y")])+1
                        self.sensformat((int(size_x.item()),int(size_y.item())))
                        events[0,:,ordering.find("x")] -= min(events[0,:,ordering.find("x")]).numpy()
                        events[0,:,ordering.find("y")] -= min(events[0,:,ordering.find("y")]).numpy()
                    for iev in range(N_max):
                        self.run(events[0][iev][ordering.find("x")].item(), \
                                 events[0][iev][ordering.find("y")].item(), \
                                 events[0][iev][ordering.find("t")].item(), \
                                 events[0][iev][ordering.find("p")].item(), \
                                 learn=True, to_record=True)
                        #if self.TS[0].iev%1000==0:
                        #    self.TS[0].plote()
            pbar.close()
            for l in range(len(self.L)):
                self.stats[l].histo = self.L[l].cumhisto.copy()
            self.save_model(dataset)
            return self

    def running(self, homeotest=False, train=True, outstyle='histo', nb_digit=500, jitonic=[None,None], dataset='nmnist', maxevts = None, subset_size=None, kfold = None, kfold_ind = None, ds_ev = None, to_record=False, verbose=True):

        output, loaded = self.load_output(dataset, homeotest, nb_digit, train, jitonic, outstyle, kfold_ind, verbose)
        if loaded:
            return output
        else:
            loader, ordering, classes = self.load(dataset, trainset=train, jitonic=jitonic, subset_size=subset_size)
            nbclass = len(classes)
            homeomod = self.L[0].homeo
            for i in range(len(self.L)):
                self.L[i].homeo=homeotest
            pbar = tqdm(total=nb_digit)
            timout = []
            xout = []
            yout = []
            polout = []
            labout = []
            labelmap = []

            labelmapav = np.zeros([nbclass, len(self.L[-1].cumhisto)])
            labelcount = np.zeros(nbclass)

            x_index = ordering.find("x")
            y_index = ordering.find("y")
            t_index = ordering.find("t")
            p_index = ordering.find("p")

            for events, target in loader:
                for i in range(len(self.L)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
                    self.L[i].cumhisto[:] = 1
                    #self.stats[i].actmap[:] = 0
                pbar.update(1)
                if ds_ev is not None:
                    events = events[:,::ds_ev,:]
                if maxevts is not None:
                    N_max = min(maxevts, events.shape[1])
                else:
                    N_max = events.shape[1]
                if dataset=='cars':
                    size_x = max(events[0,:,ordering.find("x")])-min(events[0,:,ordering.find("x")])
                    size_y = max(events[0,:,ordering.find("y")])-min(events[0,:,ordering.find("y")])
                    self.sensformat((int(size_x.item()),int(size_y.item())))
                    events[0,:,ordering.find("x")] -= min(events[0,:,ordering.find("x")]).numpy()
                    events[0,:,ordering.find("y")] -= min(events[0,:,ordering.find("y")]).numpy()

                for iev in range(N_max):
                    if events[0][iev][x_index].item()>33:
                        print('aïe', events[0][iev][x_index].item())
                    
                    out, activout = self.run(events[0][iev][x_index].item(), \
                                            events[0][iev][y_index].item(), \
                                            events[0][iev][t_index].item(), \
                                            events[0][iev][p_index].item(), \
                                            to_record=to_record)
                    if outstyle=='LR' and activout:
                        xout.append(out[0])
                        yout.append(out[1])
                        timout.append(out[2])
                        polout.append(out[3])
                        labout.append(target.item())

                if train:
                    labelmapav[target.item(),:] += self.L[-1].cumhisto.copy()/np.sum(self.L[-1].cumhisto.copy())
                    labelcount[target.item()] += 1
                    for i in range(len(labelcount)):
                        labelmapav[i,:] /= max(labelcount[i],1)
                data = (target.item(),self.L[-1].cumhisto.copy()/np.sum(self.L[-1].cumhisto.copy()))
                labelmap.append(data)

            for i in range(len(self.L)):
                self.L[i].homeo=homeomod

            pbar.close()

            if train:
                self.save_output(labelmapav, homeotest, dataset, nb=nb_digit, train=train, jitonic=jitonic, outstyle='histav', kfold_ind=kfold_ind)
            self.save_output(labelmap, homeotest, dataset, nb=nb_digit, train=train, jitonic=jitonic, outstyle='histo', kfold_ind=kfold_ind)

            if outstyle=='LR':
                camsize = self.TS[-1].camsize
                nbpola = self.L[-1].kernel.shape[1]
                eventsout = [xout,yout,timout,polout,labout,camsize,nbpola]
                self.save_output(eventsout, homeotest, dataset, nb=nb_digit, train=train, jitonic=jitonic, outstyle='LR', kfold_ind=kfold_ind)
                output = eventsout
            elif outstyle=='histo':
                output = labelmap
            elif outstyle=='histav':
                output = labelmapav
            return output

    def run(self, x, y, t, p, learn=False, to_record=False):
        lay = 0
        activout=False
        while lay<len(self.TS):
            timesurf, activ = self.TS[lay].addevent(x, y, t, p)
            if activ:
                p, dist = self.L[lay].run(timesurf, learn)
                if to_record:
                    self.stats[lay].update(p, self.L[lay].kernel, timesurf, dist)
                    #self.stats[lay].actmap[int(np.argmax(p)),self.TS[lay].x,self.TS[lay].y]=1
                if self.jitter:
                    x,y = spatial_jitter(x,y,self.TS[0].camsize)
                lay+=1
                if lay==len(self.TS):
                    activout=True
            else:
                lay = len(self.TS)
        out = [x,y,t,np.argmax(p)]
        return out, activout

    def get_fname(self):
        timestr = self.date
        algo = self.L[0].algo
        arch = [self.L[i].kernel.shape[1] for i in range(len(self.L))]
        R = [self.L[i].R for i in range(len(self.L))]
        tau = [np.round(self.TS[i].tau*1e-3,2) for i in range(len(self.TS))]
        homeo = self.L[0].homeo
        homparam = self.L[0].homparam
        krnlinit = self.L[0].krnlinit
        sigma = self.TS[0].sigma
        onebyone = self.onbon
        f_name = f'{timestr}_{algo}_{krnlinit}_{sigma}_{homeo}_{homparam}_{arch}_{tau}_{R}_{onebyone}'
        self.name = f_name
        return f_name

    def save_model(self, dataset):
        if dataset=='nmnist':
            path = '../Records/EXP_03_NMNIST/models/'
        elif dataset=='cars':
            path = '../Records/EXP_04_NCARS/models/'
        elif dataset=='poker':
            path = '../Records/EXP_05_POKERDVS/models/'
        elif dataset=='gesture':
            path = '../Records/EXP_06_DVSGESTURE/models/'
        else: print('define a path for this dataset')
        if not os.path.exists(path):
            os.makedirs(path)
        f_name = path+self.get_fname()+'.pkl'
        with open(f_name, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def load_model(self, dataset, verbose):
        model = []
        if dataset=='nmnist':
            path = '../Records/EXP_03_NMNIST/models/'
        elif dataset=='cars':
            path = '../Records/EXP_04_NCARS/models/'
        elif dataset=='poker':
            path = '../Records/EXP_05_POKERDVS/models/'
        elif dataset=='gesture':
            path = '../Records/EXP_06_DVSGESTURE/models/'
        else: print('define a path for this dataset')
        f_name = path+self.get_fname()+'.pkl'
        if verbose:
            print(f_name)
        if not os.path.isfile(f_name):
            return model
        else:
            with open(f_name, 'rb') as file:
                model = pickle.load(file)
        return model

    def save_output(self, evout, homeo, dataset, nb, train, jitonic, outstyle, kfold_ind):
        if dataset=='nmnist':
            direc = 'EXP_03_NMNIST'
        elif dataset=='cars':
            direc = 'EXP_04_NCARS'
        elif dataset=='poker':
            direc = 'EXP_05_POKERDVS'
        elif dataset=='gesture':
            direc = 'EXP_06_DVSGESTURE'
        elif dataset=='barrel':
            direc = 'EXP_01_LagorceKmeans'
        else: print('define a path for this dataset')
        if train:
            path = f'../Records/{direc}/train/'
        else:
            path = f'../Records/{direc}/test/'
        if not os.path.exists(path):
            os.makedirs(path)
        f_name = path+self.get_fname()+f'_{nb}_{jitonic}_{outstyle}'
        if kfold_ind is not None:
            f_name+='_'+str(kfold_ind)
        if homeo:
            f_name = f_name+'_homeo'
        f_name = f_name +'.pkl'
        with open(f_name, 'wb') as file:
            pickle.dump(evout, file, pickle.HIGHEST_PROTOCOL)

    def load_output(self, dataset, homeo, nb, train, jitonic, outstyle, kfold_ind, verbose):
        loaded = False
        output = []
        if dataset=='nmnist':
            direc = 'EXP_03_NMNIST'
        elif dataset=='cars':
            direc = 'EXP_04_NCARS'
        elif dataset=='poker':
            direc = 'EXP_05_POKERDVS'
        elif dataset=='gesture':
            direc = 'EXP_06_DVSGESTURE'
        else: print('define a path for this dataset')
        if train:
            path = f'../Records/{direc}/train/'
        else:
            path = f'../Records/{direc}/test/'
        f_name = path+self.get_fname()+f'_{nb}_{jitonic}_{outstyle}'
        if kfold_ind is not None:
            f_name+='_'+str(kfold_ind)
        if homeo:
            f_name = f_name+'_homeo'
        f_name = f_name +'.pkl'
        if verbose:
            print(f_name)
        if os.path.isfile(f_name):
            with open(f_name, 'rb') as file:
                output = pickle.load(file)
            loaded = True
        return output, loaded


##___________REPRODUCING RESULTS FROM LAGORCE 2017___________________________________________
##___________________________________________________________________________________________

    def learninglagorce(self, nb_cycle=3, diginit=True, filtering=None):


        #___________ SPECIAL CASE OF SIMPLE_ALPHABET DATASET _________________

        path = "../Data/alphabet_ExtractedStabilized.mat"

        image_list = [1, 32, 19, 22, 29]
        for i in range(nb_cycle-1):
            image_list += image_list
        address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)

        #___________ SPECIAL CASE OF SIMPLE_ALPHABET DATASET _________________

        nbevent = int(time.shape[0])
        for n in range(len(self.L)):
            count = 0
            pbar = tqdm(total=nbevent)
            while count<nbevent:
                pbar.update(1)
                x,y,t,p = address[count,0],address[count,1], time[count],polarity[count]
                if diginit and time[count]<time[count-1]:
                    for i in range(n+1):
                        self.TS[i].spatpmat[:] = 0
                        self.TS[i].iev = 0
                lay=0
                while lay < n+1:
                    if lay==n:
                        learn=True
                    else:
                        learn=False
                    timesurf, activ = self.TS[lay].addevent(x, y, t, p)
                    if lay==0 or filtering=='all':
                        activ2=activ
                    if activ2 and np.sum(timesurf)>0:
                        p, dist = self.L[lay].run(timesurf, learn)
                        if learn:
                            self.stats[lay].update(p, self.L[lay].kernel, timesurf, dist)
                        lay += 1
                    else:
                        lay = n+1
                count += 1
            for l in range(len(self.L)):
                self.stats[l].histo = self.L[l].cumhisto.copy()
            pbar.close()

    def traininglagorce(self, nb_digit=None, outstyle = 'histo', to_record=True):
        
        class_data = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25,
    "0": 26,
    "1": 27,
    "2": 28,
    "3": 29,
    "4": 30,
    "5": 31,
    "6": 32,
    "7": 33,
    "8": 34,
    "9": 35,
}
        
        path = "../Data/alphabet_ExtractedStabilized.mat"
        nblist = 36
        image_list=list(np.arange(0, nblist))
        address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)
        with open('../Data/alphabet_label.pkl', 'rb') as file:
            label_list = pickle.load(file)
        label = label_list[:nblist]

        learn=False
        output = []
        count = 0
        count2 = 0
        nbevent = int(time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        labelmap = []
        timout = []
        xout = []
        yout = []
        polout = []
        labout = []
        for i in range(len(self.L)):
            self.TS[i].spatpmat[:] = 0
            self.TS[i].iev = 0
            self.L[i].cumhisto[:] = 1

        while count<nbevent:
            pbar.update(1)
            out, activout = self.run(address[count,0],address[count,1],time[count],polarity[count], learn, to_record)
            if outstyle=='LR' and activout:
                xout.append(out[0])
                yout.append(out[1])
                timout.append(out[2])
                polout.append(out[3])
                labout.append(class_data[label[idx][0]])
                
            if count2==label[idx][1]:
                data = (label[idx][0],self.L[-1].cumhisto.copy())
                labelmap.append(data)
                for i in range(len(self.L)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
                    self.L[i].cumhisto[:] = 1
                idx += 1
                count2=-1
            count += 1
            count2 += 1
        pbar.close()
        if outstyle=='LR':
            camsize = self.TS[-1].camsize
            nbpola = self.L[-1].kernel.shape[1]
            eventsout = [xout,yout,timout,polout,labout,camsize,nbpola]
            self.date = '2020-12-01'
            self.save_output(eventsout, False, 'barrel', len(label), True, None, 'LR', None)
        return labelmap

    def testinglagorce(self, nb_digit=None, outstyle = 'histo', to_record=True):
        
        class_data = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25,
    "0": 26,
    "1": 27,
    "2": 28,
    "3": 29,
    "4": 30,
    "5": 31,
    "6": 32,
    "7": 33,
    "8": 34,
    "9": 35,
}
        
        path = "../Data/alphabet_ExtractedStabilized.mat"
        image_list=list(np.arange(36, 76))
        address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)
        with open('../Data/alphabet_label.pkl', 'rb') as file:
            label_list = pickle.load(file)
        label = label_list[36:76]

        learn = False
        output = []
        count = 0
        count2 = 0
        nbevent = int(time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        labelmap = []
        timout = []
        xout = []
        yout = []
        polout = []
        labout = []
        for i in range(len(self.L)):
            self.TS[i].spatpmat[:] = 0
            self.TS[i].iev = 0
            self.L[i].cumhisto[:] = 1
        while count<nbevent:
            pbar.update(1)
            out, activout = self.run(address[count,0],address[count,1],time[count],polarity[count], learn, to_record)
            if outstyle=='LR' and activout:
                xout.append(out[0])
                yout.append(out[1])
                timout.append(out[2])
                polout.append(out[3])
                labout.append(class_data[label[idx][0]])
            if count2==label[idx][1]:
                data = (label[idx][0],self.L[-1].cumhisto.copy())
                labelmap.append(data)
                for i in range(len(self.L)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
                    self.L[i].cumhisto[:] = 1
                idx += 1
                count2=-1
            count += 1
            count2 += 1

        pbar.close()
        if outstyle=='LR':
            camsize = self.TS[-1].camsize
            nbpola = self.L[-1].kernel.shape[1]
            eventsout = [xout,yout,timout,polout,labout,camsize,nbpola]
            self.date = '2020-12-01'
            self.save_output(eventsout, False, 'barrel', len(label), False, None, 'LR', None)

        return labelmap

##___________________PLOTTING________________________________________________________________
##___________________________________________________________________________________________

    def plotlayer(self, maxpol=None, hisiz=2, yhis=0.3):
        '''
        '''
        N = []
        P = [2]
        R2 = []
        for i in range(len(self.L)):
            N.append(int(self.L[i].kernel.shape[1]))
            if i>0:
                P.append(int(self.L[i-1].kernel.shape[1]))
            R2.append(int(self.L[i].kernel.shape[0]/P[i]))
        if maxpol is None:
            maxpol=P[-1]

        fig = plt.figure(figsize=(16,9))
        gs = fig.add_gridspec(np.sum(P)+hisiz, np.sum(N)+len(self.L)-1, wspace=0.05, hspace=0.05)
        if self.L[-1].homeo:
            fig.suptitle('Activation histograms and associated time surfaces with homeostasis', size=20, y=0.95)
        else:
            fig.suptitle('Activation histograms and associated time surfaces for original hots', size=20, y=0.95)

        for i in range(len(self.L)):
            ax = fig.add_subplot(gs[:hisiz, int(np.sum(N[:i]))+1*i:int(np.sum(N[:i+1]))+i*1])
            plt.bar(np.arange(N[i]), self.stats[i].histo/np.sum(self.stats[i].histo), width=1, align='edge', ec="k")
            ax.set_xticks(())
            #if i>0:
                #ax.set_yticks(())
            ax.set_title('Layer '+str(i+1), fontsize=16)
            plt.xlim([0,N[i]])
            yhis = 1.1*max(self.stats[i].histo/np.sum(self.stats[i].histo))
            plt.ylim([0,yhis])

        #f3_ax1.set_title('gs[0, :]')
            for k in range(N[i]):
                vmaxi = max(self.L[i].kernel[:,k])
                for j in range(P[i]):
                    if j>maxpol-1:
                        pass
                    else:
                        axi = fig.add_subplot(gs[j+hisiz,k+1*i+int(np.sum(N[:i]))])
                        krnl = self.L[i].kernel[j*R2[i]:(j+1)*R2[i],k].reshape((int(np.sqrt(R2[i])), int(np.sqrt(R2[i]))))

                        axi.imshow(krnl, vmin=0, vmax=vmaxi, cmap=plt.cm.plasma, interpolation='nearest')
                        axi.set_xticks(())
                        axi.set_yticks(())
        plt.show()
        return fig

    def plotconv(self):
        fig = plt.figure(figsize=(15,5))
        for i in range(len(self.L)):
            ax1 = fig.add_subplot(1,len(self.stats),i+1)
            x = np.arange(len(self.stats[i].dist))
            ax1.plot(x, self.stats[i].dist)
            ax1.set(ylabel='error', xlabel='events (x'+str(self.stats[i].nbqt)+')', title='Mean error (eucl. dist) on '+str(self.stats[i].nbqt)+' events - Layer '+str(i+1))
        #ax1.title.set_color('w')
            ax1.tick_params(axis='both')

    def plotactiv(self, maxpol=None):
        N = []
        for i in range(len(self.L)):
            N.append(int(self.L[i].kernel.shape[1]))

        fig = plt.figure(figsize=(16,5))
        gs = fig.add_gridspec(len(self.L), np.max(N), wspace=0.05, hspace=0.05)
        fig.suptitle('Activation maps of the different layers', size=20, y=0.95)

        for i in range(len(self.L)):
            for k in range(N[i]):
                    axi = fig.add_subplot(gs[i,k])
                    axi.imshow(self.stats[i].actmap[k].T, cmap=plt.cm.plasma, interpolation='nearest')
                    axi.set_xticks(())
                    axi.set_yticks(())
                    
    def plotTS(self, maxpol=None):
        N = []
        for i in range(len(self.TS)):
            N.append(int(self.TS[i].spatpmat.shape[1]))

        fig = plt.figure(figsize=(16,5))
        gs = fig.add_gridspec(len(self.TS), np.max(N), wspace=0.05, hspace=0.05)
        fig.suptitle('Global TS of the different layers', size=20, y=0.95)

        for i in range(len(self.TS)):
            for k in range(N[i]):
                axi = fig.add_subplot(gs[i,k])
                axi.imshow(self.TS[i].spatpmat, cmap=plt.cm.plasma, interpolation='nearest')
                axi.set_xticks(())
                axi.set_yticks(())


##________________POOLING NETWORK____________________________________________________________
##___________________________________________________________________________________________


class poolingnetwork(network):

    def __init__(self,
                        # architecture of the network (default=Lagorce2017)
                        nbclust = 4,
                        K_clust = 2, # nbclust(L+1) = K_clust*nbclust(L)
                        nblay = 3,
                        # parameters of time-surfaces and datasets
                        tau = 10, #timestamp en millisec/
                        K_tau = 10,
                        decay = 'exponential', # among ['exponential', 'linear']
                        nbpolcam = 2,
                        R = 2,
                        K_R = 2,
                        camsize = (34, 34),
                        # functional parameters of the network
                        algo = 'lagorce', # among ['lagorce', 'maro', 'mpursuit']
                        krnlinit = 'rdn',
                        hout = False, #works only with mpursuit
                        homeo = False,
                        homparam = [.25, 1],
                        pola = True,
                        to_record = True,
                        filt = 2,
                        sigma = None,
                        jitter = False,
                        homeinv = False,

                        Kstride = 2,
                        Kevtstr = False,
                ):
        super().__init__(
                        nbclust = nbclust,
                        K_clust = K_clust,
                        nblay = nblay,
                        tau = tau,
                        K_tau = K_tau,
                        decay = decay,
                        nbpolcam = nbpolcam,
                        R = R,
                        K_R = K_R,
                        camsize = camsize,
                        algo = algo,
                        krnlinit = krnlinit,
                        hout = hout,
                        homeo = homeo,
                        homparam = homparam,
                        pola = pola,
                        to_record = to_record,
                        filt = filt,
                        sigma = sigma,
                        jitter = jitter,
                        homeinv = homeinv
                )

        self.Kstride = Kstride
        self.Kevtstr = Kevtstr
        for lay in range(1,nblay):
            camsize = np.array(camsize)//Kstride
            self.TS[lay] = TimeSurface(R, tau*(K_tau**lay), camsize, nbclust*(K_clust**(lay-1)), pola, filt, sigma)
            self.L[lay] = layer(R, 16, pola, nbclust*(K_clust**(lay-1)), homeo, homparam, homeinv, algo, hout, krnlinit, to_record)
            self.stats[lay] = stats(nbclust*(K_clust**lay), camsize)

 ##____________________________________________________________________________________

    def run(self, x, y, t, p, learn=False, to_record=False):
        lay = 0
        activout=False
        while lay<len(self.TS):
            timesurf, activ = self.TS[lay].addevent(x, y, t, p)
            if activ:
                p, dist = self.L[lay].run(timesurf, learn)
                if to_record:
                    self.stats[lay].update(p, self.L[lay].kernel, timesurf, dist)
                    self.stats[lay].actmap[int(np.argmax(p)),self.TS[lay].x,self.TS[lay].y]=1
                if self.jitter:
                    x,y = spatial_jitter(x,y,self.TS[0].camsize)
                # pooling
                if lay<len(self.TS)-1:
                    x = min(x//self.Kstride,self.TS[lay+1].camsize[0]-1)
                    y = min(y//self.Kstride,self.TS[lay+1].camsize[1]-1)
                lay+=1
                if lay==len(self.TS):
                    activout=True
            else:
                lay = len(self.TS)
        out = [x,y,t,np.argmax(p)]
        return out, activout


    def learning1by1(self, nb_digit=2, dataset='nmnist', diginit=True, filtering=None):

        loader, ordering, classes = self.load(dataset)
        nbclass = len(classes)
        #eventslist = [next(iter(loader))[0] for i in range(nb_digit)]
        eventslist = []
        nbloadz = np.zeros([nbclass])
        while np.sum(nbloadz)<nb_digit*nbclass:
            loadev, loadtar = next(iter(loader))
            if nbloadz[loadtar]<nb_digit:
                eventslist.append(loadev)
                nbloadz[loadtar]+=1

        for n in range(len(self.L)):
            pbar = tqdm(total=nb_digit*nbclass)
            for idig in range(nb_digit*nbclass):
                pbar.update(1)
                events = eventslist[idig]
                if diginit:
                    for l in range(n+1):
                        self.TS[l].spatpmat[:] = 0
                        self.TS[l].iev = 0
                for iev in range(events.shape[1]):
                    x,y,t,p =   events[0,iev,ordering.find("x")].item(), \
                                events[0,iev,ordering.find("y")].item(), \
                                events[0,iev,ordering.find("t")].item(), \
                                events[0,iev,ordering.find("p")].item()
                    lay=0
                    while lay < n+1:
                        if lay==n:
                            learn=True
                        else:
                            learn=False
                        timesurf, activ = self.TS[lay].addevent(x, y, t, p)
                        if lay==0 or filtering=='all':
                            activ2=activ
                        if activ2 and np.sum(timesurf)>0:
                        #if activ==True:
                            p, dist = self.L[lay].run(timesurf, learn)
                            if learn:
                                self.stats[lay].update(p, self.L[lay].kernel, timesurf, dist)
                            if self.jitter:
                                x,y = spatial_jitter(x,y,self.TS[0].camsize)
                            # no stride for the last layer
                            if lay<len(self.TS)-1:
                                x = min(x//self.Kstride,self.TS[lay+1].camsize[0]-1)
                                y = min(y//self.Kstride,self.TS[lay+1].camsize[1]-1)
                            lay += 1
                        else:
                            lay = n+1
            pbar.close()
        for l in range(len(self.L)):
            self.stats[l].histo = self.L[l].cumhisto.copy()
        return loader, ordering


#__________________OLD_CODE___________________________________________________________________________
#_____________________________________________________________________________________________________

def LoadFromMat(path, image_number, OutOnePolarity=False, verbose=0):
    '''
            Load Events from a .mat file. Only the events contained in ListPolarities are kept:
            INPUT
                + path : a string which is the path of the .mat file (ex : './data_cache/alphabet_ExtractedStabilized.mat')
                + image_number : list with all the numbers of image to load
    '''
    from scipy import io
    obj = io.loadmat(path)
    ROI = obj['ROI'][0]

    if type(image_number) is int:
        image_number = [image_number]
    elif type(image_number) is not list:
        raise TypeError(
                    'the type of argument image_number should be int or list')
    if verbose > 0:
        print("loading images {0}".format(image_number))
    Total_size = 0
    for idx, each_image in enumerate(image_number):
        image = ROI[each_image][0, 0]
        Total_size += image[1].shape[1]

    address = np.zeros((Total_size, 2)).astype(int)
    time = np.zeros((Total_size))
    polarity = np.zeros((Total_size))
    first_idx = 0

    for idx, each_image in enumerate(image_number):
        image = ROI[each_image][0, 0]
        last_idx = first_idx + image[0].shape[1]
        address[first_idx:last_idx, 0] = (image[1] - 1).astype(int)
        address[first_idx:last_idx, 1] = (image[0] - 1).astype(int)
        time[first_idx:last_idx] = (image[3] * 1e-6)
        polarity[first_idx:last_idx] = image[2].astype(int)
        first_idx = last_idx

    polarity[polarity.T == -1] = 0
    polarity = polarity.astype(int)
            # Filter only the wanted polarity
    ListPolarities = np.unique(polarity)
    if OutOnePolarity == True:
        polarity = np.zeros_like(polarity)
        ListPolarities = [0]

    return address, time, polarity, ListPolarities
