import numpy as np
import matplotlib.pyplot as plt
from Layer import layer
from TimeSurface import TimeSurface
from Stats import stats
from tqdm import tqdm
import tonic
import os
import pickle

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
                        begin = 0, #first event indice taken into account
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
        if to_record:
            self.stats = [[]]*nblay
        self.TS = [[]]*nblay
        self.L = [[]]*nblay
        for lay in range(nblay):
            if lay == 0:
                self.TS[lay] = TimeSurface(R, tau, camsize, nbpolcam, pola, filt, sigma)
                self.L[lay] = layer(R, nbclust, pola, nbpolcam, homeo, homparam, homeinv, algo, hout, krnlinit, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust, camsize)
            else:
                self.TS[lay] = TimeSurface(R*(K_R**lay), tau*(K_tau**lay), camsize, nbclust*(K_clust**(lay-1)), pola, filt, sigma)
                self.L[lay] = layer(R*(K_R**lay), nbclust*(K_clust**lay), pola, nbclust*(K_clust**(lay-1)), homeo, homparam, homeinv, algo, hout, krnlinit, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust*(K_clust**lay), camsize)

##___________________________________________________________________________________________

    def load(self, dataset, trainset=True, jitonic=[None,None]):
        self.jitonic = jitonic
        if jitonic[1] is not None:
            print(f'spatial jitter -> var = {jitonic[1]}')
            transform = tonic.transforms.Compose([tonic.transforms.SpatialJitter(variance_x=jitonic[1], variance_y=jitonic[1], sigma_x_y=0, integer_coordinates=True, clip_outliers=True)])

        if jitonic[0] is not None:
            print(f'time jitter -> var = {jitonic[0]}')
            transform = tonic.transforms.Compose([tonic.transforms.TimeJitter(variance=jitonic[0], integer_timestamps=False, clip_negative=True)])

        if jitonic == [None,None]:
            print('no jitter')
            transform = None

        download=False
        if dataset == 'nmnist':
            path = '../Data/'
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
            eventset = tonic.datasets.POKERDVS(save_to='../Data/',
                                train=trainset,
                                transform=transform)
        elif dataset == 'gesture':
            eventset = tonic.datasets.DVSGesture(save_to='../Data/',
                                train=trainset,
                                transform=transform)
        elif dataset == 'cars':
            eventset = tonic.datasets.NCARS(save_to='../Data/',
                                train=trainset,
                                transform=transform)
        elif dataset == 'ncaltech':
            eventset = tonic.datasets.NCALTECH101(save_to='../Data/',
                                transform=transform)
        else: print('incorrect dataset')

        loader = tonic.datasets.DataLoader(eventset, shuffle=True)

        if eventset.sensor_size!=self.TS[0].camsize:
            print('sensor formatting...')
            for i in range(1,len(self.TS)):
                self.TS[i].camsize = eventset.sensor_size
                self.TS[i].spatpmat = np.zeros((self.L[i-1].kernel.shape[1],eventset.sensor_size[0]+1,eventset.sensor_size[1]+1))
                self.stats[i].actmap = np.zeros((self.L[i-1].kernel.shape[1],eventset.sensor_size[0]+1,eventset.sensor_size[1]+1))
            self.TS[0].spatpmat = np.zeros((2,eventset.sensor_size[0]+1,eventset.sensor_size[1]+1))
            self.stats[0].actmap = np.zeros((2,eventset.sensor_size[0]+1,eventset.sensor_size[1]+1))
        return loader, eventset.ordering, len(eventset.classes)


    def learning1by1(self, nb_digit=2, dataset='nmnist', diginit=True, filtering=None, jitonic=[None,None]):

        self.onbon = True
        model = self.load_model(dataset)
        if model:
            return model
        else:

            loader, ordering, nbclass = self.load(dataset, jitonic=jitonic)
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
                                lay += 1
                            else:
                                lay = n+1
                pbar.close()
            for l in range(len(self.L)):
                self.stats[l].histo = self.L[l].cumhisto.copy()

            self.save_model(dataset)

            return self


    def learningall(self, nb_digit=2, dataset='nmnist', diginit=True, jitonic=[None,None]):

        self.onbon = False
        model = self.load_model(dataset)
        if model:
            return model
        else:

            loader, ordering, nbclass = self.load(dataset, jitonic=jitonic)

            pbar = tqdm(total=nb_digit*nbclass)

            nbloadz = np.zeros([nbclass])
            while np.sum(nbloadz)<nb_digit*nbclass:
            #for idig in range(nb_digit):
                if diginit:
                    for i in range(len(self.L)):
                        self.TS[i].spatpmat[:] = 0
                        self.TS[i].iev = 0
                events, target = next(iter(loader))
                if nbloadz[target]<nb_digit:
                    nbloadz[target]+=1
                    pbar.update(1)
                    for iev in range(events.shape[1]):
                        self.run(events[0][iev][ordering.find("x")].item(), \
                                 events[0][iev][ordering.find("y")].item(), \
                                 events[0][iev][ordering.find("t")].item(), \
                                 events[0][iev][ordering.find("p")].item(), \
                                 learn=True, to_record=True)
            pbar.close()
            for l in range(len(self.L)):
                self.stats[l].histo = self.L[l].cumhisto.copy()

            self.save_model(dataset)

            return self


    def running(self, homeotest=False, train=True, LR=False, nb_digit=500, jitonic=[None,None], dataset='nmnist', to_record=False):

        output, loaded = self.load_output(dataset, homeotest, nb_digit, train, jitonic, LR)
        if loaded:
            return output
        else:
            loader, ordering, nbclass = self.load(dataset, trainset=train, jitonic=jitonic)

            homeomod = self.L[0].homeo

            for i in range(len(self.L)):
                self.L[i].homeo=homeotest

            pbar = tqdm(total=nb_digit)
            timout = []
            xout = []
            yout = []
            polout = []
            labout = []
            if train:
                labelmap = np.zeros([nbclass, len(self.L[-1].cumhisto)])
                labelcount = np.zeros(nbclass)
            else:
                labelmap = []

            x_index = ordering.find("x")
            y_index = ordering.find("y")
            t_index = ordering.find("t")
            p_index = ordering.find("p")

            for idig in range(nb_digit):
                for i in range(len(self.L)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
                    self.L[i].cumhisto[:] = 1
                    #self.stats[i].actmap[:] = 0
                pbar.update(1)
                events, target = next(iter(loader))
                for iev in range(events.shape[1]):
                    out, activout = self.run(events[0][iev][x_index].item(), \
                                            events[0][iev][y_index].item(), \
                                            events[0][iev][t_index].item(), \
                                            events[0][iev][p_index].item(), \
                                            to_record=to_record)
                    if LR and activout:
                        xout.append(out[x_index])
                        yout.append(out[y_index])
                        timout.append(out[t_index])
                        polout.append(out[p_index])
                        labout.append(target.item())

                if not LR:
                    if train:
                        labelmap[target.item(),:] += self.L[-1].cumhisto/np.sum(self.L[-1].cumhisto)
                        labelcount[target.item()] += 1
                    else:
                        data = (target.item(),self.L[-1].cumhisto.copy())
                        labelmap.append(data)
                    eventsout = []
            if LR:
                camsize = self.TS[-1].camsize
                nbpola = self.L[-1].kernel.shape[1]
                eventsout = [xout,yout,timout,polout,labout,camsize,nbpola]
            pbar.close()

            for i in range(len(self.L)):
                self.L[i].homeo=homeomod

            if LR:
                self.save_output(eventsout, homeotest, dataset, nb=nb_digit, train=train, jitonic=jitonic, LR=True)
                output = eventsout
            else:
                if train:
                    for i in range(len(labelcount)):
                        labelmap[i,:] /= labelcount[i]

                self.save_output(labelmap, homeotest, dataset, nb=nb_digit, train=train, jitonic=jitonic, LR=False)
                output = labelmap
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
        #if self.TS[0].iev//500==0:
            #self.TS[0].plote()
        return out, activout

    def get_fname(self):
        timestr = self.date
        algo = self.L[0].algo
        arch = [self.L[i].kernel.shape[1] for i in range(len(self.L))]
        R = [self.L[i].R for i in range(len(self.L))]
        tau = [self.TS[i].tau for i in range(len(self.TS))]
        homeo = self.L[0].homeo
        homparam = self.L[0].homparam
        krnlinit = self.L[0].krnlinit
        sigma = self.TS[0].sigma
        onebyone = self.onbon
        f_name = f'{timestr}_{algo}_{krnlinit}_{sigma}_{homeo}_{homparam}_{arch}_{tau}_{R}_{onebyone}'
        self.name = f_name
        #print(self.name)
        return f_name

    def save_model(self, dataset):
        if dataset=='nmnist':
            path = f'../Records/EXP_03_NMNIST/models/'
        else: print('define a path for this dataset')
        if not os.path.exists(path):
            os.makedirs(path)
        f_name = path+self.get_fname()+'.pkl'
        with open(f_name, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def load_model(self, dataset):
        model = []
        if dataset=='nmnist':
            path = f'../Records/EXP_03_NMNIST/models/'
        else: print('define a path for this dataset')
        f_name = path+self.get_fname()+'.pkl'
        if not os.path.isfile(f_name):
            return model
        else:
            with open(f_name, 'rb') as file:
                model = pickle.load(file)
        return model

    def save_output(self, evout, homeo, dataset, nb, train, jitonic, LR):
        if dataset=='nmnist':
            dataset = 'EXP_03_NMNIST'
        else: print('define a path for this dataset')
        if train:
            path = f'../Records/{dataset}/train/'
        else:
            path = f'../Records/{dataset}/test/'
        if not os.path.exists(path):
            os.makedirs(path)
        f_name = path+self.name+f'_{nb}_{jitonic}'
        if homeo:
            f_name = f_name+'_homeo'
        if not LR:
            f_name = f_name+'_histo'
        f_name = f_name +'.pkl'
        with open(f_name, 'wb') as file:
            pickle.dump(evout, file, pickle.HIGHEST_PROTOCOL)

    def load_output(self, dataset, homeo, nb, train, jitonic, LR):
        loaded = False
        output = []
        if dataset=='nmnist':
            dataset = 'EXP_03_NMNIST'
        else: print('define a path for this dataset')
        if train:
            path = f'../Records/{dataset}/train/'
        else:
            path = f'../Records/{dataset}/test/'
        f_name = path+self.name+f'_{nb}_{jitonic}'
        if homeo:
            f_name = f_name+'_homeo'
        if not LR:
            f_name = f_name+'_histo'
        f_name = f_name +'.pkl'
        print(f_name)
        if os.path.isfile(f_name):
            with open(f_name, 'rb') as file:
                output = pickle.load(file)
            loaded = True
        return output, loaded


##___________REPRODUCING RESULTS FROM LAGORCE 2017___________________________________________

    def learninglagorce(self, nb_cycle=3, dataset='simple', diginit=True, filtering=None):


        #___________ SPECIAL CASE OF SIMPLE_ALPHABET DATASET _________________

        path = "../Data/alphabet_ExtractedStabilized.mat"

        image_list = [1, 32, 19, 22, 29]
        image_list += image_list
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


    def traininglagorce(self, nb_digit=None, dataset='simple', to_record=True):

        if dataset == 'simple':
            path = "../Data/alphabet_ExtractedStabilized.mat"
            image_list=list(np.arange(0, 36))
            address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)
            with open('../Data/alphabet_label.pkl', 'rb') as file:
                label_list = pickle.load(file)
            label = label_list[:36]
        else:
            print('not ready yet')
            event = []

        learn=False
        output = []
        count = 0
        count2 = 0
        nbevent = int(time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        labelmap = []
        for i in range(len(self.L)):
            self.TS[i].spatpmat[:] = 0
            self.TS[i].iev = 0
            self.L[i].cumhisto[:] = 1

        while count<nbevent:
            pbar.update(1)
            self.run(address[count,0],address[count,1],time[count],polarity[count], learn, to_record)
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
        return labelmap

    def testinglagorce(self, trainmap, nb_digit=None, dataset='simple', to_record=True):

        if dataset == 'simple':
            path = "../Data/alphabet_ExtractedStabilized.mat"
            image_list=list(np.arange(36, 76))
            address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)
            with open('../Data/alphabet_label.pkl', 'rb') as file:
                label_list = pickle.load(file)
            label = label_list[36:76]
        else:
            print('not ready yet')
            event = []

        learn = False
        output = []
        count = 0
        count2 = 0
        nbevent = int(time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        labelmap = []
        for i in range(len(self.L)):
            self.TS[i].spatpmat[:] = 0
            self.TS[i].iev = 0
            self.L[i].cumhisto[:] = 1
        while count<nbevent:
            pbar.update(1)
            self.run(address[count,0],address[count,1],time[count],polarity[count], learn, to_record)
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

        score1=accuracy_lagorce(trainmap,labelmap,'bhatta')
        score2=accuracy_lagorce(trainmap,labelmap,'eucli')
        score3=accuracy_lagorce(trainmap,labelmap,'norm')
        print('bhatta:'+str(score1*100)+'% - '+'eucli:'+str(score2*100)+'% - '+'norm:'+str(score3*100)+'%')

        return labelmap, [score1,score2,score3]

##___________________PLOTTING________________________________________________________________

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
                        begin = 0, #first event indice taken into account
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
                        begin = begin,
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

        loader, ordering, nbclass = self.load(dataset)
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


##__________________TOOLS____________________________________________________________________
##___________________________________________________________________________________________

def EuclidianNorm(hist1,hist2):
    return np.linalg.norm(hist1-hist2)

def NormalizedNorm(hist1,hist2):
    return np.linalg.norm(hist1-hist2)/(np.linalg.norm(hist1)*np.linalg.norm(hist2))

def BattachaNorm(hist1, hist2):
    return -np.log(np.sum(np.sqrt(hist1*hist2)))

def KullbackLeibler(hist_test, hist_train):
    return np.sum(hist_test*np.log(hist_test/hist_train))

def JensenShannon(hist1, hist2):
    hist3 = (hist1+hist2)*0.5
    return (KullbackLeibler(hist1,hist3)+KullbackLeibler(hist2,hist3))*0.5

def accuracy(trainmap,testmap,measure):
    accuracy=0
    total = 0
    for i in range(len(testmap)):
        dist = np.zeros([trainmap.shape[0]])
        histest = testmap[i][1]/np.sum(testmap[i][1])
        for k in range(trainmap.shape[0]):
            histrain = trainmap[k,:]/np.sum(trainmap[k,:])
            if measure=='bhatta':
                dist[k] = BattachaNorm(histest,histrain)
            elif measure=='eucli':
                dist[k] = EuclidianNorm(histest,histrain)
            elif measure=='norm':
                dist[k] = NormalizedNorm(histest,histrain)
            elif measure == 'KL':
                dist[k] = KullbackLeibler(histest,histrain)
            elif measure == 'JS':
                dist[k] = JensenShannon(histest,histrain)
        if testmap[i][0]== np.argmin(dist):
            accuracy+=1
        total+=1
    return accuracy/total

def accuracy_lagorce(trainmap,testmap,measure):
    accuracy=0
    total = 0
    for i in range(len(testmap)):
        dist = np.zeros(len(trainmap))
        # print(i, testmap[i][1])
        # print(i, np.sum(testmap[i][1][1]))
        # print('doh', testmap[i][1][1])
        histest = testmap[i][1]/np.sum(testmap[i][1])
        for k in range(len(trainmap)):
            histrain = trainmap[k][1]/np.sum(trainmap[k][1])
            if measure=='bhatta':
                dist[k] = BattachaNorm(histest,histrain)
            elif measure=='eucli':
                dist[k] = EuclidianNorm(histest,histrain)
            elif measure=='norm':
                dist[k] = NormalizedNorm(histest,histrain)
            elif measure == 'KL':
                dist[k] = KullbackLeibler(histest,histrain)
            elif measure == 'JS':
                dist[k] = JensenShannon(histest,histrain)
        if testmap[i][0]==trainmap[np.argmin(dist)][0]:
            accuracy+=1
        total+=1
    return accuracy/total

#def knn(trainmap,testmap,k):
#    from sklearn.neighbors import KNeighborsClassifier

#    X_train = np.array([trainmap[i][1]/np.sum(trainmap[i][1]) for i in range(len(trainmap))]).reshape(len(trainmap),len(trainmap[0][1]))
#    knn = KNeighborsClassifier(n_neighbors=k)
#    knn.fit(X_train,[trainmap[i][0] for i in range(len(trainmap))])
#    accuracy = 0
#    for i in range(len(testmap)):
#        if knn.predict([testmap[i][1]/np.sum(testmap[i][1])])==testmap[i][0]:
#            accuracy += 1
#    return accuracy/len(testmap)

def histoscore(trainmap,testmap,k=6, verbose = True):
    bhat_score = accuracy(trainmap, testmap, 'bhatta')
    norm_score = accuracy(trainmap, testmap, 'norm')
    eucl_score = accuracy(trainmap, testmap, 'eucli')
    KL_score = accuracy(trainmap,testmap,'KL')
    JS_score = accuracy(trainmap,testmap,'JS')
    #knn_score = knn(trainmap,testmap,k)
    #k2 = k//2
    #k2nn_score = knn(trainmap,testmap,k2)
    if verbose:
        print(47*'-'+'SCORES'+47*'-')
        print(f'Classification scores with HOTS measures: bhatta = {bhat_score*100}% - eucli = {eucl_score*100}% - norm = {norm_score*100}%')
        #print(f'Classification scores with kNN: {k2}-NN = {k2nn_score*100}% - {k}-NN = {knn_score*100}%')
        print(f'Classification scores with entropy: Kullback-Leibler = {KL_score*100}% - Jensen-Shannon = {JS_score*100}%')
        print(100*'-')
    return JS_score

def spatial_jitter(
    x_index, y_index,
    sensor_size,
    variance_x=1,
    variance_y=1,
    sigma_x_y=0,
    ):
    """Changes position for each pixel by drawing samples from a multivariate
    Gaussian distribution with the following properties:
        mean = [x,y]
        covariance matrix = [[variance_x, sigma_x_y],[sigma_x_y, variance_y]]
    Jittered events that lie outside the focal plane will be dropped if clip_outliers is True.
    Args:
        events: ndarray of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess through
                  guess_event_ordering_numpy. This function requires 'x'
                  and 'y' to be in the ordering
        variance_x: squared sigma value for the distribution in the x direction
        variance_y: squared sigma value for the distribution in the y direction
        sigma_x_y: changes skewness of distribution, only change if you want shifts along diagonal axis.
        integer_coordinates: when True, shifted x and y values will be integer coordinates
        clip_outliers: when True, events that have been jittered outside the focal plane will be dropped.
    Returns:
        spatially jittered set of events.
    """

    shifts = np.random.multivariate_normal(
        [0, 0], [[variance_x, sigma_x_y], [sigma_x_y, variance_y]]
    )

    shifts = shifts.round()

    xs = (x_index + shifts[0])
    ys = (y_index + shifts[1])

    if xs<0: xs=0
    elif xs>sensor_size[0]-1: xs = sensor_size[0]-1
    if ys<0: ys=0
    elif ys>sensor_size[1]-1: ys = sensor_size[1]-1

    return xs, ys


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
