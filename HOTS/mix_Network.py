import numpy as np
from mix_Layer import *
from mix_TimeSurface import *
from mix_Stats import *
from Event import Event
from Tools import LoadObject
from tqdm import tqdm as tqdm
from sklearn.neighbors import KNeighborsClassifier
import tonic
#from threading import Thread, Rlock

#loco = Rlock()

class network(object):
    """network is an object composed of nblay layers (dico in Layer.py) and the same numbers of TS (TimeSurface.py) as input of the different layers. It processes the different """

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
                        camsize = [34, 34],
                        begin = 0, #first event indice taken into account
                        # functional parameters of the network
                        algo = 'lagorce', # among ['lagorce', 'maro', 'mpursuit']
                        krnlinit = 'rdn',
                        hout = False, #works only with mpursuit
                        homeo = False,
                        pola = True,
                        to_record = True,
                        filt = 2
                ):
        tau *= 1e-3 # to enter tau in ms
        if to_record == True:
            self.stats = [[]]*nblay
        self.TS = [[]]*nblay
        self.L = [[]]*nblay
        for lay in range(nblay):
            if lay == 0:
                self.TS[lay] = TimeSurface(R, tau, camsize, nbpolcam, pola, filt)
                self.L[lay] = layer(R, nbclust, pola, nbpolcam, camsize, homeo, algo, hout, krnlinit, to_record)
                if to_record == True:
                    self.stats[lay] = stats(nbclust, camsize)
            else:
                self.TS[lay] = TimeSurface(R*(K_R**lay), tau*(K_tau**lay), camsize, nbclust*(K_clust**(lay-1)), pola, filt)
                self.L[lay] = layer(R*(K_R**lay), nbclust*(K_clust**lay), pola, nbclust*(K_clust**(lay-1)), camsize, homeo, algo, hout, krnlinit, to_record)
                if to_record == True:
                    self.stats[lay] = stats(nbclust*(K_clust**lay), camsize)
        self.L[lay].out = 1
        
##____________________________________________________________________________________

    def learning1by1(self, nb_digit=15, dataset='nmnist', diginit=True, filtering=None):

        if dataset == 'nmnist':
            learningset = tonic.datasets.NMNIST(save_to='../Data/',
                                train=False,
                                transform=None)
            loader = tonic.datasets.DataLoader(learningset, shuffle=True)
            
        eventslist = [next(iter(loader))[0] for i in range(nb_digit)]
        for n in range(len(self.L)):
            pbar = tqdm(total=nb_digit)
            for idig in range(nb_digit):
                pbar.update(1)
                events = eventslist[idig]
                if diginit==True:
                    for l in range(n+1):
                        self.TS[l].spatpmat[:] = 0
                        self.TS[l].iev = 0
                for iev in range(events.shape[1]):
                    x,y,t,p = events[0][iev][0],events[0][iev][1],events[0][iev][2]*1e-6,events[0][iev][3]
                    lay=0
                    while lay < n+1:
                        if lay==n:
                            learn=True
                        else:
                            learn=False
                        timesurf, activ = self.TS[lay].addevent(x, y, t, p)
                        if lay==0 or filtering=='all':
                            activ2=activ
                        if activ2==True and np.sum(timesurf)>0:
                        #if activ==True:
                            p, dist = self.L[lay].run(timesurf, learn)
                            if learn==True:
                                self.stats[lay].update(p, self.L[lay].kernel, timesurf, dist)
                            lay += 1
                        else:
                            lay = n+1           
            pbar.close()
        for l in range(len(self.L)):
            self.stats[l].histo = self.L[l].cumhisto.copy()
        return loader
    
    
    def learningall(self, nb_digit=15, dataset='nmnist', diginit=True, filtering=None):

        if dataset == 'nmnist':
            learningset = tonic.datasets.NMNIST(save_to='../Data/',
                                train=False,
                                transform=None)
            loader = tonic.datasets.DataLoader(learningset, shuffle=True)
            
        pbar = tqdm(total=nb_digit)
        for idig in range(nb_digit):
            if diginit==True:
                for i in range(len(self.L)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
            pbar.update(1)
            events, target = next(iter(loader))
            for iev in range(events.shape[1]):
                self.run(events[0][iev][0],events[0][iev][1],events[0][iev][2]*1e-6, events[0][iev][3], learn=True, to_record=True)
        pbar.close()
        for l in range(len(self.L)):
            self.stats[l].histo = self.L[l].cumhisto.copy()
        return loader
    
    def training(self, loader, nb_digit=40):
        pbar = tqdm(total=nb_digit)
        labelmap = []
        for idig in range(nb_digit):
            for i in range(len(self.L)):
                self.TS[i].spatpmat[:] = 0
                self.TS[i].iev = 0
                self.L[i].cumhisto[:] = 0
            pbar.update(1)
            events, target = next(iter(loader))
            for iev in range(events.shape[1]):
                self.run(events[0][iev][0],events[0][iev][1],events[0][iev][2]*1e-6, events[0][iev][3], to_record=True)
            data = (target,self.L[-1].cumhisto.copy())
            labelmap.append(data)
        pbar.close()
        return labelmap, loader
    
 
    def testing(self, loader, trainmap, nb_digit=40):
        
        testmap, loader = self.training(loader)
        score1=accuracy(trainmap,testmap,'bhatta')
        score2=accuracy(trainmap,testmap,'eucli')
        score3=accuracy(trainmap,testmap,'norm')
        print('bhatta:'+str(score1*100)+'% - '+'eucli:'+str(score2*100)+'% - '+'norm:'+str(score3*100)+'%')
        return testmap, loader

    def run(self, x, y, t, p, learn=False, to_record=False):
        lay = 0
        while lay<len(self.TS):
            timesurf, activ = self.TS[lay].addevent(x, y, t, p)
            if activ==True:
                p, dist = self.L[lay].run(timesurf, learn)
                if to_record==True:
                    self.stats[lay].update(p, self.L[lay].kernel, timesurf, dist)
                    #self.stats[lay].actmap[int(np.argmax(p)),self.TS[lay].x,self.TS[lay].y]=1
                lay+=1
            else:
                lay = len(self.TS)
        out = [x,y,t,np.argmax(p)]
        return out, activ


##___________REPRODUCING RESULTS FROM LAGORCE 2017________________________________________

    def learninglagorce(self, nb_cycle=3, dataset='simple', diginit=True, filtering=None):

        #___________ SPECIAL CASE OF SIMPLE_ALPHABET DATASET _________________
        if dataset == 'simple':
            event = Event(ImageSize=(32, 32))
            digit_numbers = [1,32,19,22,29]
            diglist = []
            for nbd in range(nb_cycle):
                diglist+=digit_numbers
            event.LoadFromMat("../Data/alphabet_ExtractedStabilized.mat", image_number=diglist)
        #___________ SPECIAL CASE OF SIMPLE_ALPHABET DATASET _________________
        
        nbevent = int(event.time.shape[0])    
        for n in range(len(self.L)):
            count = 0
            pbar = tqdm(total=nbevent)
            while count<nbevent:
                pbar.update(1)
                x,y,t,p = event.address[count,0],event.address[count,1], event.time[count],event.polarity[count]
                if diginit==True and event.time[count]<event.time[count-1]:
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
                    if activ2==True and np.sum(timesurf)>0:
                        p, dist = self.L[lay].run(timesurf, learn)
                        if learn==True:
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
            event = Event(ImageSize=(32, 32))
            event.LoadFromMat("../Data/alphabet_ExtractedStabilized.mat", image_number=list(
                                                                                            np.arange(0, 36)))
            label_list = LoadObject('../Data/alphabet_label.pkl')
            label = label_list[:36]
        else:
            print('not ready yet')
            event = []

        learn=False
        output = []
        count = 0
        count2 = 0
        nbevent = int(event.time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        labelmap = []
        for i in range(len(self.L)):
            self.TS[i].spatpmat[:] = 0
            self.TS[i].iev = 0
            self.L[i].cumhisto[:] = 0
            
        while count<nbevent:
            pbar.update(1)
            self.run(event.address[count,0],event.address[count,1],event.time[count], event.polarity[count], learn, to_record)
            if count2==label[idx][1]:
                data = (label[idx][0],self.L[-1].cumhisto.copy())
                labelmap.append(data)
                for i in range(len(self.L)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
                    self.L[i].cumhisto[:] = 0
                idx += 1
                count2=-1
            count += 1
            count2 += 1
        pbar.close()
        return labelmap

    def testinglagorce(self, trainmap, nb_digit=None, dataset='simple', to_record=True):

        if dataset == 'simple':
            event = Event(ImageSize=(32, 32))
            event.LoadFromMat("../Data/alphabet_ExtractedStabilized.mat", image_number=list(
                                                                                            np.arange(36, 76)))
            label_list = LoadObject('../Data/alphabet_label.pkl')
            label = label_list[36:76]
        else:
            print('not ready yet')
            event = []

        learn = False
        output = []
        count = 0
        count2 = 0
        nbevent = int(event.time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        labelmap = []
        for i in range(len(self.L)):
            self.TS[i].spatpmat[:] = 0
            self.TS[i].iev = 0
            self.L[i].cumhisto[:] = 0
        while count<nbevent:
            pbar.update(1)
            self.run(event.address[count,0],event.address[count,1],event.time[count],event.polarity[count], learn, to_record)
            if count2==label[idx][1]:
                data = (label[idx][0],self.L[-1].cumhisto.copy())
                labelmap.append(data)
                for i in range(len(self.L)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
                    self.L[i].cumhisto[:] = 0
                idx += 1
                count2=-1
            count += 1
            count2 += 1

        pbar.close()
        
        score1=accuracy(trainmap,labelmap,'bhatta')
        score2=accuracy(trainmap,labelmap,'eucli')
        score3=accuracy(trainmap,labelmap,'norm')
        print('bhatta:'+str(score1*100)+'% - '+'eucli:'+str(score2*100)+'% - '+'norm:'+str(score3*100)+'%')
        
        return labelmap, [score1,score2,score3]

##___________________PLOTTING_________________________________________________________

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
        if self.L[-1].homeo==True:
            fig.suptitle('Activation histograms and associated features with homeostasis', size=20, y=0.95)
        else:
            fig.suptitle('Activation histograms and associated features without homeostasis', size=20, y=0.95)

        for i in range(len(self.L)):
            ax = fig.add_subplot(gs[:hisiz, int(np.sum(N[:i]))+1*i:int(np.sum(N[:i+1]))+i*1])
            plt.bar(np.arange(N[i]), self.stats[i].histo/np.sum(self.stats[i].histo), width=1, align='edge', ec="k")
            ax.set_xticks(())
            if i>0:
                ax.set_yticks(())
            ax.set_title('Layer '+str(i+1), fontsize=16)
            plt.xlim([0,N[i]])
            plt.ylim([0,yhis])

        #f3_ax1.set_title('gs[0, :]')
            for k in range(N[i]):
                for j in range(P[i]):
                    if j>maxpol-1:
                        pass
                    else:
                        axi = fig.add_subplot(gs[j+hisiz,k+1*i+int(np.sum(N[:i]))])
                        krnl = self.L[i].kernel[j*R2[i]:(j+1)*R2[i],k].reshape((int(np.sqrt(R2[i])), int(np.sqrt(R2[i]))))
                        axi.imshow(krnl, cmap=plt.cm.plasma, interpolation='nearest')
                        axi.set_xticks(())
                        axi.set_yticks(())

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

                    
##__________________TOOLS_____________________________________________________________________
    
def EuclidianNorm(hist1,hist2):
    return np.linalg.norm(hist1-hist2)

def NormalizedNorm(hist1,hist2):
    hist1/=np.sum(hist1)
    hist2/=np.sum(hist2)
    return np.linalg.norm(hist1-hist2)/(np.linalg.norm(hist1)*np.linalg.norm(hist2))

def BattachaNorm(hist1, hist2):
    hist1/=np.sum(hist1)
    hist2/=np.sum(hist2)
    return -np.log(np.sum(np.sqrt(hist1*hist2)))

def accuracy(trainmap,testmap,measure):
    accuracy=0
    total = 0
    for i in range(len(testmap)):
        dist = np.zeros([len(trainmap)])
        for k in range(len(trainmap)):
            if measure=='bhatta':
                dist[k] = BattachaNorm(testmap[i][1],trainmap[k][1])
            elif measure=='eucli':
                dist[k] = EuclidianNorm(testmap[i][1],trainmap[k][1])
            elif measure=='norm':
                dist[k] = NormalizedNorm(testmap[i][1],trainmap[k][1])
        if testmap[i][0]==trainmap[np.argmin(dist)][0]:
            accuracy+=1
        total+=1
    return accuracy/total

def knn(trainmap,testmap,k=6):
    X_train = np.array([trainmap[i][1] for i in range(len(trainmap))]).reshape(len(trainmap),len(trainmap[0][1]))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,[trainmap[i][0] for i in range(len(trainmap))])
    accuracy = 0
    for i in range(len(testmap)):
        if knn.predict([testmap[i][1]]).item()==testmap[i][0].item():
            accuracy += 1
    print(accuracy/len(testmap)) 