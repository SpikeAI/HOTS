import numpy as np
from mix_Layer import *
from mix_TimeSurface import *
from mix_Stats import *
from Event import Event
from Tools import LoadObject
from tqdm import tqdm
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
                        tau = 10, #timestamp en microsec/
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
        tau *= 1000 # to enter tau in ms
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

    # faire un merge de run et train?     
    def run(self, x, y, t, p, to_record=False):
        lay = 0
        learn = False
        while lay<len(self.TS):
            timesurf, activ = self.TS[lay].addevent(x, y, t, p)
            if activ==True:
                p, dicprev = self.L[lay].run(timesurf, learn)
                if to_record==True:
                    self.stats[lay].update(p, self.L[lay].kernel, dicprev, timesurf)
                    self.stats[lay].actmap[int(np.argmax(p)),self.TS[lay].x,self.TS[lay].y]=1
                lay+=1
            else:
                lay = len(self.TS)
        out = [x,y,t,np.argmax(p)]
        return out, activ
           
        
    def train(self, x, y, t, p):
        lay = 0
        learn = True
        while lay<len(self.TS):
            timesurf, activ = self.TS[lay].addevent(x, y, t, p)
            if activ==True:
                p, dicprev = self.L[lay].run(timesurf, learn)
                if hasattr(self, 'stats'):
                    self.stats[lay].update(p, self.L[lay].kernel, dicprev, timesurf)
                    #self.stats[lay].actmap[int(np.argmax(p)),self.TS[lay].x,self.TS[lay].y]=1
                lay+=1
            else:
                lay = len(self.TS)
                
                
    def learn(self, nb_digit=None, dataset='simple', diginit=False):
        
        #___________ SPECIAL CASE OF SIMPLE_ALPHABET DATASET _________________
        if dataset == 'simple':
            event = Event(ImageSize=(32, 32))
            event.LoadFromMat("../Data/alphabet_ExtractedStabilized.mat", image_number=[1, 32, 19, 22, 29, 1, 32, 19, 22, 29, 1, 32, 19, 22, 29])
        #___________ SPECIAL CASE OF SIMPLE_ALPHABET DATASET _________________
        else: 
            event = []
        
        count = 0
        nbevent = int(event.time.shape[0])
        pbar = tqdm(total=nbevent)
        while count<nbevent:
            pbar.update(1)
            if diginit==True and event.time[count]<event.time[count-1]:
                for i in range(len(self.TS)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
            self.train(event.address[count,1],event.address[count,0],event.time[count],event.polarity[count])
            count += 1
        pbar.close()
        
        
    def labelhisto(self, nb_digit=None, dataset='simple', diginit=False, to_record=False):
        
        if dataset == 'simple':
            event = Event(ImageSize=(32, 32))
            event.LoadFromMat("../Data/alphabet_ExtractedStabilized.mat", image_number=list(
                                                                                            np.arange(0, 36)))
            label_list = LoadObject('../Data/alphabet_label.pkl')
        else:
            event = []
           
        output = []
        count = 0
        count2 = 0
        nbevent = int(event.time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        digit = label_list[idx][0]
        labmap = []
        while count<nbevent:
            pbar.update(1)
            if diginit==True and event.time[count]<event.time[count-1]:
                for i in range(len(self.TS)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
            self.run(event.address[count,1],event.address[count,0],event.time[count], event.polarity[count], to_record) 
            if count2==label_list[idx][1]:
                data = (digit,self.L[-1].cumhisto)
                labmap.append(data) 
                
                idx += 1
                digit = label_list[idx][0]
                count2=-1
                
            count += 1
            count2 += 1
            
        pbar.close()
        return labmap
 
    def testhisto(self, nb_digit=None, dataset='simple', diginit=False, to_record=False):
        
        if dataset == 'simple':
            event = Event(ImageSize=(32, 32))
            event.LoadFromMat("../Data/alphabet_ExtractedStabilized.mat", image_number=list(
                                                                                            np.arange(36, 76)))
            label_list = LoadObject('../Data/alphabet_label.pkl')
            label = []
            for i in range(40):
                label.append(label_list[i][0])
        else:
            event = []
            
        output = []
        count = 0
        count2 = 0
        nbevent = int(event.time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        digit = label_list[idx][0]
        labmap = []
        while count<nbevent:
            pbar.update(1)
            if diginit==True and event.time[count]<event.time[count-1]:
                for i in range(len(self.TS)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
            self.run(event.address[count,1],event.address[count,0],event.time[count], event.polarity[count], to_record) 
            if count2==label_list[idx][1]:
                data = (digit,self.L[-1].cumhisto)
                labmap.append(data) 
                
                idx += 1
                digit = label_list[idx][0]
                count2=-1
                
            count += 1
            count2 += 1
            
        pbar.close()
        return labmap
            
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
            plt.bar(np.arange(N[i]), self.L[i].cumhisto, width=1, align='edge', ec="k")
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