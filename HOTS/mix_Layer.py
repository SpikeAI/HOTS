import numpy as np
from scipy import signal
import copy
from mix_TimeSurface import *
import matplotlib.pyplot as plt
from IPython import display
import time

class layer(object):
    """layer aims at learning an online sparse dictionary to describe an input stimulus (here a time-surface). Given the input size (square 2R+1 matrix) it will create a (2R+1)^2 by N dictionary matrix: self.dic . compute h (output), which is the sparse map: the coefficients by which one multiply the dictionary (basis) to get the reconstructed signal.
    """
    
    def __init__(self, R, N_clust, pola, nbpola, homeo, homeinv, algo, hout, krnlinit, to_record):
        self.out = False          # boolean to indicate if this layer is the last one of the network
        self.hout = hout
        self.to_record = to_record
        self.R = R
        self.algo = algo
        self.homeo = homeo        # boolean indicating if homeostasis is used or not
        self.homeinv = homeinv
        self.nbtrain = 0          # number of TS sent in the layer
        self.ratihom = 0.01/(self.nbtrain+1)
        self.krnlinit = krnlinit
        
        if not pola:
            self.kernel = np.random.rand((2*R+1)**2, N_clust)
        else:
            self.kernel = np.random.rand(nbpola*(2*R+1)**2, N_clust)
        some = np.sqrt(np.sum(self.kernel**2, axis=0))
        self.kernel = self.kernel/some[None,:]
            
        self.cumhisto = np.zeros([N_clust])
        if self.homeinv:
            self.tphisto = np.ones([N_clust])/N_clust
            
        if algo == 'maro':
            self.last_time_activated = np.zeros(N_clust).astype(int)
        
    def homeorule(self): # careful with the method you choose for example: gain(mpursuit) != gain(lagorce) 
        '''
        '''
        #______USED IN MP WITH COSINE SIMILARITY:
        histo = self.cumhisto.copy()+np.ones([len(self.cumhisto)])
        histo/=np.sum(histo)
        # faire une moyenne flottante sur un certain nombre d'events (ou avec le temps) 
        # faire deux règles lente -> homeo ou rapide pour le codage
        if self.algo=='mpursuit':
            mu = 1
            gain = np.log(histo)/np.log(mu/self.kernel.shape[1])
        #__________________________________________
        else:
            gain = np.exp(self.kernel.shape[1]/4*(histo-1/self.kernel.shape[1]))
        return gain
    
    def inversehomeo(self): # rule for translation invariance of the features
        gainv = np.exp(self.kernel.shape[1]/400*(1/self.kernel.shape[1]-self.tphisto))
        return gainv
    
    def run(self, TS, learn):
        plotdic = False
        if self.algo=='lagorce':
            h, temphisto, dist = self.lagorce(TS, learn)
        elif self.algo=='mpursuit':
            h, temphisto, dist = self.mpursuit(TS, learn)
        elif self.algo=='maro':
            h, temphisto, dist = self.maro(TS, learn)         
        self.cumhisto += temphisto
        self.tphisto = 0.5*self.tphisto+0.5*temphisto
            
        if learn:
            self.nbtrain += 1
        if self.hout == 1:
            p = np.ceil(h)
        elif self.hout == 2:
            p = h
        else:
            p = np.zeros([len(h),1])
            p[np.argmax(h)] = 1
            
        if plotdic==True:
            if self.nbtrain % 10000==0:
                self.plotdicpola(self,len(h),self.R)
        return p, dist
    
    
##____________DIFFERENT METHODS________________________________________________________
    
    def lagorce(self, TS, learn):
        
        if self.krnlinit=='first':
            while self.nbtrain<self.kernel.shape[1]:
                self.kernel[:,self.nbtrain]=TS.T
                h = np.zeros([self.kernel.shape[1]])
                h[self.nbtrain] = 1
                temphisto = h.copy()
                return h, temphisto, 0

        Distance_to_proto = np.linalg.norm(TS - self.kernel, ord=2, axis=0)
        
        gain = np.ones([len(self.cumhisto)])
        if self.homeo:
            gain *= self.homeorule()
        if self.homeinv:
            gain *= self.inversehomeo()
        closest_proto_idx = np.argmin(Distance_to_proto*gain.T)

        if learn:
            pk = self.cumhisto[closest_proto_idx]
            Ck = self.kernel[:,closest_proto_idx]
            alpha = 0.01/(1+pk/20000)
            beta = np.dot(Ck.T, TS)[0]/(np.linalg.norm(TS)*np.linalg.norm(Ck))
            Ck_t = Ck + alpha*(TS.T[0] - beta*Ck)
            self.kernel[:,closest_proto_idx] = Ck_t
            
        h = np.zeros([self.kernel.shape[1]])
        h[closest_proto_idx] = 1
        temphisto = h.copy()
        
        return h, temphisto, Distance_to_proto[closest_proto_idx]
    
    def maro(self, TS, learn):

        Distance_to_proto = np.linalg.norm(TS - self.kernel, ord=2, axis=0)
        
        if self.homeo:
            gain = self.homeorule()
            closest_proto_idx = np.argmin(Distance_to_proto*gain.T)
        else:
            closest_proto_idx = np.argmin(Distance_to_proto)
            
        if learn:
            pk = self.cumhisto[closest_proto_idx]
            Ck = self.kernel[:,closest_proto_idx]
            self.last_time_activated[closest_proto_idx] = self.nbtrain
            alpha = 1/(1+pk)
            beta = np.dot(Ck.T, TS)[0]/(np.linalg.norm(TS)*np.linalg.norm(Ck))
            Ck_t = Ck + alpha*(TS.T[0] - beta*Ck)
            self.kernel[:,closest_proto_idx] = Ck_t
            
            critere = (self.nbtrain-self.last_time_activated) > 10000
            critere2 = self.cumhisto < 25000
            if np.any(critere2*critere):
                cri = self.cumhisto[critere] < 25000
                idx_critere = np.arange(0, self.kernel.shape[1])[critere][cri]
                for idx_c in idx_critere:
                    beta = np.dot(Ck.T, TS)[0]/(np.linalg.norm(TS)*np.linalg.norm(Ck))
                    Ck_t = Ck + 0.2*beta*(TS.T[0]-Ck)
                    self.kernel[:,idx_c] = Ck_t
        
        h = np.zeros([self.kernel.shape[1]])
        h[closest_proto_idx] = 1
        temphisto = h.copy()
        
        return h, temphisto, Distance_to_proto[closest_proto_idx]
    
    def mpursuit(self, TS, learn):
        alpha = 1
        eta = 0.005
        h = np.zeros([self.kernel.shape[1]]) # sparse vector
        temphisto = np.zeros([len(h)])
        corr = np.dot(self.kernel.T,TS).T[0]
        Xcorr = np.dot(self.kernel.T, self.kernel)
        if self.homeo:
            gain = self.homeorule()
        while np.max(corr)>0: # here, Xcorr has relatively high values, meaning clusters are correlated. With the update rule of the MP, coefficients of corr can get negative after few iterations. This criterion is used to stop the loop
            if self.homeo:
                ind = np.argmax(corr*gain)
            else:
                ind = np.argmax(corr)
            h[ind] = corr[ind].copy()/Xcorr[ind,ind]
            corr -= alpha*h[ind]*Xcorr[:,ind]
            if learn:
                self.kernel[:,ind] = self.kernel[:,ind] + eta*h[ind]*(TS.T-self.kernel[:,ind])
                self.kernel[:,ind] = self.kernel[:,ind]/np.sqrt(np.sum(self.kernel[:,ind]**2))
            temphisto[ind] += 1
        return h, temphisto
    
##____________PLOTTING_________________________________________________________________________
    
    def plotdicpola(lay, pola, R):
        fig = plt.figure(figsize=(15,5))
        fig.suptitle("Dictionary after {0} events" .format(lay.nbtrain))
        for n in range(len(lay.kernel[0,:])):
            for p in range(pola):
                sub = fig.add_subplot(pola,len(lay.kernel[0,:]),n+len(lay.kernel[0,:])*p+1)
                dico = np.reshape(lay.kernel[p*(2*R+1)**2:(p+1)*(2*R+1)**2,n], [int(np.sqrt(len(lay.kernel)/pola)), int(np.sqrt(len(lay.kernel)/pola))])
                sub.imshow((dico))
                sub.axes.get_xaxis().set_visible(False)
                sub.axes.get_yaxis().set_visible(False)
        plt.close("all")
        display.clear_output(wait=True)
        display.display(fig)