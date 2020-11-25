import numpy as np
from mix_Layer import *
from mix_TimeSurface import *
from mix_Stats import *
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
                        hout = False, #works only with mpursuit
                        homeo = False,
                        pola = True,
                        to_record = True,
                ):
        tau *= 1000 # to enter tau in ms
        if to_record == True:
            self.stats = [[]]*nblay
        self.TS = [[]]*nblay
        self.L = [[]]*nblay
        self.count = 0
        for lay in range(nblay):
            if lay == 0:
                self.TS[lay] = TimeSurface(R, tau, camsize, nbpolcam, pola)
                self.L[lay] = layer(R, nbclust, pola, nbpolcam, camsize, homeo, algo, hout, to_record)
                if to_record == True:
                    self.stats[lay] = stats(pola, R)
            else:
                self.TS[lay] = TimeSurface(R*(K_R**lay), tau*(K_tau**lay), camsize, nbclust*(K_clust**(lay-1)), pola)
                self.L[lay] = layer(R*(K_R**lay), nbclust*(K_clust**lay), pola, nbclust*(K_clust**(lay-1)), camsize, homeo, algo, hout, to_record)
                if to_record == True:
                    self.stats[lay] = stats(pola, R*(K_R**lay))
        self.L[lay].out = 1

    def run(self, x, y, t, p):
        lay = 0
        learn = False
        while lay<len(self.TS):
            timesurf, activ = self.TS[lay].addevent(x, y, t, p)
            if activ==True:
                p, dicprev = self.L[lay].run(timesurf, learn)
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
                lay+=1
            else:
                lay = len(self.TS)
