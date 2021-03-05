import numpy as np

class stats(object):
    """ """

    def __init__(self, N, camsize):
        self.nbqt = 1000
        self.count = 0
        self.dist_cum = 0
        self.dist = []
        self.actmap = np.zeros([N,camsize[0]+1,camsize[1]+1])
        self.histo = []
        #self.Ddic = []
        #self.spar = []
        #self.Ddic_cum = 0
        #self.spar_cum = 0
        #self.predrcum = 0
        #self.predr = []

    def update(self, h, dic, X, dist):
            idx = np.argmax(h)
            self.dist_cum += dist
            #self.Ddic_cum += np.linalg.norm(dic-dicprev)/np.linalg.norm(dicprev)
            #self.spar_cum += len(np.nonzero(h)[0])
            #self.predrcum += np.linalg.norm(X)
            self.count += 1
            if self.count==self.nbqt:
                self.dist.append(self.dist_cum/self.nbqt)
                #self.Ddic.append(self.Ddic_cum/self.nbqt)
                #self.spar.append(self.spar_cum/self.nbqt)
                #self.predr.append(self.predrcum/self.nbqt)
                self.dist_cum = 0
                #self.Ddic_cum = 0
                #self.spar_cum = 0
                #self.predrcum = 0
                self.count = 0
