import numpy as np

class stats(object):
    """ """

    def __init__(self, pol, R):
        self.nbqt = 500
        self.count = 0
        self.dist_cum = 0
        self.dist = []
        #self.Ddic = []
        #self.spar = []
        #self.Ddic_cum = 0
        #self.spar_cum = 0
        #self.predrcum = 0
        #self.predr = []
        
    def update(self, h, dic, dicprev, X):
            self.dist_cum += np.linalg.norm(np.dot(dic,h)-X)
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