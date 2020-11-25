import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import copy

class TimeSurface(object):
    """ TimeSurface is a class created from a serie of events. It stores the events on the pixel grid and apply an exponential decay to past events when updating the time surface. It returns the timesurface defined by a spatial window. The output is a 1D vector representing the time-surface.
    
    ATTRIBUTES:
            R -> the parameter defining the length of the spatial window of the time-surface 
            tau -> the constant of the decay applied to events
            camsize -> the size of the pixel grid
            iev -> the indice of the reference event to build the time-surface
            spatpmat -> the spatiotemporal matrix of the whole pixel grid
            x, y, t, p -> position, time and polarity of the last event of the time surface
            dtemp -> minimum time required between 2 events on the same pixel to avoid camera issues 
                                        (some pixels (x>255) spike 2 times)
            kthrs -> constante*tau defining a null threshold for past events
            beta -> is a constante to apply a temporal decay if p[pol]<1 
            pola -> boolean indicating if the network uses polarities or not

    METHODS: 
            .addevent -> add an event to the time surface when event.x, event.y, event.t and p is given as input
                                (p is a 1D vector containing different weights for all polarities)
            .plote -> plot the timesurface TimeSurface.timesurf
                parameters: timesurf, gamma to display events (2.2 default)
"""

    def __init__(self, R, tau, camsize, nbpol, pola, filt=2):
        # PARAMETERS OF THE TIME SURFACES
        self.R = R
        self.tau = tau 
        self.camsize = camsize
        self.dtemp = 2
        self.kthrs = 5
        self.beta = 1000
        self.pola = pola
        self.filt = filt
        # VARIABLES OF THE TIME SURFACE
        self.x = 0
        self.y = 0
        self.t = 0
        self.iev = 0
        self.spatpmat = np.zeros((nbpol,camsize[0],camsize[1]))
  
    def addevent(self, xev, yev, tev, pev):
        timesurf = np.zeros((len(pev),2*self.R+1,2*self.R+1))
        if self.iev==0:
            self.iev += 1
            self.t = tev
            polz = np.nonzero(pev)[0]
            for i in range(len(polz)):
                self.spatpmat[polz[i], xev, yev] = np.exp(-self.beta*(1-pev[polz[i]])/self.tau)
                timesurf[polz[i], self.R, self.R] = np.exp(-self.beta*(1-pev[polz[i]])/self.tau)
        elif xev==self.x and yev==self.y and tev-self.t<self.dtemp:
            pass # no update because camera can spike two times for 1 event
        else:
            self.iev += 1
            self.x = xev
            self.y = yev
            # updating the spatiotemporal surface
            self.spatpmat = self.spatpmat*np.exp(-(float(tev-self.t))/self.tau)
            # making threshold for small elements
            self.spatpmat[self.spatpmat<np.exp(-float(self.kthrs))]=0
            self.t = tev
            polz = np.nonzero(pev)[0]
            for i in range(len(polz)):
                self.spatpmat[polz[i], xev, yev] = np.exp(-self.beta*(1-pev[polz[i]])/self.tau)
            # making time surface
            xshift = copy.copy(self.x)
            yshift = copy.copy(self.y)
            # padding for events near the edges of the pixel grid
            temp_spatpmat = copy.copy(self.spatpmat)
            if self.x<self.R:
                temp_spatpmat = np.lib.pad(temp_spatpmat,((0,0),(self.R+1,0),(0,0)),'symmetric')
                xshift += self.R+1
            elif self.camsize[0]-(self.x+1)<self.R:
                temp_spatpmat = np.lib.pad(temp_spatpmat,((0,0),(0,self.R+1),(0,0)),'symmetric')
            if self.y<self.R:
                temp_spatpmat = np.lib.pad(temp_spatpmat,((0,0),(0,0),(self.R+1,0)),'symmetric')
                yshift += self.R+1
            elif self.camsize[1]-(self.y+1)<self.R:
                temp_spatpmat = np.lib.pad(temp_spatpmat,((0,0),(0,0),(0,self.R+1)),'symmetric')
            timesurf = temp_spatpmat[:,int(xshift-self.R):int(xshift+self.R)+1,int(yshift-self.R):int(yshift+self.R)+1]
            # reshaping timesurf as a 1D vector
        activ = False
        if self.pola == False:
            TS = np.reshape(timesurf[np.argmax(pev)], [(2*self.R+1)**2,1])
            card = np.nonzero(timesurf)
            minact = self.filt*self.R
        else: 
            TS = np.reshape(timesurf, [len(timesurf)*(2*self.R+1)**2,1])
            n = np.argmax(pev)
            card = np.nonzero(timesurf[n])
            minact = self.filt*self.R
        normTS = np.linalg.norm(TS)
        if np.linalg.norm(TS)>0:
            TS /= normTS
        if len(card[0])>minact:
            activ = True
        return TS, activ

    def plote(self, timesurf, gamma=2.2):
        fig = plt.figure(figsize=(10,5))
        sub1 = fig.add_subplot(1,3,1)
        sub1.imshow((self.spatpmat[0].T)**gamma)
        sub1.plot(self.x,self.y,'r*')
        sub2 = fig.add_subplot(1,3,2)
        sub2.imshow((timesurf[0].T)**gamma)
        sub3 = fig.add_subplot(1,3,3)
        sub3.imshow((timesurf[1].T)**gamma)
        plt.close("all")
        display.clear_output(wait=True)
        display.display(fig) 
