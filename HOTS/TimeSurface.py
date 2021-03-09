import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from IPython import display
import copy

class TimeSurface(object):
    """ TimeSurface is a class created from a stream of events. It stores the events on the pixel grid and apply an exponential decay to past events when updating the time surface. It returns the timesurface defined by a spatial window. The output is a 1D vector representing the time-surface.

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
            beta -> is a constant to apply a temporal decay if p[pol]<1
            pola -> boolean indicating if the network uses polarities or not

    METHODS:
            .addevent -> add an event to the time surface when event.x, event.y, event.t and p is given as input
                                (p is a 1D vector containing different weights for all polarities)
                        output: time surface, activ (bolean indicating if the number of non zero pixels in the time surface is above a threshold)
            .plote -> plot the timesurface TimeSurface.timesurf
                parameters: timesurf, gamma to display events (2.2 default)
            .getts -> take the time surface within the spatial window defined by R on the matrix spatpmat
"""

    def __init__(self, R, tau, camsize, nbpol=2, pola=True, filt=2, sigma=None, decay='exponential'):
        # PARAMETERS OF THE TIME SURFACES
        self.R = R
        self.tau = tau # in ms
        self.camsize = camsize
        self.dtemp = 0.002
        self.kthrs = 5
        self.beta = 1
        self.pola = pola
        self.filt = filt
        self.sigma = sigma
        self.decay = decay
        # VARIABLES OF THE TIME SURFACE
        self.x = 0
        self.y = 0
        self.t = 0
        self.p = 0
        self.iev = 0
        self.spatpmat = np.zeros((nbpol,camsize[0]+1,camsize[1]+1))

    def addevent(self, xev, yev, tev, pev):
        timesurf = np.zeros((self.spatpmat.shape[0],2*self.R+1,2*self.R+1))
        xev, yev = int(min(xev,self.camsize[0])), int(min(yev,self.camsize[1]))
        xev, yev = int(max(xev,0)), int(max(yev,0))
        if isinstance(pev, (int, float)):
            p = np.zeros((self.spatpmat.shape[0]))
            p[int(pev)]=1
            pev = p.copy()
        elif pev.size<2:
            p = np.zeros((self.spatpmat.shape[0]))
            p[int(pev)]=1
            pev = p.copy()
        if self.iev==0:
            self.iev += 1
            self.t = tev
            self.p = np.argmax(pev)
            polz = np.nonzero(pev)[0]
            #to change if pev has multiple polarities
            self.spatpmat[np.argmax(pev), xev, yev] = 1
            timesurf[np.argmax(pev), self.R, self.R] = 1
            #for i in range(len(polz)):
                #self.spatpmat[polz[i], xev, yev] = np.exp(-self.beta*(1-pev[polz[i]])/self.tau)
                #timesurf[polz[i], self.R, self.R] = np.exp(-self.beta*(1-pev[polz[i]])/self.tau)
        elif xev==self.x and yev==self.y and tev-self.t<self.dtemp:
            #print('error in time between 2 events on the same pixel: '+ str(tev-self.t) +' ms')
            pass # no update because camera can spike two times for 1 event
        else:
            self.iev += 1
            self.x = xev
            self.y = yev
            # updating the spatiotemporal surface
            if self.decay == 'exponential':
                self.spatpmat = self.spatpmat*np.exp(-(float(tev-self.t))/self.tau)
                # making threshold for small elements
                self.spatpmat[self.spatpmat<np.exp(-float(self.kthrs))]=0
            elif self.decay == 'linear':
                self.spatpmat = max(self.spatpmat-(tev-self.t)/self.tau,0)
            self.t = tev
            self.p = np.argmax(pev)
            polz = np.nonzero(pev)[0]
            #to change if pev has multiple polarities
            self.spatpmat[np.argmax(pev), xev, yev] = 1
            #for i in range(len(polz)):
                #self.spatpmat[polz[i], xev, yev] = np.exp(-self.beta*(1-pev[polz[i]])/self.tau)
            # making time surface

            #print(timesurf.shape, xev, yev)
            timesurf = self.getts()
            #if xev<self.R or xev+self.R>self.camsize[0]:
            #self.plote()
            #print(timesurf.shape)
            if np.sum(timesurf[:,self.R,self.R])==0:
                print('TS pas centrée', xev, yev)

            if self.sigma is not None:
                X_p, Y_p = np.meshgrid(np.arange(-self.R, self.R+1),
                                         np.arange(-self.R, self.R+1))
                radius = np.sqrt(X_p**2 + Y_p**2)
                mask_circular = np.exp(- .5 * radius**2 / self.R ** 2 / self.sigma**2)
                timesurf *= mask_circular

            #if self.camsize[1]-(self.y+1)<self.R or self.camsize[0]-(self.x+1)<self.R:
                #self.plote()
            # reshaping timesurf as a 1D vector
        activ = False
        if self.pola == False:
            TS = np.reshape(timesurf[np.argmax(pev)], [(2*self.R+1)**2,1])
            card = np.nonzero(timesurf[np.argmax(pev)])
            minact = self.filt*self.R
        else:
            TS = np.reshape(timesurf, [len(timesurf)*(2*self.R+1)**2,1])
            card = np.nonzero(timesurf)
            minact = self.filt*self.R
        normTS = np.linalg.norm(TS)
        if np.linalg.norm(TS)>0:
            TS /= normTS
        if len(card[0])>minact:
            activ = True
        return TS, activ

    def getts(self):
        xshift = copy.copy(self.x)
        yshift = copy.copy(self.y)
        # padding for events near the edges of the pixel grid
        temp_spatpmat = copy.copy(self.spatpmat)
        if self.x-self.R<0:
            temp_spatpmat = np.lib.pad(temp_spatpmat,((0,0),(self.R,0),(0,0)),'symmetric')
            xshift += self.R
        elif self.camsize[0]<self.x+self.R:
            temp_spatpmat = np.lib.pad(temp_spatpmat,((0,0),(0,self.R),(0,0)),'symmetric')
        if self.y-self.R<0:
            temp_spatpmat = np.lib.pad(temp_spatpmat,((0,0),(0,0),(self.R,0)),'symmetric')
            yshift += self.R
        elif self.camsize[1]<self.y+self.R:
            temp_spatpmat = np.lib.pad(temp_spatpmat,((0,0),(0,0),(0,self.R)),'symmetric')
        timesurf = temp_spatpmat[:,int(xshift-self.R):int(xshift+self.R)+1,int(yshift-self.R):int(yshift+self.R)+1]
        return timesurf

    def plote(self, gamma=2.2):

        timesurf = self.getts()

        fig = plt.figure(figsize=(10,5))
        sub1 = fig.add_subplot(1,3,1)
        mapa = sub1.imshow((self.spatpmat[self.p].T)**gamma, cmap=plt.cm.plasma)
        sub1.plot(self.x,self.y,'r*')
        sub1.plot([self.x-self.R, self.x-self.R], [self.y-self.R, self.y+self.R], color='red')
        sub1.plot([self.x-self.R, self.x+self.R], [self.y-self.R, self.y-self.R], color='red')
        sub1.plot([self.x-self.R, self.x+self.R], [self.y+self.R, self.y+self.R], color='red')
        sub1.plot([self.x+self.R, self.x+self.R], [self.y-self.R, self.y+self.R], color='red')
        sub1.set_title('OFF events with exponential decay')
        cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
        fig.colorbar(mapa, cax=cbar_ax);
        sub2 = fig.add_subplot(1,3,2)
        sub2.imshow((timesurf[0].T)**gamma, cmap= plt.cm.plasma)
        sub2.set_title('OFF time-surface')
        sub3 = fig.add_subplot(1,3,3)
        sub3.imshow((timesurf[1].T)**gamma, cmap= plt.cm.plasma)
        sub3.set_title('ON time-surface')
        plt.close("all")
        display.clear_output(wait=True)
        display.display(fig)


    def plot3D(self, gamma=2.2):

        timesurf = self.getts()

        fig = plt.figure(figsize=(10,5))
        sub1 = fig.add_subplot(1,2,1, projection="3d")
        x = np.linspace(int(self.x-self.R),int(self.x+self.R),2*self.R+1)
        y = np.linspace(int(self.y-self.R),int(self.y+self.R),2*self.R+1)
        X,Y = np.meshgrid(x,y)
        sub1.plot_surface(X,Y,timesurf[0,:,:].T, cmap= plt.cm.plasma, alpha=0.5)
        sub1.set_title('OFF 3D time-surface')
        sub2 = fig.add_subplot(1,2,2, projection="3d")
        x = np.linspace(int(self.x-self.R),int(self.x+self.R),2*self.R+1)
        y = np.linspace(int(self.y-self.R),int(self.y+self.R),2*self.R+1)
        X,Y = np.meshgrid(x,y)
        sub2.plot_surface(X,Y,timesurf[1,:,:].T, cmap= plt.cm.plasma, alpha=0.5)
        sub2.set_title('ON 3D time-surface')

        plt.show()
