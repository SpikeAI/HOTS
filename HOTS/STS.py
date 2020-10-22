__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import numpy as np
import time

class STS(object):
    '''
    Object to represent Spatio-Temporal Surface, and methods to generate them.

    INPUT :
        + tau : (<float>) time constant (in second) of the decay of the kernel
        + R : (<int>) half size of the neighbourhood to consider, 1 time surface (for 1 polarity), will have a
            size (2*R+1,2*R+1)
        + initial_time : (<float>) initialization timing value i.e all event will -initial_time as time stamp
            at initialization
        + sigma : (<float>) parameter of filtering in the circular filter. If None, there is no filter applied
        + verbose : (<int>) control the verbosity
    '''

    def __init__(self, tau, R, initial_time=100, sigma=None, verbose=0):
        self.verbose = verbose
        self.tau = tau
        self.R = R

        self.RF_diam = 2*self.R + 1
        self.area = self.RF_diam**2

        self.initial_time = initial_time

        self.X_p, self.Y_p = np.meshgrid(np.arange(-self.R, self.R+1),
                                         np.arange(-self.R, self.R+1), indexing='ij')
        self.radius = np.sqrt(self.X_p**2 + self.Y_p**2)

        if sigma is None:
            self.mask_circular = np.ones_like(self.radius)
        else:
            self.mask_circular = np.exp(- .5 *
                                        self.radius**2 / self.R ** 2 / sigma**2)
        # self.mask_circular = self.mask_circular.reshape((1, self.mask.shape[0]self.mask.shape[1]))

    def create(self, event, stop=None, kernel='exponential'):
        '''
        Method to generate spatiotemporal surface
        INPUT :
            + event : (<object event>) stream of event used to generate the spatiotemporal surface
            + stop : (<int>) tools to debug, stopping the generation at event number stop
            + kernel : (<string>) type of kernel to use, could be exponential or linear
        OUTPUT :
            + Surface : (<np array>) matrix of size (nb_of_event, nb_polarity*((2*R+1)*(2*R+1)))
                representing the SpatioTemporalSurface
        '''
        self.ListPolarities = event.ListPolarities
        self.nb_polarities = len(self.ListPolarities)

        self.width = event.ImageSize[0] + 2*self.R
        self.height = event.ImageSize[1] + 2*self.R
        self.ListOfTimeMatrix = np.zeros((self.nb_polarities, self.width, self.height))-self.initial_time
        self.BinaryMask = np.zeros((self.nb_polarities, self.width, self.height))

        if stop is not None:
            self.Surface = np.zeros((stop+1, self.nb_polarities * self.area))
        else:
            self.Surface = np.zeros(
                (event.address.shape[0], self.nb_polarities * self.area))
        timer = time.time()
        idx = 0
        t_previous = 0
        for idx_event, [addr, t, pol] in enumerate(zip(event.address, event.time, event.polarity)):
            if t < t_previous:
                self.ListOfTimeMatrix = np.zeros(
                    (self.nb_polarities, self.width, self.height)) - self.initial_time

            x, y = addr + self.R
            self.ListOfTimeMatrix[pol, x, y] = t
            self.LocalTimeDiff = t - self.ListOfTimeMatrix[:, (x-self.R):(x+self.R+1), (y-self.R):(y+self.R+1)]

            if kernel == 'exponential':
                SI = np.exp(-(self.LocalTimeDiff)/self.tau)*self.mask_circular
            elif kernel == 'linear':
                # TODO : deprecate this kernel
                mask = (self.LocalTimeDiff < self.tau) * 1.
                mask *= self.mask_circular
                SI = ((1-self.LocalTimeDiff/self.tau)  * mask)
            else:
                print('error')

            self.Surface[idx_event, :] = SI.reshape((len(event.ListPolarities) * self.area))
            #normalization of the STS
            self.Surface[idx_event,:] /= np.linalg.norm(self.Surface[idx_event,:])
            
            t_previous = t
            if idx_event == stop:
                break

        tac = time.time()
        if self.verbose != 0:
            print(
                'Generation of SpatioTemporal Surface in ------ {0:.2f} s'.format((tac-timer)))
        return self.Surface

    def FilterRecent(self, event, threshold=0):
        '''
        Method to filter events. Only the events associated with a surface having
            enough recent events in the neighbourhood with be kept.

        INPUT :
            + event : (<object event>) stream of event to filter
            + threshold : (<float>), filtering parameter. 0 means no filter
        OUTPUT :
            + event_output : (<object event>) filtered stream of event
            + filt : (<np array>) bolean vector of size (nb_of_input event). A False is assocated with the
                removed event and a True is assocated with kept event
        '''
        
        threshold = threshold*self.R
        binsurf = (self.Surface != 0).astype(int)
        filt = np.sum(binsurf, axis=1) > threshold
        self.Surface = self.Surface[filt]
        event_output = event.filter(filt)

        return event_output, filt
