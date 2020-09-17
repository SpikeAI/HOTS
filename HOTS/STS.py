__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import numpy as np
import time


class STS(object):
    '''
    Object with spatiotemporal Surface and methods to generate them
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
        self.ListOfTimeMatrix = np.zeros(
            (self.nb_polarities, self.width, self.height))-self.initial_time
        self.BinaryMask = np.zeros(
            (self.nb_polarities, self.width, self.height))

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
            self.LocalTimeDiff = t - \
                self.ListOfTimeMatrix[:, (x-self.R):(x+self.R+1), (y-self.R):(y+self.R+1)]

            if kernel == 'exponential':
                #SI = np.exp(-(t-self.LocalTimeMatrix)/self.tau).reshape((len(self.ListPolarities)* self.area))
                SI = np.exp(-(self.LocalTimeDiff)/self.tau)*self.mask_circular
                SI2 = SI.reshape((len(self.ListPolarities)*self.area))

            elif kernel == 'linear':
                mask = (self.LocalTimeDiff < self.tau)
                SI = ((1-self.LocalTimeDiff/self.tau)
                      * mask)*self.mask_circular
                SI2 = SI.reshape((len(self.ListPolarities) * self.area))
            else:
                print('error')
            self.Surface[idx_event, :] = SI2  # SI*self.mask
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
        Method to filter the event. Only the event associated with a surface having enought recent event
            in the neighbourhood with be kept
        INPUT :
            + event : (<object event>) stream of event to filter
            + threshold : (<float>), filtering parameter. 0 means no filter
        OUTPUT :
            + event_output : (<object event>) filtered stream of event
            + filt : (<np array>) bolean vector of size (nb_of_input event). A False is assocated with the
                removed event and a True is assocated with kept event
        '''
        # if self.verbose > 0:
        #    print('threshold', threshold)
        threshold = threshold*self.R
        filt = np.sum(self.Surface, axis=1) > threshold
        self.Surface = self.Surface[filt]
        event_output = event.filter(filt)

        return event_output, filt



class Propagate(object):
    def __init__(self, radius, tau, stop=None, recenter_Factor=1.5):
        self.stop = stop
        self.r = radius
        self.recenter_Factor = recenter_Factor
        self.r_recenter = int(np.floor(self.r * self.recenter_Factor))
        self.tau = tau

    def SaveTCandSC(self, event, recenter=True):
        self.TS_Mem = np.zeros(
            (1, event.ImageSize[0], event.ImageSize[1]), dtype=np.int64) - 3*self.tau
        #print('init', self.TS_Mem.copy())
        self.Spike_Mem = np.zeros(
            (1, event.ImageSize[0], event.ImageSize[1]), dtype=bool)
        for idx, each_polarity in enumerate(event.polarity):
            x, y = event.address[0, idx], event.address[1, idx]
            print('old x,y', x, y)
            t = event.time[idx]
            self.TS_Mem[0, event.address[0, idx],
                        event.address[1, idx]] = event.time[idx]
            self.Spike_Mem[0, event.address[0, idx],
                           event.address[1, idx]] = True
            if recenter == True:
                x_d, y_d = self.recenter(x, y, t, each_polarity)
                print('new x,y', x, y)
            TC = t - self.TS_Mem[0, x_d - self.r:x_d +
                                 self.r + 1, y_d - self.r:y_d + self.r + 1]
            print('TC', TC)
            SC = self.Spike_Mem[0, x_d - self.r:x_d +
                                self.r + 1, y_d - self.r:y_d + self.r + 1]
            print('SC', SC)
            if idx == self.stop:
                break
        return TC, SC

    def recenter(self, x, y, t, p):
        # print(self.TS_Mem)

        recenter_ROI_TC = t - self.TS_Mem[0, x - self.r_recenter:x +
                                          self.r_recenter + 1, y - self.r_recenter:y + self.r_recenter + 1]

        te = recenter_ROI_TC.copy()
        print('ROI_TC', te)
        recenter_ROI_TC[recenter_ROI_TC < 3 * self.tau] = 1
        recenter_ROI_TC[recenter_ROI_TC > 1] = 0

        recenter_ROI_SC = self.Spike_Mem[0, x - self.r_recenter:x +
                                         self.r_recenter + 1, y - self.r_recenter:y + self.r_recenter + 1]
        recenter_ROI = recenter_ROI_TC
        #recenter_ROI = recenter_ROI_SC & recenter_ROI_TC

        # print(recenter_ROI_TC)

        selonx = np.sum(recenter_ROI, axis=0)
        selony = np.sum(recenter_ROI, axis=1)
        steps = np.arange(- self.r_recenter, self.r_recenter + 1)
        dx = np.int(np.round(np.dot(steps, selonx) / np.sum(selonx)))
        dy = np.int(np.round(np.dot(steps, selony) / np.sum(selony)))
        # update event coords
        x_d = x + dx
        y_d = y + dy
        #TC = t - self.TS_Mem[0, x_d - self.r:x_d + self.r + 1, y_d - self.r:y_d + self.r + 1]
        #SC = self.Spike_Mem[0, x_d - self.r:x_d + self.r + 1, y_d - self.r:y_d + self.r + 1]
        return x_d, y_d
