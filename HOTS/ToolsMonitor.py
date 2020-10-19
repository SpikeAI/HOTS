__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"

import numpy as np
import matplotlib
import math
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def dispo(pola, nb_center=None, scale=1):
    '''
    helper function fr graph position
    INPUT :
        + pola : (<int>) number of polarity
        + nb_center: (<int>) number of cluster center
        + scale : (<int>) scaling parameter, if 1 graph will have 6 patch
            per column, if 2 graph will have 6 patch per column
    OUTPUT :
        + dispo : (<tuple>) disposition in the format (nb_line,nb_column)
    '''
    if nb_center is None:
        nb_center = 1
    if scale == 1:
        if nb_center*pola >= 8:
            dispo = (((nb_center*pola)//8)+1, 8)
        else:
            dispo = (nb_center, 8)
    elif scale == 2:
        if nb_center*pola >= 16:
            dispo = (((nb_center*pola)//16)+1, 16)
        else:
            dispo = (nb_center, 16)
    return dispo


def DisplayImage(list_of_event, multi_image=0):
    '''
    Function to accumulated event as an image
    INPUT :
        + list_of_event : (<list>) of (<object event>) stream of event to display
        + multi_image : (<int>) option to display another image than the first one
    '''
    subplotpars = matplotlib.figure.SubplotParams(
        left=0., right=1., bottom=0., top=1., wspace=0.1, hspace=0.2)
    if type(list_of_event) is not list:
        raise TypeError(
            'the argument of the function should be a list of event object')
    nb_of_image = len(list_of_event)
    disp = dispo(nb_of_image)
    fig = plt.figure(figsize=(12, 12*disp[0]/disp[1]), subplotpars=subplotpars)
    idx = 0
    for each_event in list_of_event:
        ax = fig.add_subplot(disp[0], disp[1], idx+1)
        image = np.zeros(each_event.ImageSize)
        if multi_image == 0:
            lst_idx = each_event.ChangeIdx[0] + 1
            fst_idx = 0
        else:
            lst_idx = each_event.ChangeIdx[multi_image] + 1
            fst_idx = each_event.ChangeIdx[multi_image-1]
        image[each_event.address[fst_idx:lst_idx, 0].T,
              each_event.address[fst_idx:lst_idx, 1].T] = each_event.polarity[fst_idx:lst_idx].T
        img = ax.imshow(image, interpolation='nearest')
        ax.axis('off')
        ax.set_title('Image {0}'.format(idx+1), fontsize=8)
        idx += 1
        
        
def DisplayPola(list_of_event, ImageSize, nb_pola, R=2, rect=False):
    '''
    Function to accumulated event as an image
    INPUT :
        + list_of_event : (<list>) of (<object event>) stream of event to display
        + multi_image : (<int>) option to display another image than the first one
    '''
    idl = int(np.floor(np.sqrt(nb_pola)))
    idc = int(np.ceil(nb_pola/idl))
    fig, axes = plt.subplots(nrows=idl, ncols=idc, figsize=(15,8))
    sub = []
    idp = 0
    for ax in axes.flat:
        if idp == nb_pola:
            break
        ev_p = list_of_event.polarity[list_of_event.polarity==idp].copy()
        ev_t = list_of_event.time[list_of_event.polarity==idp].copy()
        ev_x = list_of_event.address[list_of_event.polarity==idp].copy()
        ima = np.zeros(ImageSize)
        for i in range(len(ev_p)):
            ima[ev_x[i][0], ev_x[i][1]] = ev_t[i]*1000
        ima[ima==0]='nan'
        impol = ax.imshow(ima)
        idp += 1
        if rect==True:
            ax.plot([list_of_event.address[-1][1]-R, list_of_event.address[-1][1]-R], [list_of_event.address[-1][0]-R, list_of_event.address[-1][0]+R], color='red')
            ax.plot([list_of_event.address[-1][1]+R, list_of_event.address[-1][1]+R], [list_of_event.address[-1][0]-R, list_of_event.address[-1][0]+R], color='red')
            ax.plot([list_of_event.address[-1][1]-R, list_of_event.address[-1][1]+R], [list_of_event.address[-1][0]-R, list_of_event.address[-1][0]-R], color='red')
            ax.plot([list_of_event.address[-1][1]-R, list_of_event.address[-1][1]+R], [list_of_event.address[-1][0]+R, list_of_event.address[-1][0]+R], color='red')
            ax.plot(list_of_event.address[-1][1], list_of_event.address[-1][0], marker = 'o', color='red')
    fig.suptitle('OFF and ON events', fontsize=16, fontweight='bold')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(impol, cax=cbar_ax, label="timescale in millisec")
    plt.show()  


def DisplaySurface3D(Surface, nb_polarities, angle=(20, 90)):
    '''
    Function to display 3D graph of spatiotemporal surface
    INPUT :
        + Surface : (<np.array>) of size (nb_surface,nb_polarity*(2*R+1)*(2*R+1))
        + nb_polarities : (<int>) number of polarities per surface
        + angle : (<tuple>) of (<int>) describing the displaying angle of the
            spatiotemporal surface
    '''
    mini = np.amin(Surface)-0.01
    maxi = np.max(Surface)+0.01
    cNorm = matplotlib.colors.Normalize(vmin=mini-0.1, vmax=maxi+0.1)
    cmapo = colormaps()[2]

    idx = 0
    nb_center = Surface.shape[0]
    if len(Surface.shape) == 2:
        area = int(Surface.shape[1]/nb_polarities)
        Surface = Surface.reshape((nb_center, nb_polarities, area))
    else:
        area = int(Surface.shape[2])

    size = int(np.sqrt(area))
    R = int((size-1)/2)
    subplotpars = matplotlib.figure.SubplotParams(
        left=0., right=1., bottom=0., top=1., wspace=0, hspace=0)
    # ,indexing='ij')
    X, Y = np.meshgrid(np.arange(-R, R+1), np.arange(-R, R+1))
    disp = dispo(nb_polarities, nb_center)
    #fig = plt.figure(figsize=(disp[1]*3,disp[0]*3))
    fig = plt.figure(figsize=(12, 12*disp[0]/disp[1]), subplotpars=subplotpars)
    for idx_surf, each_surf in enumerate(Surface):
        for idx_pol, pol_surf in enumerate(each_surf):
            ax = fig.add_subplot(disp[0], disp[1], idx+1, projection='3d')
            final = pol_surf.reshape((size, size), order='F')
            surf = ax.plot_surface(X, Y, final, rstride=1, cstride=1,
                                   cmap=cmapo, norm=cNorm,
                                   linewidth=0, antialiased=True)

            ax.set_zlim(mini, maxi)
            ax.view_init(angle[0], angle[1])
            axe = np.linspace(-R, R, 5).astype(int)
            plt.yticks(axe)
            plt.xticks(axe)
            ax.tick_params(labelsize=6)
            ax.set_title('Cluster {0}, polarity {1}'.format(
                idx_surf, idx_pol), fontsize=10)
            idx = idx+1


def DisplaySurface2D(Surface, nb_polarities):
    '''
    Function to display 2D graph of spatiotemporal surface
    INPUT :
        + Surface : (<np.array>) of size (nb_surface,nb_polarity*(2*R+1)*(2*R+1))
        + nb_polarities : (<int>) number of polarities per surface
    '''
    subplotpars = matplotlib.figure.SubplotParams(
        left=0., right=1., bottom=0., top=1., wspace=0.1, hspace=0.1)
    nb_center = Surface.shape[0]  # len(ClusterCenter)

    if len(Surface.shape) == 2:
        area = int(Surface.shape[1]/nb_polarities)
        Surface = Surface.reshape((nb_center, nb_polarities, area))
    else:
        area = int(Surface.shape[2])
    fig = plt.figure(figsize=(nb_center*0.6, nb_polarities*0.6),
                     subplotpars=subplotpars)
    dim_patch = int(np.sqrt(area))
    idx = 0
    for idx_center, each_center in enumerate(Surface):
        for idx_pol, surface in enumerate(each_center):
            ax = fig.add_subplot(nb_polarities, nb_center,
                                 idx_center+idx_pol*nb_center+1)

            cmin = 0
            cmax = np.max(surface)
            # print(cmax)
            ax.imshow(surface.reshape((dim_patch, dim_patch)), cmap=plt.cm.gray_r, vmin=cmin, vmax=cmax,
                      interpolation='nearest')
            ax.set_xticks(())
            ax.set_yticks(())
            #ax.set_title('Cl {0} - Pol {1}'.format(idx_center,idx_pol),fontsize= 6)
            idx = idx+1


def GenerateActivationMap(Event, Cluster, mode='separate'):
    '''
    Function to generate activation map from a stream of event
    INPUT :
        + Event : (<object event>) holding the polarities and addresses to generate the activation map
        + Cluster : (<object Cluster>) holding the cluster centers
        + mode : (<string>) parameter to choose if we want to separate each activation map in different
            image ('separate'), or if we want to gather all activation map into one images ('global')
    OUTPUT :
        + activation_map : (<np.array>) represnting the activation map .
            size : (nb_polarities,image_height,image_width) if mode='separate'
                if mode = 'global' the size is (1,image_height,image_width)
    '''
    nb_cluster = Cluster.nb_cluster
    if mode == 'separate':
        activation_map = np.zeros(
            (nb_cluster, Event.ImageSize[0], Event.ImageSize[1]))
    elif mode == 'global':
        activation_map = np.zeros(Event.ImageSize)
    else:
        raise KeyError('the mode argument is not valid')
    for idx_event, ev in enumerate(Event.polarity[0:Event.ChangeIdx[0]]):
        address_int = Event.address[idx_event, :]
        x, y = address_int[0], address_int[1]

        if mode == 'global':
            activation_map[x, y] = ev+1
        else:
            for i in range(nb_cluster):
                activation_map[i, x, y] = 0
            try:
                activation_map[ev, x, y] = 1
            except IndexError:
                print(ev, idx_event)
    return activation_map


def DisplayActivationMap(activation_map, scale=1):
    '''
    Function display the activation map
    INPUT :
        + activation_map : (<np.array>) represnting the activation map .
            size : (nb_polarities,image_height,image_width) if mode='separate'
                if mode = 'global' the size is (1,image_height,image_width)
        + scale : (<int>) scaling parameter to determine the number of image per line
    '''
    subplotpars = matplotlib.figure.SubplotParams(
        left=0., right=1., bottom=0., top=1., wspace=0.1, hspace=0.2)
    nb_map = activation_map.shape[0]  # len(ClusterCenter)
    disp = dispo(nb_map, scale=scale)
    fig = plt.figure(figsize=(10, 10*disp[0]/disp[1]), subplotpars=subplotpars)
    idx = 0
    for idx_map, each_map in enumerate(activation_map):
        ax = fig.add_subplot(disp[0], disp[1], idx+1)
        cmin = 0
        cmax = 1
        to_plot = ax.imshow(each_map, cmap=plt.cm.gray_r, vmin=cmin, vmax=cmax,
                            interpolation='nearest')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title('Map Cl {0}'.format(idx_map), fontsize=8)
        idx = idx+1


def DisplayConvergence(ClusterLayer, to_display=['error'], eta=None, eta_homeo=None):
    '''
    Function to display the monitored variable during the training
    INPUT :
        + ClusterLayer : (<list>) of (<object Cluster>) holding the cluster object for each layer
        + to_display : (<list>) of (<string>) to indicate which monitoring variable to display. 'error' will plot the L2 error,
            'histo' will the activated cluster histogram
        + eta : (<float>) to add eta in the name of the graph
        + eta_homeo : (<float>) to add eta_homeo in the name of the graph
    '''
    if type(ClusterLayer) is not list:
        ClusterLayer = [ClusterLayer]
    subplotpars = matplotlib.figure.SubplotParams(
        left=0., right=1., bottom=0., top=1., wspace=0.1, hspace=0.2)
    fig = plt.figure(figsize=(10, 2*len(ClusterLayer)),
                     subplotpars=subplotpars)
    location = 1
    for idx, each_Layer in enumerate(ClusterLayer):
        for idx_type, each_type in enumerate(to_display):
            each_type = str(each_type)
            ax = fig.add_subplot(len(ClusterLayer), len(to_display), location)
            max_x = each_Layer.record[each_type].shape[0] * \
                each_Layer.record_each
            ax.set_xticks([0, roundup(max_x/3, each_Layer.record_each),
                           roundup(2*max_x/3, each_Layer.record_each)])
            if each_type == 'error':
                to_plot = plt.plot(each_Layer.record[each_type])
                if (eta is not None) and (eta_homeo is not None):
                    ax.set_title('Convergence Layer {0} with eta : {1} and eta_homeo : {2}'.format(
                        idx+1, eta, eta_homeo), fontsize=8)
                else:
                    ax.set_title(
                        'Convergence Layer {0}'.format(idx+1), fontsize=8)
            elif each_type == 'histo':
                to_plot = plt.bar(np.arange(each_Layer.nb_cluster), each_Layer.record[each_type].values[-1],
                                  width=np.diff(np.arange(each_Layer.nb_cluster+1)), ec="k", align="edge")
                if (eta is not None) and (eta_homeo is not None):
                    ax.set_title('Histogram of activation at Layer {0} with eta : {1} and eta_homeo : {2}'.format(
                        idx+1, eta, eta_homeo), fontsize=8)
                else:
                    ax.set_title(
                        'Histogram of activation at Layer {0}'.format(idx+1), fontsize=8)
            location += 1


def roundup(x, step):
    return int(math.ceil(x / step)) * step


def DisplayHisto(freq, pola):
    return plt.bar(pola[:-1], freq, width=np.diff(pola), ec="k", align="edge")


'''
def Displ(ClusterLayer):
    subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.2, hspace=0.2)
    fig = plt.figure(figsize=(10,10/5),subplotpars=subplotpars)

    for idx,each_Layer in enumerate(ClusterLayer) :

        #print('number of record',each_Layer.record['error'].shape)
        #print('recordstep',each_Layer.record_each)
        ax = fig.add_subplot(1,len(ClusterLayer),idx+1)
        max_x = each_Layer.record['error'].shape[0]*each_Layer.record_each
        ax.set_xticks([0,roundup(max_x/3,each_Layer.record_each),roundup(2*max_x/3,each_Layer.record_each)])
        to_plot = plt.plot(each_Layer.record['error'])
        ax.set_title('Convergence Layer {0}'.format(idx+1),fontsize= 8)
        #ax.tick_params(axis='x',length=10)
'''
