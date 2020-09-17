#!/usr/bin/python
# -*- coding: utf8 -*
####################
# libDataHelper.py #
####################
# April 2017 - Jean-Matthieu Maro
# Email: jean-matthieu dot maro, hosted at inserm, which is located in France.

# read db files and so on to easily load data and labels

import numpy as np


class db:
    filenames = []
    labelnames = []
    labelids = []
    dblabels = []
    path = ''


class layer_specs:
    ncenters = 0
    tau = 0.0
    radius = 0


class hots_network_settings:
    enableviewer = False
    tcp_port = 3333
    update_sec = 8
    n_dim = 2
    n_pol = 1
    max_dim = np.empty(0, dtype=np.int16)
    layers_specs = np.empty(0, dtype=layer_specs)
    fixed_centers_files = []
    data_file = 'none'
    db_file = 'none'
    db_path = 'none'
    workdir_path = ''
    save_prefix = ''


def superReadDB(_filename, _dbpath='none'):
    dbspecs = db()
    needFirstLine = True
    file = open(_filename, 'r')
    for line in file:
        if line[0] == '?' or line[0] == '%':
            # print('discard comment line')
            pass
        else:
            if needFirstLine:
                n_files = int(line)
                needFirstLine = False
                print('{0} files in the DB.'.format(n_files))
            else:
                l = line.strip('\n').split()
                dbspecs.filenames.append(l[0])
                dbspecs.labelnames.append(l[1])
                dbspecs.labelids.append(int(l[2]))
                dbspecs.dblabels.append(int(l[3]))
    if _dbpath != 'none':
        dbspecs.path = _dbpath

    return dbspecs


def superLoadHOTSNetworkSettingsFromFile(_filename, _verbose):
    file = open(_filename, 'r')
    specs = hots_network_settings()
    for line in file:
        if line[0] == '?' or line[0] == '%':
            # print('discard comment')
            pass
        else:
            split = line.split(':')
            if split[0] == 'viewer_enable':
                specs.enableviewer = bool(split[1])
            elif split[0] == 'tcp_port':
                specs.tcp_port = int(split[1])
            elif split[0] == 'update_sec':
                specs.update_sec = int(split[1])
            elif split[0] == 'n_pol':
                specs.n_pol = int(split[1])
            elif split[0] == 'n_dim':
                specs.n_dim = int(split[1])
            elif split[0] == 'max_dim':
                specs.max_dim = np.zeros(len(split) - 1, dtype=np.int16)
                for ii in np.arange(len(split) - 1):
                    specs.max_dim[ii] = int(split[ii+1])
            elif split[0] == 'layers':
                specs.layers_specs = np.empty(
                    len(split) - 1, dtype=layer_specs)
                for ii in np.arange(len(split) - 1):
                    l = split[ii+1].split('-')
                    ls = layer_specs()
                    ls.ncenters = int(l[0])
                    ls.tau = float(l[1])
                    ls.radius = int(l[2])
                    specs.layers_specs[ii] = ls
            elif split[0] == 'data_file':
                specs.data_file = split[1].strip('\n')
            elif split[0] == 'db_file':
                specs.db_file = split[1].strip('\n')
            elif split[0] == 'db_path':
                specs.db_path = split[1].strip('\n')
            elif split[0] == 'workdir_path':
                specs.workdir_path = split[1].strip('\n')
            elif split[0] == 'save_prefix':
                specs.save_prefix = split[1].strip('\n')
            elif split[0] == 'fixed_centers_files':
                for ii in np.arange(len(split) - 1):
                    specs.fixed_centers_files.append(split[ii+1].strip('\n'))
    if _verbose:
        print('------ Loaded settings ------')
        print('Enable viewer: {0} (Port: {1}, update {2})'.format(
            specs.enableviewer, specs.tcp_port, specs.update_sec))
        print(' ')
        print('Number of polarities of the input: {0}'.format(specs.n_pol))
        print('Number of dimensions and sizes: {0}, {1}'.format(
            specs.n_dim, specs.max_dim))
        print('Number of layers: {0}'.format(specs.layers_specs.size))
        for l in specs.layers_specs:
            print('> {0}, {1}, {2}'.format(l.ncenters, l.tau, l.radius))
        print('Number of fixed layers: {0}'.format(
            len(specs.fixed_centers_files)))
        for l in specs.fixed_centers_files:
            print('> {0}'.format(l))
        print(' ')
        print('Data file: {0}'.format(specs.data_file))
        print('DB file: {0}'.format(specs.db_file))
        print('DB path: {0}'.format(specs.db_path))
        print('Workdir path: {0}'.format(specs.workdir_path))
        print('Save id: {0}'.format(specs.save_prefix))
        print('------ End of settings ------')
        print(' ')
    return specs
