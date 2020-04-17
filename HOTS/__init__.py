

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
___author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS"
__version__ = '2017-01-10'
__licence__ = 'GPLv3'
__all__ = ['STS.py']  # , 'shl_tools', 'shl_learn', 'shl_encode']

"""
========================================================
A Hierarchy of event-based Time-Surfaces for Pattern Recognition
========================================================

* This code aims to replicate the paper : 'HOTS : A Hierachy of Event Based Time-Surfaces for
Pattern Recognition' Xavier Lagorce, Garrick Orchard, Fransesco Gallupi, And Ryad Benosman'
"""

from HOTS import STS
from HOTS import Event
from HOTS import Monitor
from HOTS import KmeansCluster
from HOTS import Tools
from HOTS import KmeansHomeoCluster
from HOTS import HomeoTest
