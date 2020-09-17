___author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS (2017-) Antoine Grimaldi (2020-)"
__version__ = '2020-09-17'
__licence__ = 'GPLv3'
__all__ = ['STS.py']
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
from HOTS import Tools
from HOTS import KmeansCluster
from HOTS import KmeansLagorce
from HOTS import KmeansMaro
from HOTS import KmeansHomeo
from HOTS import KmeansCompare
