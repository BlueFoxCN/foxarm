from abc import ABCMeta, abstractmethod

import copy
import itertools as it
import logging
import matplotlib.pyplot as plt
try:
    import mayavi.mlab as mlab
except:
    logging.warning('Failed to import mayavi')

import numpy as np
import os
import sys
import time

import IPython

class GraspQualityConfig(object):
    __metaclass__ = ABCMeta
    def __init__(self, config):
        self.check_valid(config)

        for key, value in config.iteritems():
            setattr(self, key, value)

    def contains(self, key):
        """ Checks whether or not the key is supported """
        if key in self.__dict__.keys():
            return True
        return False

    def __getattr__(self, key):
        if self.contains(key):
            return object.__getattribute__(self, key)
        return None

    def __getitem__(self, key):
        if self.contains(key):
            return object.__getattribute__(self, key)
        raise KeyError('Key %s not found' %(key))
    
    def keys(self):
        return self.__dict__.keys()

    @abstractmethod
    def check_valid(self, config):
        """ Raise an exception if the config is missing required keys """
        pass

