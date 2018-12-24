"""
Contains the GrainDistribution class, which stores and processes grain size surveys.

"""
import logging

import matplotlib.pyplot as plt
import numpy as np

import streamconstants as sc
import streammath as sm


class GrainDistribution(object):
    
    """
    A generic grain size/particle sample.
        Lengths are expressed in terms of meters or feet.
        Time is expressed in terms of seconds.
        Mass is express in terms of kilograms or slugs.
        Weights are express in terms of newtons or pounds.
    
    Attributes:
        distr(dict): the dictonary relating grain size and prevalence
        name(str): the name of the XS
        metric(bool): whether the units are inches (False) or mm (True)
        triggerRecalc(bool): whether to recalculate statistics upon modificatoin of self.dist
        self.cumSum(:obj:'list' of :obj:'float'): the cumulative sum of the grain prevalences.
        """
    
    def __init__(self, distr, name = None, metric = False, triggerRecalc = True):
        """
        Method to initialize a CrossSection.
        
        Args:
            distr: a dictionary that relates a grain size to a count or % of the total distribution. Will be sorted by key on initialization.
            name: the name of the grain size count
            metric: True for units of mm, False for inches
            triggerRecalc: if triggerRecalc is True, then modifying the self.dist will trigger recalculation of statistics
            
        Raises:
            None.
        """
        self._distr = distr
        self.name = name
        self.metric = metric
        self.triggerRecalc = triggerRecalc
        
        self.calculate_stats()
           
    def reset_distr(self,value,recalculate = None):
        """
        When distr is redefined.
        """
        self._distr = value
        if recalculate == None:
            recalculate = self.triggerRecalc
        if recalculate:
            self.sort_distr()
        
    def get_distr(self):
        """
        Getter when self.distr is called.
        """
        return(self._distr)
        
    distr = property(fget = get_distr,fset = reset_distr)
    # note that the above does not trigger a recalculation if you change a dict value by key rather than redefining the whole dict
    
    def calculate_stats(self):
        """
        Sort the distr, get the cumsum and calculate statistics.
        """
        self.sort_distr()
        self.make_cum_sum()
        self.normalize_cum_sum()
    
    def sort_distr(self):
        """
        Sorts the distr dict by key.
        """
        self._distr = {k: self._distr[k] for k in sorted(self._distr.keys())}
        
    def make_cum_sum(self):
        """
        Makes the self.cumsum property.
        """
        keys = list(self._distr.keys())
        
        cumSum = [self._distr[keys[0]]]
        for i in range(1,len(keys)):
            cumSum.append(cumSum[i-1] + self._distr[keys[i]])
        self.cumSum = cumSum
        
    def normalize_cum_sum(self):
        """
        Normalize the cumSum so that the max is 100.
        """
        ratio = 100/max(self.cumSum)
        normCumSum = np.asarray(self.cumSum)*ratio
        self.normCumSum = list(normCumSum)
    
    def dx(self,x):
        """
        Returns the particle size s where x% of the particles in the distribution are smaller than s.
        """
        sizes = [0]
        sizes.extend(self._distr.keys())
        
        cSum = [0]
        cSum.extend(self.normCumSum)
        print(cSum)
        
        for i in range(0,len(cSum)):
            if cSum[i] >= x:
                overIndex = i
                break
        
        underSize = sizes[overIndex-1]
        underSum = cSum[overIndex-1]
        
        overSize = sizes[overIndex]
        overSum = cSum[overIndex]
        
        regression = sm.line_from_points((underSize,underSum),(overSize,overSum))
        dx = sm.x_from_equation(x,regression)
        return(dx)
        
        
        
    def cplot(self):
        """
        A cumulative distribution plot of the particle sizes.
        """
        pass
    
    def dplot(self):
        """
        A distribution plot of the particle sizes.
        """
        pass
    
    def bplot(self):
        """
        A bar plot of of the particle sizes.
        """
        pass
