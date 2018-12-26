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
        Lengths are expressed in terms of millimeters or inches.
    
    Attributes:
        distr(dict): the dictonary relating grain size and prevalence
        name(str): the name of the XS
        metric(bool): whether the units are inches (False) or mm (True)
        triggerRecalc(bool): whether to recalculate statistics upon modificatoin of self.dist
        cumSum(:obj:'list' of :obj:'float'): the cumulative sum of the grain prevalences.
        bins(dict): a dictionary that relates ISO size classes to their minimum size and prevalence
        medianSize(float): the median grainsize
        meanSize(float): the mean grainsize
        sort(float): the sorting coefficient
        skewness(float): the skewness coefficient
        kurtosis(float): the kurtosis coefficient
        stddev(float): the standard deviation of the sample
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
        if self.metric:
            self.unitDict = sc.METRIC_CONSTANTS
        elif not(self.metric):
            self.unitDict = sc.IMPERIAL_CONSTANTS
        
        self.calculate_stats()
        
    def __str__(self):
        """
        Prints the name of the GrainDistribution object. If the name attribute is None, prints "UNNAMED".
        """
        if self.name:
            return(self.name)
        else:
            return("UNNAMED")
            
    def make_bindict(self):
        """
        Makes a dictionary that relates ISO 14688-1:2002 classification names to lower size bounds.
        """
        self.bins = {
                'C':[0], #clay
                'FM':[0.002], #fine silt
                'MM':[0.0063], #medium silt
                'CM':[0.02], #coarse sit
                'FS':[0.063], #fine sand
                'MS':[0.2], #medium sand
                'CS':[0.63], #coarse sand
                'FG':[2], #fine gravel
                'MG':[6.3], #medium gavel
                'CG':[20], # coarse gravel
                'Cob':[63], #cobble
                'Bol':[200], #boulder
                'LBol':[630] #large boulder
                }
        if not self.metric:
            for key in self.bins:
                self.bins[key] = [self.bins[key][0] * self.unitDict['milToInches']]   
        
    def reset_distr(self,value,recalculate = None):
        """
        When distr is redefined.
        """
        self._distr = value
        if recalculate == None:
            recalculate = self.triggerRecalc
        if recalculate:
            self.calculate_stats()
        
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
        self.make_bindict()
        self.sort_distr()
        self.make_cum_sum()
        self.normalize_cum_sum()
        self.make_median_grainsize()
        self.make_mean_grainsize()
        self.make_sorting()
        self.make_skewness()
        self.make_kurtosis()
        self.make_stddev()
        self.bin_particles()
    
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
        
    def num_to_phi(self,n,metric=None):
        """
        Converts a grainsize to its phi (-log2) representation. Takes inches or mm.
        """
        if metric is None:
            metric = self.metric
        if not metric:
            n = n * self.unitDict['inchesToMil']
            
        phi = -np.log2(n)
        return(phi)
        
    def phi_to_num(self,phi,metric=None):
        """
        Converts a grainsize from its phi (-log2) representation into inches or mm.
        """
        if metric is None:
            metric = self.metric
        n = 2**(-phi)
        if not metric:
            n = n * self.unitDict['milToInches']
        return(n)
        
    def make_median_grainsize(self):
        """
        Calculates the median grainsize of the pebble count.
        """
        self.medianSize = self.dx(50)
        
    # grainsize stats from http://core.ecu.edu/geology/rigsbyc/rigsby/Sedimentology/CalculationOfGrainSizeStatistics.pdf
    
    def make_mean_grainsize(self):
        """
        Calculates the mean grainsize of the pebble count in the proper unit (inches or millimeters, not phi).
        """
        meanGrainsize = (self.num_to_phi(self.dx(16)) + self.num_to_phi(self.dx(50)) + self.num_to_phi(self.dx(84))) / 3
        self.meanGrainsize = self.phi_to_num(meanGrainsize)
        
    def make_sorting(self):
        """
        Calculates the sorting of the pebble count.
        """
        sorting = (self.num_to_phi(self.dx(84)) - self.num_to_phi(self.dx(16)))/4 + (self.num_to_phi(self.dx(95)) - self.num_to_phi(self.dx(5)))/6.6
        self.sorting = sorting
        
    def make_skewness(self):
        """
        Calculates the skewness of the curve.
        """
        skewness = (self.num_to_phi(self.dx(16)) + self.num_to_phi(self.dx(84)) - 2*self.num_to_phi(self.dx(50))) / (2*(self.num_to_phi(self.dx(84)) - self.num_to_phi(self.dx(16))))
        + (self.num_to_phi(self.dx(5)) + self.num_to_phi(self.dx(95)) - 2*self.num_to_phi(self.dx(50))) / (2*(self.num_to_phi(self.dx(95)) - self.num_to_phi(self.dx(5))))
        self.skewness = skewness
        
    def make_kurtosis(self):
        """
        Calculates the kurtosis of the curve.
        """
        kurtosis = (self.num_to_phi(self.dx(95)) - self.num_to_phi(self.dx(5)))/(2.44*(self.num_to_phi(self.dx(75)) - self.num_to_phi(self.dx(25))))
        self.kurtosis = kurtosis
        
    def make_stddev(self):
        """
        Sets the standard deviation of the pebble count.
        """
        self.stddev = (self.dx(84) - self.dx(16)) / 2
        
    def make_countlist(self):
        """
        Takes the distr dict and returns a list that contains every particle as its own entry in a list.
        """
        partlist = []
        for key in self.distr:
            partlist.extend([key]*self.distr[key])
        return(partlist)
        
    def bin_particles(self):
        """
        Bins the distr into appropriate categories (fine sand, coarse gravel, silt, etc.) and appends it to the appriate self.bins key
        Should not be called twice before resetting the distr dictionary.
        """
        particles = self.make_countlist()
        partbins = list(self.bins.values())
        newpartbins = [] # we need to remove the inner lists from partbins
        for partbin in partbins:
            newpartbins.append(partbin[0])
        bincount = np.digitize(particles, newpartbins, right=False) - 1
        
        tacklist = [0]*(len(partbins))
        for i in range(0,len(bincount)):
            location = bincount[i]
            tacklist[location] += 1
            
        keys = list(self.bins.keys()) 
        for i in range(len(keys)):
            key = keys[i]
            self.bins[key].append(tacklist[i])
            
    def estimate_mannings_n(self):
        """
        Estimate's Manning's n in the channel based on roughness. Not accurate for vegetated channels.
        """
        pass
    
    def extract_binned_counts(self):
        """
        Returns a list of pebble count values that corresponds to the order of the keys for self.bins
        """
        thelist = []
        for key in self.bins:
            thelist.append(self.bins[key][1])
        return(thelist)
    
    def extract_binned_cumsum(self):
        binned = self.extract_binned_counts()
        return(list(np.cumsum(binned)))
        
    def cplot(self,normalize=True):
        """
        A cumulative distribution plot of the particle sizes.
        """
        keys = list(self.bins.keys())
        vals = self.extract_binned_cumsum()
        if normalize:
            vals = np.asarray(vals)
            factor = 100/max(vals)
            vals = list(vals*factor)
        plt.plot(keys,vals)
    
    def dplot(self):
        """
        A distribution plot of the particle sizes.
        """
        pass
    
    def bplot(self,normalize=True):
        """
        A bar plot of of the particle sizes.
        """
        keys = list(self.bins.keys())
        vals = self.extract_binned_counts()
        if normalize:
            vals = np.asarray(vals)
            factor = 100/max(vals)
            vals = list(vals*factor)
        plt.bar(keys,vals)
