"""
Contains the GrainDistribution class, which stores and processes grain size surveys.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import streamconstants as sc
from . import streammath as sm


class GrainDistribution():

    """
    A generic grain size/particle sample.
        Lengths are expressed in terms of millimeters or inches.

    Attributes:
        distr(dict): the dictonary relating grain size and prevalence
        name(str): the name of the XS
        metric(bool): whether the units are inches (False) or mm (True)
        """

    def __init__(self, distr, name=None, metric=False):
        """
        Method to initialize a CrossSection.

        Args:
            distr: a dictionary that relates a grain size to a count or % of
                the total distribution. Will be sorted by key on initialization.
                A second, less preferred option is to pass distr as a list,
                where each entry is a grainsize. This will be converted to a
                dictionary relating grainsize to its prevalence. If a dict is
                passed and a key is 'Bedrock', or 'Bedrock' occurs in a list passed
                then those counts will be treated as large boulders (>1024mm).
                No other string keys or entries are permitted.
            name: the name of the grain size count
            metric: True for units of mm, False for inches

        Raises:
            None.
        """
        bedrockCalls = 'Bedrock' in distr
        if bedrockCalls:
            if metric:
                bedrockVal = 1024
            else:
                bedrockVal = 40.31496

        self.distr = distr.copy()
        if isinstance(self.distr, list):
            self.distr = [i if i != 'Bedrock' else bedrockVal for i in self.distr]
            self.distr = sm.make_countdict(self.distr)
        else:
            if bedrockCalls:
                try:
                    self.distr[bedrockVal] += self.distr['Bedrock']
                    del self.distr['Bedrock']
                except KeyError:
                    self.distr[bedrockVal] = self.distr['Bedrock']
                    del self.distr['Bedrock']
        for key, value in self.distr.items():
            if np.isnan(value):
                self.distr[key] = 0
        self.name = name
        self.metric = metric
        if self.metric:
            self.unitDict = sc.METRIC_CONSTANTS
        elif not self.metric:
            self.unitDict = sc.IMPERIAL_CONSTANTS

        self.sortdistr()

    def __str__(self):
        """
        Prints the name of the GrainDistribution object. If the name attribute is None, prints "UNNAMED".
        """
        if self.name:
            return self.name
        return "UNNAMED"

    def make_bindict(self):
        """
        Makes a dictionary that relates ISO 14688-1:2002 classification names to lower size bounds.
        """
        bins = {'C':[0], #clay
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
            for key in bins:
                bins[key] = [bins[key][0] * self.unitDict['milToInches']]

        return bins

    def sortdistr(self):
        """
        Sorts the distr dict by key.
        """
        self.distr = {k: self.distr[k] for k in sorted(self.distr.keys())}

    def cumulative_sum(self):
        """
        Returns the cumulative sum of distr. Returns a list, not a dict.
        """
        keys = list(self.distr.keys())

        cumSum = [self.distr[keys[0]]]
        for i in range(1, len(keys)):
            cumSum.append(cumSum[i-1] + self.distr[keys[i]])

        return cumSum

    def normalize_cum_sum(self):
        """
        Normalizes the cumulative sum so that the max is 100.
        """
        cumSum = self.cumulative_sum()
        ratio = 100/max(cumSum)
        normCumSum = np.asarray(cumSum)*ratio
        return list(normCumSum)

    def dx(self, x):
        """
        Returns the particle size s where x% of the particles in the distribution are smaller than s.
        """
        sizes = [0]
        sizes.extend(self.distr.keys())

        cSum = [0]
        cSum.extend(self.normalize_cum_sum())

        for i, _ in enumerate(cSum):
            if cSum[i] >= x:
                overIndex = i
                break

        underSize = sizes[overIndex-1]
        underSum = cSum[overIndex-1]

        overSize = sizes[overIndex]
        overSum = cSum[overIndex]

        regression = sm.line_from_points((underSize, underSum), (overSize, overSum))
        dx = sm.x_from_equation(x, regression)
        return dx

    def num_to_phi(self, n, metric=None):
        """
        Converts a grainsize to its phi (-log2) representation. Takes inches or mm.
        """
        if metric is None:
            metric = self.metric
        if not metric:
            n = n * self.unitDict['inchesToMil']

        phi = -np.log2(n)
        return phi

    def phi_to_num(self, phi, metric=None):
        """
        Converts a grainsize from its phi (-log2) representation into inches or mm.
        """
        if metric is None:
            metric = self.metric
        n = 2**(-phi)
        if not metric:
            n = n * self.unitDict['milToInches']
        return n

    def median(self):
        """
        Calculates the median grainsize of the pebble count.
        """
        return self.dx(50)

    def mean(self):
        """
        Calculates the mean grainsize of the pebble count.
        From http://core.ecu.edu/geology/rigsbyc/rigsby/Sedimentology/CalculationOfGrainSizeStatistics.pdf
        """
        meanGrainsize = (self.num_to_phi(self.dx(16)) + self.num_to_phi(self.dx(50)) + self.num_to_phi(self.dx(84))) / 3
        return self.phi_to_num(meanGrainsize)

    def sorting(self):
        """
        Calculates the sorting coefficient of the pebble count.
        From http://core.ecu.edu/geology/rigsbyc/rigsby/Sedimentology/CalculationOfGrainSizeStatistics.pdf
        """
        sorting = (self.num_to_phi(self.dx(84)) - self.num_to_phi(self.dx(16)))/4 + (self.num_to_phi(self.dx(95)) - self.num_to_phi(self.dx(5)))/6.6
        return sorting

    def skewness(self):
        """
        Calculates the skewness of the curve.
        From http://core.ecu.edu/geology/rigsbyc/rigsby/Sedimentology/CalculationOfGrainSizeStatistics.pdf
        """
        skewness = (self.num_to_phi(self.dx(16)) + self.num_to_phi(self.dx(84)) - 2*self.num_to_phi(self.dx(50))) / (2*(self.num_to_phi(self.dx(84)) - self.num_to_phi(self.dx(16)))) + (self.num_to_phi(self.dx(5)) + self.num_to_phi(self.dx(95)) - 2*self.num_to_phi(self.dx(50))) / (2*(self.num_to_phi(self.dx(95)) - self.num_to_phi(self.dx(5))))
        return skewness

    def kurtosis(self):
        """
        Calculates the kurtosis of the curve.
        From http://core.ecu.edu/geology/rigsbyc/rigsby/Sedimentology/CalculationOfGrainSizeStatistics.pdf
        """
        kurtosis = (self.num_to_phi(self.dx(95)) - self.num_to_phi(self.dx(5)))/(2.44*(self.num_to_phi(self.dx(75)) - self.num_to_phi(self.dx(25))))
        return kurtosis

    def stddev(self):
        """
        Returns the standard deviation of the pebble count.
        """
        return (self.dx(84) - self.dx(16)) / 2

    def make_countlist(self):
        """
        Takes the distr dict and returns a list that contains every particle
        as its own entry in a list. If the dict values are floats they will be
        coerced to ints.
        """
        partlist = []
        for key in self.distr:
            partlist.extend([key]*int(self.distr[key]))
        return partlist

    def bin_particles(self):
        """
        Bins the distr into appropriate categories (fine sand, coarse gravel, silt, etc.) and appends it to the appriate self.bins key
        """
        binned = self.make_bindict()
        particles = self.make_countlist()
        partbins = list(binned.values())
        newpartbins = [] # we need to remove the inner lists from partbins
        for partbin in partbins:
            newpartbins.append(partbin[0])
        bincount = np.digitize(particles, newpartbins, right=False) - 1

        tacklist = [0]*(len(partbins))
        for i, _ in enumerate(bincount):
            location = bincount[i]
            tacklist[location] += 1

        keys = list(binned.keys())
        for i, _ in enumerate(keys):
            key = keys[i]
            binned[key].append(tacklist[i])

        return binned


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
        binned = self.bin_particles()
        for key in binned:
            thelist.append(binned[key][1])
        return thelist

    def extract_binned_cumsum(self):
        """
        Returns the cumulative sum for the counts in distr after binning them.
        """
        binned = self.extract_binned_counts()
        return list(np.cumsum(binned))

    def extract_unbinned_cumsum(self):
        """
        Returns the cumulative sum for the counts in distr wihthout binning them.
        """
        unbinned = list(self.distr.values())
        return list(np.cumsum(unbinned))

    def sizeplot(self, normalize=True, semilog=True, cumulative=True):
        """
        A plot of the particles in the distribution.
        This calls a new plot figure.
        """
        _, ax = plt.subplots()

        cumulativeDict = {True:self.extract_unbinned_cumsum(),
                          False:self.distr.values()
                          }

        y = cumulativeDict[cumulative]
        if normalize:
            y = [float(i)/sum(y)*100 for i in y]
        x = list(self.distr.keys())

        if semilog:
            ax.semilogx(x, y)
        else:
            ax.plot(x, y)
        ax.grid()

        normalDict = {True:' (Normalized)', False:''}
        isCumulativeDict = {True:'Cumulative ', False:''}
        plt.title(str(self))
        plt.xlabel('Size (' + self.unitDict['smallLengthUnit'] + ')')
        plt.ylabel(f'{isCumulativeDict[cumulative]}Count{normalDict[normalize]}')

    def bplot(self, normalize=True):
        """
        A bar plot of of the particle sizes against the size bins
        """
        binned = self.bin_particles()
        keys = list(binned.keys())
        vals = self.extract_binned_counts()
        if normalize:
            vals = [float(val)/sum(vals)*100 for val in vals]
        plt.bar(keys, vals)

        normalDict = {True:' (Normalized)', False:''}
        plt.title(str(self))
        plt.xlabel('Size Class')
        plt.ylabel('Count' + normalDict[normalize])
