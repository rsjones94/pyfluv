"""
Contains the Reference class, which stores, plots and fits regressions to
reference reach data.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from . import streammath as sm

class Reference():

    """
    A collection of reference reach data.

    Attributes:
        reaches(pandas.core.frame.DataFrame): a pandas dataframe containing
            data for each reference reach: name, drainage, XS area, XS width,
            XS depth, XS flow rate.
        eco(str or int): the area that the Reference object should represent,
            usually a level III or IV ecoregion code (e.g., 71, 71f)
    """

    def __init__(self, reaches, eco=None):
        """
        Method to initialize a CrossSection.

        Args:
            reaches(pandas.core.frame.DataFrame): a pandas dataframe containing
                data for each reference reach: name, drainage, XS area, XS width,
                XS depth, XS flow rate.
                eco(str): the level IV ecoregion code (e.g., 71f)
        """
        self.reaches = reaches.copy()
        self.eco = eco

    def identify_draincol(self):
        """
        Determines which column contains drainage data.
        """
        drains = [col for col in self.reaches if 'Drain' in col or 'drain' in col]
        return drains[0]

    def qplot(self, col, plotType='loglog'):
        """
        Plots the drainage area against a specified column.

        Args:
            col: a string pointing to a column in self.reaches
            pltType: a string specifying the plot type. Can be 'loglog', 'semilogx', 'semilogy' or 'linear'
        """
        plotDict = {'loglog':plt.loglog, 'semilogx':plt.semilogx,
                    'semilogy':plt.semilogy, 'linear':None}

        drainCol = self.identify_draincol()
        x = self.reaches[drainCol]
        y = self.reaches[col]

        try:
            plotDict[plotType]()
        except TypeError:
            pass

        plt.scatter(x, y)
        plt.xlabel(drainCol)
        plt.ylabel(col)
        plt.title(f'{self.eco}: {drainCol} vs. {col}')

    def fit(self, col):
        """
        Fits an exponential regression to a specified column and returns the
        coefficients (a,b) for y = a*x^b. Returns the a tuple where the first
        entry is the tuple (a,b) and the second is the r^2 value of the fit.

        Args:
            col: a string pointing to a column in self.reaches
        """
        drainCol = self.identify_draincol()
        exes = np.array(self.reaches[drainCol])
        whys = np.array(self.reaches[col])

        res = curve_fit(sm.func_powerlaw, exes, whys)[0]

        predictions = [sm.func_powerlaw(x, res[0], res[1]) for x in exes]
        r2 = sm.r2(predictions, whys)

        return(res, r2)

    def trend(self, col):
        """
        Adds a power trendline to a plot given a column name in self.reaches.
        Also returns the fit and the fit's r^2 a la self.fit.
        """
        
        fit = self.fit(col)
        res = fit[0]
        drainCol = self.identify_draincol()
        xMin = min(self.reaches[drainCol])
        xMax = max(self.reaches[drainCol])
        xSpace = np.linspace(xMin, xMax, 10000)
        newY = [sm.func_powerlaw(x, res[0], res[1]) for x in xSpace]
        plt.plot(xSpace, newY)
        
        return(fit)
