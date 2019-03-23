"""
Contains the Reference class, which stores, plots and fits regressions to
reference reach data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

class Reference():

    """
    A collection of reference reach data.

    Attributes:
        reaches(pandas.core.frame.DataFrame): a pandas dataframe containing
            data for each reference reach: name, drainage, XS area, XS width,
            XS depth, XS flow rate.
        eco(str or int): the level III or IV ecoregion code (e.g., 71, 71f)
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

    def qplot(self, col, logx=True, logy=True):
        """
        Plots the drainage area against a specified column.
        """
        _, ax = plt.subplots()

        drainCol = self.identify_draincol()
        x = self.reaches[drainCol]
        y = self.reaches[col]

        ax.loglog()
        plt.scatter(x,y)
        plt.xlabel(drainCol)
        plt.ylabel(col)
        plt.title(f'{self.eco}: {drainCol} vs. {col}')

    def fit(self, col):
        """
        Fits an exponential regression to a specified column and returns the
        coefficients (a,b) for y = a*x^b
        """
        pass
