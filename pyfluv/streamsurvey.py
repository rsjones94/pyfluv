"""
Contains the StreamSurvey class, which reads and formats raw survey data.
"""

import logging

import numpy as np
import pandas as pd

from . import streamexceptions
from . import streamconstants as sc
from . import streammath as sm

class StreamSurvey(object):
    
    """
    Reads in a geomorphic survey and formats it for further use.
    
    Attributes:
        file(str): name or filepath of the csv that contains the survey data.
        keywords(dict): a dictionayr that relates keywords in the survey descriptions to geomorphic features.
        data(pandas.core.frame.DataFrame): pandas dataframe representing the imported survey
    """
    
    def __init__(self,file,keywords=None,colRelations=None):
        """
        file(str): name or filepath of the csv that contains the survey data.
        keywords(dict): a dictionary that relates geomorphic features to how they were called out in the survey.
                        If nothing is passed, a default dictionary is used.
        colRelations(dict): a dictionary that relates standardized names used by the parser to the column names of the survey.
                            If nothing is passed, a default dictionary is used.
        """
        self.file = file
        if keywords is None:
            self.keywords = {'Profile':'pro',
                             'Cross Sectoin':'xs',
                             'Riffle':'ri',
                             'Run':'ru',
                             'Pool':'po',
                             'Glide':'gl',
                             'Top of Bank':'tob',
                             'bankfull':'bkf',
                             'Water Surface':'ws',
                             'Thalweg':'thw'
                             }
        else:
            self.keywords = keywords
        
        if colRelations is None:
            self.colRelations = {'shotnum':'Name',
                                 'whys':'Northing',
                                 'exes':'Easting',
                                 'zees':'Elevation':
                                 'desc':'Description'
                                 }
        else:
            self.colRelations = colRelations
            
        self.importSurvey()
        
    def importSurvey(self):
        df=pd.read_csv(self.file, sep=',')
        self.data = df
        
    def writeSurvey(self,name,parsed = True):
        """
        Writes the survey data to a csv. If parsed is true, cross sections and profiles
        will be written to separate files.
        """
        if parsed:
            raise NotImplementedError('Parsing not yet implemented.')
        else:
            self.data.to_csv(name)
            
    def parseNames(self):
        """
        Parses the survey using the desc column.
        """