"""
Contains the StreamSurvey class, which reads and formats raw survey data as well
as additional helper classes.
"""

import logging
import re

import numpy as np
import pandas as pd

from . import streamexceptions

class StreamSurvey(object):
    
    """
    Reads in a geomorphic survey and formats it for further use.
    
    Attributes:
        file(str): name or filepath of the csv that contains the survey data.
        sep(str): the separating character in the file.
        keywords(dict): a dictionary that relates keywords in the survey descriptions to geomorphic features.
        data(pandas.core.frame.DataFrame): pandas dataframe representing the imported survey.
        colRelations(dict): a dictionary that relates standardized names to the column names of the survey.
    """
    
    def __init__(self,file,sep=',',keywords=None,colRelations=None):
        """
        file(str): name or filepath of the csv that contains the survey data.
        sep(str): the separating character in the file.
        keywords(dict): a dictionary that relates geomorphic features to how they were called out in the survey.
                        If nothing is passed, a default dictionary is used.
        colRelations(dict): a dictionary that relates standardized names to the column names of the survey.
                            If nothing is passed, a default dictionary is used.
        """
        self.file = file
        if keywords is None:
            self.keywords = {'Profile':'pro', #mandatory
                             'Cross Section':'xs', #mandatory
                             'Riffle':'ri',
                             'Run':'ru',
                             'Pool':'po',
                             'Glide':'gl',
                             'Top of Bank':'tob',
                             'Bankfull':'bkf',
                             'Water Surface':'ws',
                             'Thalweg':'thw',
                             'breakChar':'-', #mandatory
                             'commentChar':'_' #mandatory
                             }
        else:
            self.keywords = keywords
        
        if colRelations is None:
            self.colRelations = {'shotnum':'Name',
                                 'whys':'Northing',
                                 'exes':'Easting',
                                 'zees':'Elevation',
                                 'desc':'Description',
                                 }
        else:
            self.colRelations = colRelations
            
        self.sep = sep
            
        self.importSurvey()
        
    def importSurvey(self):
        df=pd.read_csv(self.file, sep=',')
        self.data = df
            
    def pack_shots(self):
        """
        Packs each row in self.data into a Shot object and returns an array
        """
        packed = [Shot(shotLine,self.colRelations,self.keywords) for shotLine in self.data.itertuples()]
        return(packed)
        
    def filter_survey_type(self,packedShots,surveyType):
        """
        Filters a list of packed shots by the 'type' key in the meaning attribute.
        """
        result = [pack for pack in packedShots if pack.meaning['type'] == surveyType]
        return(result)
        
    def get_names(self,packedShots):
        """
        Takes a list of packed shots in and returns a dict relating names to count
        """
        names = [shot.meaning['name'] for shot in packedShots]
        counter = {}
        for name in names:
            try:
                counter[name] += 1
            except KeyError:
                counter[name] = 1
        return(counter)
    
class Parser(object):
    """
    Parses desc strings.
    """
    
    def __init__(self,parseDict):
        self.parseDict = parseDict
        
    def dictSplit(self,string):
        """
        Breaks the desc string into its name, descriptors and comment (if any)
        """
        result = {'name':None,
                  'descriptors':[None],
                  'comment':None
                 }
        
        breakChar = self.parseDict['breakChar']
        commentChar = self.parseDict['commentChar']
        
        splitAtComment = string.split(commentChar)
        try:
            result['comment'] = splitAtComment[1]
        except IndexError:
            pass
            
        splitByBreaker = splitAtComment[0].split(breakChar)
        result['name'] = splitByBreaker[0]
        try:
            result['descriptors'] = splitByBreaker[1:]
        except IndexError:
            pass
        
        return(result)
    
    def string_is_in(self,matchString,string):
        """
        Returns true if matchString is in string.
        """
        contained = re.search(matchString,string)
        if contained:
            return True
        else:
            return False
        
    def key_is_in(self,key,string):
        return(self.string_is_in(self.parseDict[key],string))
    
    def get_meaning(self,string):
        """
        Gets the semantic meaning of the dictionary returned by self.dictSplit.
        """
        result = {'type':None, # profile or cross section
                  'morphs':[], # depends on if type is profile or cross section
                  'name':None,
                  'full':string
                 }
        splitDict = self.dictSplit(string)
        result['name'] = splitDict['name']
        
        if self.key_is_in('Profile',result['name']):
            result['type'] = 'Profile'
        elif self.key_is_in('Cross Section',result['name']):
            result['type'] = 'Cross Section'
            
        for descriptor in splitDict['descriptors']:
            for key,pattern in self.parseDict.items():
                if self.string_is_in(pattern,descriptor):
                    result['morphs'].append(key)
        
        return(result)
        
class Shot(object):
    """
    A survey shot.
    
    Attributes:
        shotline(pandas.core.frame.Pandas): A series representing the survey shot.
        keywords(dict): a dictionary that relates keywords in the survey descriptions to geomorphic features.
        colRelations(dict): a dictionary that relates column headers in the survey to standardized meanings.
        shotnum(int): the shot number
        ex(float): the x-coordinate of the shot
        why(float): the y-coordinate of the shot
        zee(float): the z-coordinate of the shot
        desc(str): the description of the shot specified when it was taken
        meaning(dict): the semantic meaning of the desc string
        """
    
    def __init__(self,shotLine,colRelations,parseDict):
        """
        shotLine(pandas.core.frame.Pandas): A series representing the survey shot.
        parseDict(dict): a dictionary that relates keywords in the survey descriptions to geomorphic features.
        colRelations(dict): a dictionary that relates column headers in the survey to standardized meanings.
        """
        self.shotLine = shotLine
        self.colRelations = colRelations
        self.parseDict = parseDict
        
        self.set_shotnum()
        self.set_ex()
        self.set_why()
        self.set_zee()
        self.set_desc()
        self.set_meaning()
        
    def __str__(self):
        return('(Shot ' + str(self.shotnum) + ':' + str(self.desc)+')')
        
    def __repr__(self):
        return('(Shot ' + str(self.shotnum) + ':' + str(self.desc)+')')
        
    def set_shotnum(self):
        self.shotnum = getattr(self.shotLine,self.colRelations['shotnum'])
        
    def set_ex(self):
        self.ex = getattr(self.shotLine,self.colRelations['exes'])
        
    def set_why(self):
        self.why = getattr(self.shotLine,self.colRelations['whys'])
        
    def set_zee(self):
        self.zee = getattr(self.shotLine,self.colRelations['zees'])
        
    def set_desc(self):
        self.desc = getattr(self.shotLine,self.colRelations['desc'])
        
    def set_meaning(self):
        parsed = Parser(self.parseDict)
        meaning = parsed.get_meaning(self.desc)
        self.meaning = meaning
        