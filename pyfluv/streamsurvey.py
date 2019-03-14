"""
Contains the StreamSurvey class, which reads and formats raw survey data as well
as additional helper classes.
"""

import logging
import re

import numpy as np
import pandas as pd


from . import streamexceptions
from . import streamgeometry as sg
from . import streamprofiles as sp

class StreamSurvey():
    """
    Reads in a geomorphic survey and formats it for further use.

    Attributes:
        file(str): name or filepath of the csv that contains the survey data.
        sep(str): the separating character in the file.
        metric(bool): True if the survey units are in meters, alse if they are in feet
        keywords(dict): a dictionary that relates keywords in the survey descriptions to geomorphic features.
                        The following keys are mandatory: 'Profile','Cross Section','Thalweg','breakChar','commentChar'
                        Additionally, following keys have a specific meaning: 'Riffle','Run','Pool','Glide','Top of Bank',
                            'Bankfull','Water Surface'.
        data(pandas.core.frame.DataFrame): pandas dataframe representing the imported survey.
        colRelations(dict): a dictionary that relates standardized names to the column names of the survey.
                            The following keys are mandatory: 'shotnum','whys','exes','zees','desc'
        profiles(:obj:'list' of :obj:'Shot'): a list of lists, where each sublist is a packed profile
        crossSections(:obj:'list' of :obj:'Shot'): a list of lists, where each sublist is a packed cross section
    """
    def __init__(self,
                 file,
                 sep=',',
                 metric=False,
                 keywords=None,
                 colRelations=None):
        """
        Args:
            file: name or filepath of the csv that contains the survey data.
            sep: the separating character in the file.
            metric: True if the survey units are in meters, alse if they are in feet
            keywords: a dictionary that relates geomorphic features to how they were called out in the survey.
                            If nothing is passed, a default dictionary is used.
            colRelations: a dictionary that relates standardized names to the column names of the survey.
                                If nothing is passed, a default dictionary is used.
        """
        self.file = file
        if keywords is None:
            self.keywords = {'Profile':'pro',
                             'Thalweg':'thw',
                             'Riffle':'ri',
                             'Run':'ru',
                             'Pool':'po',
                             'Glide':'gl',
                             'Water Surface':'ws',
                             'Bankfull':'bkf',
                             'Top of Bank':'tob',
                             'Cross Section':'xs',
                             'Structure':'str',
                             'breakChar':'-',
                             'commentChar':'_'
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
        mandatoryKeywords = ['Profile', 'Cross Section', 'Thalweg', 'breakChar', 'commentChar']
        mandatoryCols = ['shotnum', 'whys', 'exes', 'zees', 'desc']
        hasAllKeywords = all(el in self.keywords.keys() for el in mandatoryKeywords)
        hasAllCols = all(el in self.colRelations.keys() for el in mandatoryCols)
        if not hasAllKeywords:
            raise streamexceptions.MissingKeyError(f'Missing keyword keys. Required keys are {mandatoryKeywords}.')
        if not hasAllCols:
            raise streamexceptions.MissingKeyError(f'Missing column keys. Required keys are {mandatoryCols}.')
        self.metric = metric
        self.sep = sep
        self.import_survey()
        self.group_by_name()

    def import_survey(self):
        """
        Grabs the survey from a csv.
        """
        self.data = pd.read_csv(self.file, sep=self.sep)

    def pack_shots(self):
        """
        Packs each row in self.data into a Shot object and returns an array
        """
        packed = [Shot(shotLine, self.colRelations, self.keywords) for shotLine in self.data.itertuples()]
        return packed

    def filter_shots(self, packedShots, value, key, allMatch=False):
        """
        Filters a list of packed shots by the 'type' key in the meaning attribute

        Args:
            packedShots: a list of packed shots
            key: the key in the meaning dict for a packed shot to filter by.
            value: the values to filter for. Can be a single value or a list of values.
                   Valid key:value pairs are
                       'type':'Profile' or 'type':'Cross Section'
                       'name':<name that exists in the survey>
                       'morphs':<morph shorthand in survey, such as 'ri' or 'tob'>
            allMatch: True if all values must be in the shot to match. Otherwise
                      shots with any matching values will be returned.
        """
        #print(f'Filtering. value:{value},key:{key},allMatch:{allMatch}')
        if allMatch:
            gen = all
        else:
            gen = any
        if not isinstance(value, list):
            value = [value]
        if key == 'morphs': # works when trying to match a value in morphs, which is a list
            result = [pack for pack in packedShots if gen(i in pack.meaning[key] for i in value)]
        elif key == 'name':
            result = [pack for pack in packedShots if pack.meaning[key] == value[0]]
        elif key == 'type': # else need to use a different expression
            result = [pack for pack in packedShots if gen(i in [pack.meaning[key]] for i in value)]
        return result

    def count_names(self, packedShots):
        """
        Takes a list of packed shots in and returns a dict relating names to count.
        """
        names = [shot.meaning['name'] for shot in packedShots]
        counter = {}
        for name in names:
            try:
                counter[name] += 1
            except KeyError:
                counter[name] = 1
        return counter

    def get_names(self):
        """
        Returns a dict with two keys ('Profiles' and 'Cross Sections') where
        each key points to a dict relating a name to a counter that indicates
        how many times that name appears in the survey.
        """
        packedAndSeparated = self.pack_and_separate()
        theDict = {'Profiles':None, 'Cross Sections':None}
        for i,key in enumerate(theDict):
            theDict[key] = self.count_names(packedAndSeparated[i])

        return theDict

    def pack_and_separate(self):
        """
        Packs self.data and separates the profile shots from cross section shots, returning
        profiles and crossSections.
        """
        packed = self.pack_shots()
        profiles = self.filter_shots(packed, 'Profile', 'type')
        crossSections = self.filter_shots(packed, 'Cross Section', 'type')
        return(profiles,crossSections)

    def group_by_name(self):
        """
        Takes self.data, packs and separates in and then makes two lists (one each for
        cross sections and profiles) where each list contains lists of shots with the same name
        in the order that they appears in self.data. These lists are accessible with
        self.profiles and self.crossSections.
        """
        bulkProAndCross = self.pack_and_separate()
        proAndCross = [[],[]]

        for i,shotGroup in enumerate(bulkProAndCross):
            nameDict = self.count_names(shotGroup)
            names = nameDict.keys()
            for name in names:
                proAndCross[i].append(self.filter_shots(shotGroup, name, 'name'))

        self.profiles, self.crossSections = proAndCross

    def get_cross_objects(self, guessType=True, project=True, stripName=False):
        """
        Takes self.crossSections and returns a list of CrossSection objects.
            If guessType is True, the method will attempt to guess the morph type
                (Riffle,Run,Pool,Glide) for each CrossSection.
            If project is true, the CrossSections will use projected stationing.
        """
        crosses = [PackGroupCross(packGroup,
                                  self.keywords,
                                  self.metric,
                                  stripName=stripName).create_cross_object(guessType, project)
                   for packGroup in self.crossSections]
        return crosses

    def get_profile_objects(self, stripName=False):
        """
        Takes self.crossSections and returns a list of Profile objects.
        """
        profiles = [PackGroupPro(packGroup,
                                 self.keywords,
                                 self.colRelations,
                                 self.metric,
                                 stripName=stripName).create_pro_object(assignMethod='backstack')
                    for packGroup in self.profiles]
        return profiles

    def get_packgroup_coords(self,packGroup):
        """
        Takes a list of packed shots and returns a list of tuples containing the x,y coords for each shot.
        """
        coords = [[shot.ex, shot.why] for shot in packGroup]
        return coords


class PackGroupPro():
    """
    A profile represented by a list of packed shots.
    """
    def __init__(self, packGroup, keywords, colRelations, metric=False, stripName=False):
        """
        Args:
            packGroup: a list of packed shots representing a profile.
            metric: bool indicating if the profile uses metric or imperial units
            keywords: dict relating standardized names to shot keywords
            colrelations: a dict relating direction, elevation and description names to how they appear
                          in the raw data
        """
        self.packGroup = packGroup
        self.metric = metric
        self.keywords = keywords
        self.colRelations = colRelations
        self.name = self.packGroup[0].meaning['name']
        if stripName:
            self.name = self.name.replace(self.keywords['Profile'], '')

        self.make_uCols()
        self.make_sCols()

    def substrate_filter(self):
        """
        Filters the packgroup to return only substrate (thalweg, riffle, run, pool, glide) shots.
        """
        subNames = ['Thalweg', 'Riffle', 'Run', 'Pool', 'Glide']
        subShots = StreamSurvey.filter_shots(StreamSurvey, self.packGroup, subNames, 'morphs')

        return subShots

    def make_uCols(self):
        """
        Unique cols.
        """
        uCols = list(self.keywords.keys())
        remove = ['breakChar', 'commentChar', 'Profile']
        uCols = [col for col in uCols if col not in remove]

        return uCols

    def make_sCols(self):
        """
        Standardized cols.
        """
        sCols = list(self.colRelations.keys())

        return sCols

    def backstack_data(self):
        colnames = self.make_sCols()
        colnames.extend(self.make_uCols())

        allSubs = self.substrate_filter()
        blankCol = [None for i in allSubs]
        listCol = [[] for i in allSubs]
        backStacked = {col:blankCol.copy() for col in colnames}
        backStacked['desc'] = listCol
        i = -1
        for shot in self.packGroup:
            mean = shot.meaning['morphs']
            if any(morph in mean for morph in ['Thalweg', 'Riffle', 'Run', 'Pool', 'Glide']): # if it's a substrate shot
                i += 1
                backStacked['Thalweg'][i] = shot.zee
                backStacked['exes'][i] = shot.ex
                backStacked['whys'][i] = shot.why
                backStacked['zees'][i] = shot.zee
                backStacked['shotnum'][i] = shot.shotnum
            for col in mean:
                if i < 0:
                    logging.warning('Non-substrate shots before first substrate shot. These will not be backstacked.')
                    break
                backStacked[col][i] = shot.zee
                backStacked['desc'][i].append(shot.desc)
        backStacked = pd.DataFrame.from_dict(backStacked)
        return backStacked

    def create_pro_object(self, assignMethod='backstack'):
        """
        Makes a pyfluv profile object

        Args:
            packGroup: a list of packed profile shots.
            assignMethod: 'backstack' or 'nearest'.
                           If 'backstack', non-substrate profile shots will be assigned the station
                               of the previous substrate shot.
                           If 'nearest', they will be assigned the station of the nearest substrate shot.
        """
        if assignMethod == 'nearest':
            raise NotImplementedError('Nearest method not implemented yet.')
        elif assignMethod == 'backstack':
            df = self.backstack_data()
        return sp.Profile(df, name=self.name, metric=self.metric)


class PackGroupCross():
    """
    A cross section represented by a list of packed shots.
    """

    def __init__(self, packGroup, keywords, metric=False, stripName=False):
        """
        Args:
            packGroup: a list of packed shots representing a profile.
            keyWords: a dictionary relating full morph names to morph keywords
        """
        self.packGroup = packGroup
        self.keywords = keywords
        self.metric = metric
        self.name = self.packGroup[0].meaning['name']
        if stripName:
            self.name = self.name.replace(self.keywords['Cross Section'], '')

    def pull_atts(self): # was pull_xs_packgroup_atts
        """
        Takes a list of packed XS shots, all with the same name, and returns a
        dictionary representing water surface, bankfull, and top of bank elevations
        and thalweg coordinates if specified by survey keywords.
        """
        # all keys will be matched with an elevation (z) except Thalweg, which is given an x,y pair
        attributes = {'Water Surface':None,
                      'Bankfull':None,
                      'Top of Bank':None,
                      'Thalweg':None}
        for att in attributes:
            # 4 possibilities:
            # the attribute does not exist in the keywords dict
            # it does but the keyword doesn't show up in the packgroup so we turn the value back to None
            # it is in the dict and in the shotgroup and is unique
            # it is in dict and shotgroup and isn't unique so we average the results
            matchShots = StreamSurvey.filter_shots(StreamSurvey, self.packGroup, att, 'morphs')
            if matchShots == []:
                pass
            else:
                if att == 'Thalweg':
                    val = [[shot.ex,shot.why] for shot in matchShots]
                    attributes[att] = list(np.mean(val, axis=0))
                else:
                    val = [shot.zee for shot in matchShots]
                    attributes[att] = np.mean(val)
        return attributes

    def pull_xs_survey_coords(self):
        """
        Takes a list of packed XS shots, all with the same name,
        and returns 4 lists: exes, whys, zees and accompanying shot descs.
        """
        exes,whys,zees,descs = [],[],[],[]
        for shot in self.packGroup:
            exes.append(shot.ex)
            whys.append(shot.why)
            zees.append(shot.zee)
            descs.append(shot.desc)

        return(exes, whys, zees, descs)

    def get_cross_morph(self):
        """
        Takes a packed XS shot and guesses if it's a Ri, Ru, Po, Gl or None
        based off of keywords.
        """

        morphNames = ['Riffle', 'Run', 'Pool', 'Glide']
        reverseKeys = {}
        for morph in morphNames:
            try:
                reverseKeys[self.keywords[morph]] = morph
            except KeyError:
                pass

        morphType = None
        for key in reverseKeys:
            if key in self.name:
                morphType = reverseKeys[key]

        return morphType

    def create_cross_object(self, guessType=True, project=True):
        """
        Takes a group of packed shots representing a single cross section
        and returns a CrossSection object.
        """
        if guessType:
            morphType = self.get_cross_morph()
        else:
            morphType = None

        exes,whys,zees,desc = self.pull_xs_survey_coords()
        attDict = self.pull_atts()
        df = {'exes':exes, 'whys':whys, 'zees':zees, 'desc':desc}
        # have not implemented adding the cross section thalweg
        cross = sg.CrossSection(df=df,
                                name=self.name,
                                morphType=morphType,
                                metric=self.metric,
                                project=project,
                                bkfEl=attDict['Bankfull'],
                                tobEl=attDict['Top of Bank'],
                                wsEl=attDict['Water Surface'])
        return cross


class Parser():
    """
    Parses desc strings.
    """
    def __init__(self, parseDict):
        self.parseDict = parseDict

    def dict_split(self, string):
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

        return result

    def string_is_in(self, matchString, string):
        """
        Returns true if matchString is in string.
        """
        contained = re.search(matchString, string)
        return bool(contained)

    def key_is_in(self, key, string):
        return self.string_is_in(self.parseDict[key], string)

    def get_meaning(self, string):
        """
        Gets the semantic meaning of the dictionary returned by self.dict_split.
        """
        result = {'type':None, # profile or cross section
                  'morphs':[], # depends on if type is profile or cross section
                  'name':None,
                  'full':string
                 }
        splitDict = self.dict_split(string)
        result['name'] = splitDict['name']

        if self.key_is_in('Profile', result['name']):
            result['type'] = 'Profile'
        elif self.key_is_in('Cross Section', result['name']):
            result['type'] = 'Cross Section'

        for descriptor in splitDict['descriptors']:
            for key,pattern in self.parseDict.items():
                if self.string_is_in(pattern, descriptor):
                    result['morphs'].append(key)

        return result


class Shot():
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
    def __init__(self, shotLine, colRelations, parseDict):
        """
        Args:
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
        return f'<shot object:{self.shotnum},{self.desc}>'

    def __repr__(self):
        return f'<shot object:{self.shotnum},{self.desc}>'

    def set_shotnum(self):
        self.shotnum = getattr(self.shotLine, self.colRelations['shotnum'])

    def set_ex(self):
        self.ex = getattr(self.shotLine, self.colRelations['exes'])

    def set_why(self):
        self.why = getattr(self.shotLine, self.colRelations['whys'])

    def set_zee(self):
        self.zee = getattr(self.shotLine, self.colRelations['zees'])

    def set_desc(self):
        self.desc = getattr(self.shotLine, self.colRelations['desc'])

    def set_meaning(self):
        parsed = Parser(self.parseDict)
        meaning = parsed.get_meaning(self.desc)
        self.meaning = meaning
