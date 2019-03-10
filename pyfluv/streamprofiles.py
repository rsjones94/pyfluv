"""
Contains the Profile class and helper classes.
"""

import functools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import streamexceptions
from . import streamconstants as sc
from . import streammath as sm

class Profile(object):
    """
    A longitudinal stream profile.
    
    Attributes:
        df
        filldf
        metric
        name
        unitDict
        features
    """
    
    basicCols = ['exes','whys','Thalweg']
    fillCols = ['Water Surface', 'Bankfull', 'Top of Bank']
    morphCols = ['Riffle','Run','Pool','Glide','Unclassified']
    substrateCols = ['Riffle','Run','Pool','Glide']
    
    def __init__(self, df, name = None, metric = False):
        """
        Args:
            df: a dict or pandas dataframe with at least three columns/keys "exes", "whys", "Thalweg"
                and additional optional columns/keys. Standardized col/key names are 
                "Water Surface", "Bankfull", "Top of Bank", "Riffle", "Run", "Pool", "Glide", "Unclassified".
                If df is passed as a dict, it will be coerced to a Pandas dataframe.
            metric: a bool indicating if units are feet (False) or meters (True)
        
        Raises:
            x
        """
        if isinstance(df,dict):
            df = pd.DataFrame.from_dict(df)
        self.df = df.copy()
        self.filldf = df.copy()
        self.metric = metric
        self.name = name
        self.haveCols = [col for col in self.substrateCols if col in self.filldf]
        
        if self.metric:
            self.unitDict = sc.METRIC_CONSTANTS
        elif not(self.metric):
            self.unitDict = sc.IMPERIAL_CONSTANTS
            
        if not 'Station' in self.df: 
            """
            if there is no stationing column, generate it and interpolate the cols
            actualy a janky way to keep this code from running when the subclass Feature is instantiated
            """
            self.validate_df()
            self.update_filldf()
            self.validate_substrate()
            self.create_features()
        
    def validate_df(self):
        if not all(x in self.df.keys() for x in self.basicCols):
            raise streamexceptions.InputError('Input df must include keys or columns "exes", "whys", "Thalweg"')
    
    def validate_substrate(self):
        """
        For all columns Riffle, Run, Pool, Glide:
            if an index has a value, then at least one of the next or previous indices must have a value.
            no more than two of the columns can have a value on a given index.
        """
        for i in range(len(self.filldf)):
            # check for overpopulation
            row = self.filldf[i:i+1][self.haveCols]
            populated = [colName for colName in row.keys() if not pd.isnull(row[colName][i])]
            if len(populated) > 2:
                raise streamexceptions.InputError(f'More than two substrate features identified on row {i}: {populated} in Profile {self.name}.')
            
            # check for isolation
            for col in self.haveCols:
                if not pd.isnull(self.df[col][i]):
                    neighbors = [False,False]
                    try:
                        neighbors[0] = not(pd.isnull(self.filldf[col][i-1]))
                    except KeyError:
                        pass
                    try:
                        neighbors[1] = not(pd.isnull(self.filldf[col][i+1]))
                    except KeyError:
                        pass
                    if all(neighbor is False for neighbor in neighbors):
                        logging.warning(f'Isolated {col} call on row {i} in Profile {self.name}. Call will be deleted.')
                        pd.options.mode.chained_assignment = None
                        self.df[col][i] = np.NaN
                        self.filldf[col][i] = np.NaN
                        pd.options.mode.chained_assignment = 'warn'
                        
    def __str__(self):
        """
        Prints the name of the Profile object. If the name attribute is None, prints "UNNAMED".
        """
        if self.name:
            return(self.name)
        else:
            return("UNNAMED")
        
    def qplot(self, labelPlot = True, ve = None, showThw = True, showWs = True,
              showBkf = True, showTob = True, showFeatures = False):
        
        ax = plt.subplot()
        if showThw:
            plt.plot(self.filldf['Station'],self.filldf['Thalweg'], color = 'gray', linewidth = 2, label = 'Thalweg')
        
        if 'Water Surface' in self.filldf and showWs:
            if not self.filldf['Water Surface'].isnull().all():
                plt.plot(self.filldf['Station'],self.filldf['Water Surface'], "b--",
                         color = '#31A9FF', linewidth = 2, label = 'Water Surface')
                     
        if 'Bankfull' in self.filldf and showBkf:
            if not self.filldf['Bankfull'].isnull().all():
                plt.plot(self.filldf['Station'],self.filldf['Bankfull'],
                         color = '#FF0000', linewidth = 2, label = 'Bankfull')
                     
        if 'Top of Bank' in self.filldf and showTob:
            if not self.filldf['Top of Bank'].isnull().all():
                plt.plot(self.filldf['Station'],self.filldf['Top of Bank'], "b--",
                         color = '#FFBD10', linewidth = 2, label = 'Top of Bank')
                     
        if showFeatures:
            for feature in self.ordered_features():
                feature.addplot(addLabel=False)
           
        if ve is not None:
            ax.set_aspect(ve)
            
        if labelPlot:
            plt.title(str(self))
            plt.xlabel('Station (' + self.unitDict['lengthUnit'] + ')')
            plt.ylabel('Elevation (' + self.unitDict['lengthUnit'] + ')')
            
            handles,labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels,handles))
            plt.legend(by_label.values(),by_label.keys())
    
    def planplot(self,labelPlot = True,equalAspect=True):
        """
        Uses matplotlib to create a quick plot of the planform of the profile.
        """
        ax = plt.subplot()
        plt.plot(self.df['exes'],self.df['whys'],label = 'Profile Planform')
        plt.ticklabel_format(useOffset=False)
        if equalAspect:
            ax.set_aspect('equal')
        if labelPlot:
            plt.title(str(self) + ' (Planform)')
            plt.xlabel('Easting (' + self.unitDict['lengthUnit'] + ')')
            plt.ylabel('Northing (' + self.unitDict['lengthUnit'] + ')')
            plt.legend()
            
    def trend(self,col,order=1,updateLegend=True):
        """
        Adds a trendline to a plot.
        """
        x = np.linspace(self.filldf['Station'].iloc[0],self.filldf['Station'].iloc[-1]
                           ,self.length())
        eq = self.fit(col,order)
        y = np.polyval(eq,x)
        
        if order == 1:
            addon = ', Linear Fit'
        elif order == 0:
            addon = ', Constant Fit'
        else:
            addon = f', Polynomial Fit (Order {order})'
        plt.plot(x,y,linewidth=1,label=col+addon)
        
        if updateLegend:
            handles,labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels,handles))
            plt.legend(by_label.values(),by_label.keys())
            
    def _make_poly_string(self,poly):
        """
        Takes a list coefs in descending order and makes a string representing
        the equation.
        """
        poly = list(poly)
        poly.reverse()
        eq = 'y ='
        order = 0
        for coef in poly:
            space = ' '
            if coef >= 0:
                func = '+'
            else:
                func = '-'
            num = str(round(np.abs(coef),2))
            if order > 1:
                ex = f'x^{order}'
            elif order == 1:
                ex = 'x'
            elif order == 0:
                ex = ''
                func = ''
                space = ''
            
            appender = f'{space}{func} {num}{ex}'
            eq = eq+appender
            order += 1
        return(eq)

    def generate_stationing(self):
        stations = sm.get_stationing(self.df['exes'],self.df['whys'],project = False)
        self.filldf['Station'] = stations
        
    def fill_name(self,name):
        """
        Given a column name/key, interpolates all missing values.
        """
        result = sm.interpolate_series(self.filldf['Station'],self.filldf[name])
        return(result)
        
    def fill_columns(self):
        """
        Interpolates missing values in columns with names contained in fillCols
        """
        for col in self.fillCols:
            if col in self.filldf:
                self.filldf[col] = self.fill_name(col)
                
    def update_filldf(self):
        self.generate_stationing()
        self.fill_columns()
        self.create_diffcols()
        
    def create_diffcols(self):
        """
        Creates columns of interest that are the difference of two other columns.
        """
        # interesting diffs is a dict where the keys are the name of the column to be and
        # the values are the two columns that are used to create it (order matters)
        interestingDiffs = {'Water Depth':['Water Surface','Thalweg'],
                            'Bankfull Height':['Bankfull','Thalweg'],
                            'Bankfull to Water':['Bankfull','Water Surface'],
                            'Top of Bank Height':['Top of Bank','Thalweg']
                            }
        for key,value in interestingDiffs.items():
            try:
                self.filldf[key] = self.create_diff(value[0],value[1])
            except KeyError:
                next
        
    def create_features(self):
        """
        Creates Feature objects based off of the morphology calls in self.filldf.
        
        This code is gross and needs to be refactored.
        """
        featDict = {morph:[] for morph in self.haveCols}
        currentMorph = None
        startInd = 0
        for i in range(len(self.filldf)):
            try:
                if pd.isnull(self.filldf[currentMorph][i]):
                    #print(f'Gap switch {currentMorph} at {i-1}')
                    feat = Feature(df=self.filldf[startInd:i],
                                   name=None,
                                   metric=self.metric,
                                   morphType=currentMorph)
                    featDict[currentMorph].append(feat)
                    currentMorph = None
                    startInd = -1
                    for morph in self.haveCols:
                        if not(pd.isnull(self.filldf[morph][i])):
                            currentMorph = morph
                            startInd = i
                            break
                else:
                    for morph in self.haveCols:
                        if not(pd.isnull(self.filldf[morph][i])) and morph is not currentMorph:
                            #print(f'Smooth switch {currentMorph} to {morph} at {i}')
                            feat = Feature(df=self.filldf[startInd:i+1],
                                           name=None,
                                           metric=self.metric,
                                           morphType=currentMorph)
                            featDict[currentMorph].append(feat)
                            currentMorph = morph
                            startInd = i
                            break
            except ValueError: # if currentMorph is None
                #print(f'Nonetype detected')
                for morph in self.haveCols:
                    if not(pd.isnull(self.filldf[morph][i])):
                        currentMorph = morph
                        startInd = i
                        #print(f'New morph: {currentMorph}')
                        break
            #print(f'-----At {i}. Current morph: {currentMorph}-----')
            
            if i is len(self.filldf)-1 and not(pd.isnull(currentMorph)):
                feat = Feature(df=self.filldf[startInd:i+1],
                               name=None,
                               metric=self.metric,
                               morphType=currentMorph)
                featDict[currentMorph].append(feat)
                
        self.features = featDict
        self._make_unclassified()
        #as of right now calling in this order means the Unclassified column
        #does not show up in Features. Need to fix this
        
    def ordered_features(self):
        """
        Returns self.features as a list sorted by start index
        """
        orderedFeats = []
        for key in self.features:
            feats = [feat for feat in self.features[key]]
            orderedFeats.extend(feats)
        
        orderedFeats = sorted(orderedFeats,key = lambda x:x.indices[0])
        return(orderedFeats)
                        
    def _make_unclassified(self):
        """
        Populates self.features with a key populated by unclassified features.
        """
        self.features['Unclassified'] = []
        oFeats = self.ordered_features()
        if not oFeats: # if there are no features
            feat = Feature(df=self.filldf[0:len(self.filldf)],
                                          name=None,
                                          metric=self.metric,
                                          morphType='Unclassified')
            self.features['Unclassified'].append(feat)
            return(None)
            
        if oFeats[0].indices[0] != 0:
            #print('unclass start')
            feat = Feature(df=self.filldf[0:(oFeats[0].indices[0]+1)],
                                          name=None,
                                          metric=self.metric,
                                          morphType='Unclassified')
            self.features['Unclassified'].append(feat)
        for i,_ in enumerate(oFeats):
            try:
                i1 = oFeats[i].indices[-1]
                i2 = oFeats[i+1].indices[0]
                if i1 != i2:
                    #print(f'gap found: {i1} to {i2}')
                    feat = Feature(df=self.filldf[i1:i2+1],
                                   name=None,
                                   metric=self.metric,
                                   morphType='Unclassified')
                    self.features['Unclassified'].append(feat)
            except IndexError:
                if oFeats[i].indices[-1] is not len(self.filldf)-1:
                    #print('unclass end')
                    feat = Feature(df=self.filldf[oFeats[i].indices[-1]:len(self.filldf)],
                                   name=None,
                                   metric=self.metric,
                                   morphType='Unclassified')
                    self.features['Unclassified'].append(feat)
                    
        self._make_unclassified_column()
                    
    def _make_unclassified_column(self):
        unclassed = [np.nan for i in range(len(self.filldf))]
        for feat in self.features['Unclassified']:
            for i in feat.indices:
                unclassed[i] = self.filldf['Thalweg'][i]
        self.filldf['Unclassified'] = unclassed
        
    def create_diff(self,c1,c2):
        """
        Takes two df columns and returns a columns of the difference at each index.
        """
        return(self.filldf[c1]-self.filldf[c2])
        
    def repair_flow_direction(self,method = 'raise'):
        """
        Makes sure the water surface slope is always 0 or negative.
        
        If method is 'raise', then water surface points are raised up until the slope is 0.
        If method is 'lower', then water surface points are lowered until the slope is 0.
        """
        pd.options.mode.chained_assignment = None
        controlDict = {'raise':reversed(range(len(self.filldf['Water Surface']))),
                       'lower':range(len(self.filldf['Water Surface']))}
        
        for i in controlDict[method]:
            try:
                if self.filldf['Water Surface'].iloc[i] < self.filldf['Water Surface'].iloc[i+1]:
                    if method == 'raise':
                        self.filldf['Water Surface'].iloc[i] = self.filldf['Water Surface'].iloc[i+1]
                    elif method == 'lower':
                        self.filldf['Water Surface'].iloc[i+1] = self.filldf['Water Surface'].iloc[i]
            except IndexError:
                pass
        pd.options.mode.chained_assignment = 'warn'
            
    def force_water_above_thalweg(self):
        """
        Does what is says.
        """
        pd.options.mode.chained_assignment = None
        for i,_ in enumerate(self.filldf['Water Surface']):
            if self.filldf['Water Surface'].iloc[i] < self.filldf['Thalweg'].iloc[i]:
                self.filldf['Water Surface'].iloc[i] = self.filldf['Thalweg'].iloc[i]
        pd.options.mode.chained_assignment = 'warn'
        
    def length(self):
        return(self.filldf['Station'].iloc[-1] - self.filldf['Station'].iloc[0])
        
    def mean_slope(self,col):
        return((self.filldf[col].iloc[-1] - self.filldf[col].iloc[0])/self.length())
    
    def sinuosity(self):
        start = (self.filldf['exes'].iloc[0],self.filldf['whys'].iloc[0])
        end = (self.filldf['exes'].iloc[-1],self.filldf['whys'].iloc[-1])
        straightLength = sm.segment_length(start,end)
        return(self.length()/straightLength)
        
    def deepest(self,deepType):
        """
        Returns the station, index and depth or elevation
        (depending on deepType) of the deepest point as a tuple.
        
        Args:
            deepType: Specifies how the deepest point should be calculated.
                      Can be 'Water Depth', 'Bankfull Height', or 'Thalweg'.
        """
        if deepType == 'Thalweg':
            ind = self.filldf['Station'].idxmin()
        else:
            ind = self.filldf['Station'].idxmax()
            
        return(self.filldf['Station'][ind],ind,self.filldf[deepType][ind])
        
    def _deepsta(self,deepType):
        """
        Wrapper for deepest(). Just returns the station.
        """
        return(self.deepest(deepType)[0])
        
    def spacing(self,featureType,spacingFrom='deepest',spacingTo='deepest',
                deepType='Water Depth'):
        """
        Returns a list of the spacing between each specified feature.
        
        Args:
            featureType: a string specifying the type of feature for which
                         spacing is to be calculated. Can be 'Riffle', 'Run',
                         'Pool', 'Glide' or 'Unclassified'
            spacingFrom: the spot in the feature that spacing should be
                         calculated from. Can be 'start', 'end', 'middle', or
                         'deepest'.
            spacingTo: the spot in the feature that spacing should be
                         calculated to. Can be 'start', 'end', 'middle', or
                         'deepest'.
            deepType: if either spacingFrom or spacingTo is 'deepest', this
                      must be specified. Specifies how the deepest point should
                      be calculated. Can be 'Water Depth', 'Bankfull Height',
                      or 'Thalweg'.
                      
        Returns:
            a list of spacings between each feature.
        """        
        spacings = []
        length = len(self.features[featureType])
        for i,feature in enumerate(self.features[featureType]):
            if i == length-1:
                break
            sta1 = self.features[featureType][i]._feature_measurepoint(measureType=spacingFrom,
                                                     deepType=deepType)
            sta2 = self.features[featureType][i+1]._feature_measurepoint(measureType=spacingTo,
                                                     deepType=deepType)
            spacings.append(sta2-sta1)
        return(spacings)
        
    def fit(self,col,order=1):
        """
        Takes a column in filldf and finds the best polynomial regression
        against Station.
        
        Args:
            col: a string that exists as a column name in filldf
            order: the order of the polynomial regression. For a linear
            regression, set order to 1. Noninteger inputs are floored.
            
        Returns:
            numpy array representing the equation of the regression. The last
            entry is the intercept, and each previous entry is a coefficient
            of increasing order. There will be order+1 entries.
        """
        x = self.filldf['Station']
        y = self.filldf[col]
        return(np.polyfit(x,y,order))
        
    def _reclassify_feature(self,feature,newMorph):
        """
        Reclassifies a Feature and change filldf to reflect this change. Note
        that this does not reclassify *all* features. For example, if a run
        that directly follows a riffle is changed to a riffle, then there will
        be an end riffle concurrent with a begin riffle. To rectify this, call
        self.create_features(). Additionally, self.features will no longer
        contain the correct features under the correct key and they will not
        necessarily be in order by station and the filldf in the Feature will
        not be updated.
        """
        morphCheck = {morph:morph for morph in self.morphCols} # validate morph
        oldMorph = feature.morphType
        feature.morphType = morphCheck[newMorph]
        for i in feature.indices:
            self.filldf[oldMorph].iloc[i] = np.NaN
            self.filldf[newMorph].iloc[i] = self.filldf['Thalweg'][i]
            # THIS GENERATES A SETTINGWITHCOPY WARNING. FIX
            
    def _segment(self,i):
        """
        Returns a 2d line segment representing the connection between
        the ith and i+1th shot in filldf.
        """
        row1 = self.filldf.iloc[i,:]
        row2 = self.filldf.iloc[i+1,:]
        start = (row1['exes'],row1['whys'])
        end = (row2['exes'],row2['whys'])
        return(start,end)
    
    def _xsind(self,CrossSection):
        """
        Returns that index that a cross section crosses at. If the
        CrossSection does not intersect, returns None.
        
        Right now just returns index of segment where it crosses. CrossSection
        may cross closer to i+1 though. Need to add test for this.
        """
        crossseg = CrossSection._crossseg()
        for index,row in self.filldf.iterrows():
            try:
                segment = self._segment(index)
                if sm.does_intersect(segment,crossseg) or sm.does_intersect(crossseg,segment):
                    return(index)
            except IndexError:
                pass
                    
    def xssta(self,CrossSection):
        """
        Returns that station that a cross section crosses at. If the
        CrossSection does not intersect, returns None.
        """
        ind = self._xsind(CrossSection)
        if ind is None:
            return(None)
        else:
            return(self.filldf.loc[ind,'Station'])
        
    def make_elevations_agree(self,colName):
        """
        Makes the corresponding col value in a row agree with the thalweg value in that row.
        """
        pass
    
    def insert_shot(self):
        pass
    
    def modify_shot(self):
        pass
    
    
class Feature(Profile):
    """
    A subsection of a longitudinal stream profile representing a distinct morphological substrate feature.
     
    Attributes:
        x
    """
    
    morphColors = {'Unclassified':'black',
                   'Riffle':'red',
                   'Run':'yellow',
                   'Pool':'blue',
                   'Glide':'green'}
    
    def __init__(self, df, name = None, metric = False, morphType = None):
        Profile.__init__(self, df, name, metric = False)
        self.morphType = morphType
        self.indices = self.filldf.index
        
    def __str__(self):
        return(f"<Feature-{self.morphType}-{self.indices[0]}:{self.indices[-1]}>")
        
    def __repr__(self):
        return(self.__str__())
        
    def addplot(self,addLabel=False):
        """
        Adds scatter points and lines representing the feature to a plot.
        """
        if self.morphType in self.filldf:
            col = self.morphType
            label = self.morphType
        else:
            col = 'Thalweg'
            label = 'Unclassified'
        plt.plot(self.filldf['Station'],self.filldf[col],
                 color = self.morphColors[self.morphType], linewidth = 2,label='_NOLABEL_')
        plt.scatter(self.filldf['Station'],self.filldf[col],
                    color = self.morphColors[self.morphType],label=label)
        
        # code block below updates the legend, but ignores the label if it's a duplicate
        if addLabel:
            handles,labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels,handles))
            plt.legend(by_label.values(),by_label.keys())
            
    def _feature_measurepoint(self,measureType,deepType=None):
        """
        Gets the station that a feature should be measured from/to for spacing
        calculations.
        """
        measureDict = {'start':sm.get_first,
                       'end':sm.get_last,
                       'middle':sm.get_middle
                       }
        if measureType != 'deepest':
            return(measureDict[measureType](self.filldf['Station']))
        elif measureType == 'deepest':
            return(self._deepsta(deepType=deepType))
            