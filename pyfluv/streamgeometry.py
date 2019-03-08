"""
Contains the CrossSection class, which stores and processes stream geometry (cross sections).
"""
import functools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import streamexceptions
from . import streamconstants as sc
from . import streammath as sm

class CrossSection(object):
    
    """
    A generic geomorphic cross section.`
        Lengths are expressed in terms of meters or feet.
        Time is expressed in terms of seconds.
        Mass is express in terms of kilograms or slugs.
        Forces are expressed in terms of newtons or pounds.
    
    Attributes:
        name(str): the name of the XS
        metric(bool): whether the survey units are feet (False) or meters (True)
        df(dict or Pandas dataframe): dict or dataframe with keys/columns 'exes', 'whys', 'zees' and optional 'desc'
        rawSta(float): the stationing of the cross section
        rawEl(float): the elevations of the cross section
        stations(float): the stationing of the cross section with overhangs removed (may be equivalent to rawSta)
        elevations(float): the elevations of the cross section with overhangs removed (may be equivalent to rawEl)
        bStations(float): the stationing of the channel that is filled at bkf
        bElevations(float): the elevations corresponding to bStations
        morphType(string): Denotes the morphological type of the XS - 'Ri', 'Ru', 'Po', 'Gl'
        bkfEl(float): the bankfull elevation at the XS
        wsEl(float): the elevation of the water surface
        thwStation(float): the station of the thalweg
        thwIndex(int): the index of the thalweg in stations
        waterSlope(float): dimensionless slope of the water surface at the cross section
        project(bool): whether the stationing should be calculated along the XS's centerline (True) or not (False)
        hasOverhangs(bool): whether or not overhangs are present in the raw survey
        fillFraction(float): a float between 0 or 1 that specifies how overhangs are to be removed.
            0 indicates that the overhangs will be cut, 1 indicates they will be filled, and 
            intermediate values are some mix of cut and fill (intermediate values not yet supported).
        manN(float): manning's N
        sizeDist(GrainDistribution): an object of the class GrainDistribution
        unitDict(dict): a dictionary of unit values and conversion ratios; values depend on value of self.metric
        boundTruths(dict): a dictionary that stores whether an attribute (such as bkfW) is exact or represents a minimum
        """
    
    def __init__(self, df, name = None, morphType = None, metric = False, manN = None, 
                 waterSlope = None, project = True, bkfEl = None, wsEl = None, tobEl = None, 
                 thwStation = None, sizeDist = None, fillFraction = 1):
        """
        Method to initialize a CrossSection.
        
        Args:
            df: a dict or pandas dataframe with columns/keys 'exes','whys','zees' and optional 'desc'
            name: the name of the XS
            morphType: One of 'Riffle', 'Run', 'Pool', 'Glide' or None
            metric: whether the survey units are feet (False) or meters (True)
            manN: manning's N at the cross section
            waterSlope: the slope of the water at the cross section at the bankfull elevation
            project: whether the stationing should be calculated along the XS's centerline (True) or not (False)
            bkfEl: the bankfull elevation at the XS
            wsEl: the water surface elevation at the time of survey
            tobEl: the elevation of the top of bank
            thwStation: the station of the thalweg. If not specified, the deepest point in the given XS is assumed.
                        If the XS cuts across multiple channels or the channel is raised, this assumption may not be correct.
                        However unless you are certain you do not want to use the deepest surveyed point as the thw
                        it is suggested that this parameter is left unspecified.
            sizeDist: an object of type GrainDistribution which represents a pebble count, sieve or other grain distribution
                      analysis at the cross section.
            fillFraction: float between 0 or 1 that specifies how overhangs are to be removed.
                          0 indicates that the overhangs will be cut, 1 indicates they will be filled
                          and intermediate values are some mix of cut and fill.
                
        Raises:
            GeometryError: If the geometry of the cross section is not simple (non self-intersecting)
            ShapeAgreementError: If any of exes, whys or zees don't have the same length as the others
                                 or desc is specified but does not match the length of exes, whys and zees
        """
        
        self.name = name
        if isinstance(df,dict):
            df = pd.DataFrame.from_dict(df)
        self.df = df
        self.morphType = morphType
        self.project = project
        self.sizeDist = sizeDist
        self.metric = metric
        if self.metric:
            self.unitDict = sc.METRIC_CONSTANTS
        elif not(self.metric):
            self.unitDict = sc.IMPERIAL_CONSTANTS
        self.thwStation = thwStation
        self.manN = manN
        self.waterSlope = waterSlope
        self.fillFraction = fillFraction

        self.hasOverhangs = False # may be altered when _validate_geometry() is called
        
        self._create_2d_form()
        self._validate_geometry()
        
        self._check_sta_and_el()
        self._set_thw_index()
        
        self.wsEl = wsEl
        self.tobEl = tobEl
        self.bkfEl = bkfEl
        
    def __str__(self):
        """
        Prints the name of the CrossSection object. If the name attribute is None, prints "UNNAMED".
        """
        if self.name:
            return(self.name)
        else:
            return("UNNAMED")

    def _bkf_savestate(func):
        """
        To be used as a decorator for functions that alter bkfEl internally.
        Saves the bkfEl at the time of the function call, the restores it and calculates stats
        when the function exits.
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            saveEl = self.bkfEl
            try:
                result = func(self, *args, **kwargs)
            finally: # we need to clean up even if the function fails
                self.bkfEl = saveEl
            return(result)
        return(wrapper)
        
    # use the property decorator to make it so setting bkf also recalculates the bkf stations/elevations
    @property
    def bkfEl(self):
        return self._bkfEl

    @bkfEl.setter
    def bkfEl(self,value = None):
        self._bkfEl = value
        self._set_bankfull_stations_and_elevations()
        self._determine_bounding_truths()
    
    def qplot(self, labelPlot = True, ve=None, showBkf=True, showWs = True,
              showTob = True, showFloodEl = True, showCutSection=False):
        """
        Uses matplotlib to create a quick plot of the cross section.
        If showCutSection is True but no overhangs are present, no removed overhangs will be shown.
        """
        ax = plt.subplot()
        if showCutSection and self.hasOverhangs:
            plt.plot(self.rawSta,self.rawEl, "b--", color="#f44e42", linewidth = 2, label = 'Overhang')
            
        plt.plot(self.stations,self.elevations, color="black", linewidth = 2)
        plt.scatter(self.stations,self.elevations, color="black")
        
        # in retrospect, this probably should have been done with a loop and a truth dict
                
        if showFloodEl and self.bkfEl:
            broken = sm.break_at_bankfull(self.stations,
                                          self.elevations,
                                          self.flood_prone_elevation()
                                          ,self.thwIndex)
            exes = [broken[0][0],broken[0][-1]]
            whys = [broken[1][0],broken[1][-1]]
            plt.plot(exes,whys, color="#06AA00", linewidth = 2, label = 'Floodprone Elevation')
            plt.scatter(exes,whys, color="#06AA00")
            
        if showTob and self.tobEl:
            broken = sm.break_at_bankfull(self.stations,
                                          self.elevations,
                                          self.tobEl,
                                          self.thwIndex)
            exes = [broken[0][0],broken[0][-1]]
            whys = [broken[1][0],broken[1][-1]]
            plt.plot(exes,whys, color="#FFBD10", linewidth = 2, label = 'Top of Bank') 
            plt.scatter(exes,whys, color="#FFBD10")
            
        if showBkf and self.bkfEl:
            broken = sm.break_at_bankfull(self.stations,
                                          self.elevations,
                                          self.bkfEl,
                                          self.thwIndex)
            exes = [broken[0][0],broken[0][-1]]
            whys = [broken[1][0],broken[1][-1]]
            plt.plot(exes,whys, color="#FF0000", linewidth = 2, label = 'Bankfull')
            plt.scatter(exes,whys, color="#FF0000")
        
        if showWs and self.wsEl:
            broken = sm.break_at_bankfull(self.stations,
                                          self.elevations,
                                          self.wsEl,
                                          self.thwIndex)
            exes = [broken[0][0],broken[0][-1]]
            whys = [broken[1][0],broken[1][-1]]
            plt.plot(exes,whys, "b--", color = '#31A9FF', linewidth = 2, label = 'Water Surface')
            plt.scatter(exes,whys, color="#31A9FF")
        if ve is not None:
            ax.set_aspect(ve)
        if labelPlot:
            plt.title(str(self))
            plt.xlabel('Station (' + self.unitDict['lengthUnit'] + ')')
            plt.ylabel('Elevation (' + self.unitDict['lengthUnit'] + ')')
            plt.legend()
            
    def planplot(self, labelPlot = True, showProjections = True, equalAspect=True):
        """
        Uses matplotlib to create a quick plot of the planform of the cross
        section. If showProjections is True but self.project is False, no
        projects will be shown.
        
        Args:
            showProjections: If True, shows the where each shot was projected
            to.
        """
        ax = plt.subplot()
        plt.plot(self.df['exes'],self.df['whys'],
                 label = 'Cross Section Planform')
        plt.ticklabel_format(useOffset=False)
        if showProjections and self.project:
            projected = self._get_centerline_shots()
            projX = projected[0]
            projY = projected[1]
            plt.scatter(projX,projY)
            for i in range(len(projX)):
                px = (self.df['exes'][i],projX[i])
                py = (self.df['whys'][i],projY[i])
                plt.plot(px,py)
        if equalAspect:
            ax.set_aspect('equal')
        if labelPlot:
            plt.title(str(self) + ' (Planform)')
            plt.xlabel('Easting (' + self.unitDict['lengthUnit'] + ')')
            plt.ylabel('Northing (' + self.unitDict['lengthUnit'] + ')')
            plt.legend()
                
    def _get_centerline_shots(self):
        """
        Returns two lists representing the projection of the original exes and
        whys onto the centerline
        
        Args:
            None
        
        Returns:
            A tuple of lists.
            
        Raises:
            None
        """
        return(sm.centerline_series(self.df['exes'],self.df['whys']))
        
    def _create_2d_form(self):
        """
        Uses the survey x,y,z data to create stationing and elevation data.
            Defines rawSta and rawEl, representing stationing and elevation.
        """
        self.rawSta = sm.get_stationing(self.df['exes'],self.df['whys'],project = self.project)
        self.df['Station'] = self.rawSta
        self.rawEl = self.df['zees']
        if isinstance(self.rawEl,pd.core.series.Series):
            self.rawEl = self.rawEl.tolist()
    
    def _validate_geometry(self):
        """
        Checks if a cross section is self-intersecting (always illegal) and if it has 
            overhangs (okay, but changes data processing).
        """
        noOverhangs = not(sm.monotonic_increasing(self.rawSta))
        if noOverhangs:
            self.hasOverhangs = True
            logging.warning(f'Overhangs present in geometry on XS {self.name}.')
        
        simplicity = sm.is_simple(self.rawSta,self.rawEl)
        if not(simplicity[0]): # if the geometry is not simple
            raise streamexceptions.GeometryError('Error: geometry is self-intersecting on segments ' + str(simplicity[1]) + ' and ' + str(simplicity[2]))
    
    def _set_thw_index(self):
        """
        Finds the index of the thw in stations. If user didn't specify thwSta, then we guess it by finding the index of
        the minimum elevation in the channel. If this value is not unique, the leftmost is selected.
        """
        if not(self.thwStation): # if the user didn't specify this, we need to guess it. If the min value is not unique, the leftmost value is used.
            self.thwIndex = sm.find_min_index(self.elevations)
        else:
            if self.thwStation < self.stations[0] or self.thwStation > self.stations[len(self.stations)-1]:
                logging.warning('Thalweg station specified is out of surveyed bounds. Guessing thalweg station.')
                self.thwIndex = sm.find_min_index(self.elevations)
            else:
                first = sm.get_nth_closest_index_by_value(self.stations,self.thwStation,1)
                self.thwIndex = first
                """
                second = sm.get_nth_closest_index_by_value(self.stations,self.thwStation,2)
                # we want to find the two closest points to the station specified and pick the lowest one
                if self.elevations[first] <= self.elevations[second]:
                    self.thwIndex = first
                else:
                    self.thwIndex = second
                """
    
    def _set_bankfull_stations_and_elevations(self):
        """
        Sets bStations and bElevations.
        """
        if self.bkfEl:
            if self.bkfEl <= min(self.elevations):
                raise streamexceptions.PhysicsLogicError('Bankfull elevation is at or below XS bottom.')
            if self.elevations[self.thwIndex] >= self.bkfEl:
                raise streamexceptions.PhysicsLogicError('Thw index (' + str(self.thwIndex) + ') is at or above bankfull elevation.')
            
            broken = sm.break_at_bankfull(self.stations,self.elevations,self.bkfEl,self.thwIndex)
            self.bStations = broken[0]
            self.bElevations = broken[1]
        else:
            self.bStations = None
            self.bElevations = None
    
    def _check_sta_and_el(self):
        """
        Checks the raw sta and el to make sure there are no overhangs. If there are, removes them.
            Either way this method sets self.sta and self.el.
        """
        
        if self.fillFraction == 1:
            method = 'fill'
        elif self.fillFraction == 0:
            method = 'cut'
        
        if not(self.hasOverhangs):
            self.stations = self.rawSta
            self.elevations = self.rawEl
        elif self.hasOverhangs:
            try:
                removed = sm.remove_overhangs(self.rawSta,self.rawEl,method=method,adjustY=True) # remove_overhangs will change soon; this will need to be updated
                self.stations = removed[0]
                self.elevations = removed[1]
            except IndexError:
                """
                This will get thrown when self.hasOverhangs is True, but remove_overhangs can't 
                actually find any due to the points causing the overhang to be unusually close together.
                When this happens, the numerical error in calculations due to leaving the points in
                will be small, so we will just pass the raw stations and elevations.
                HOWEVER, this is a bug and should be fixed in the future.
                """
                logging.warning(f'Overhangs detected but could not be removed on XS {self.name}.')
                self.stations = self.rawSta
                self.elevations = self.rawEl
                
    def _determine_bounding_truths(self):
        """
        Creates a dictionary that stores whether width-centered attributes are bounded on each side.
        There are only two keys - bkfW and floodproneWidth - and they point to a tuple (bool,bool) that
        indicates if the left and right side respectively are bound.
        
        If bkfW is unbounded on either side, then all values based on bkfEl are
        minima with the exception of the entrenchment ratio which is neither a minimum nor a maximum.
        
        If floodproneWidth is unbounded on either side, then floodprone width, floodprone elevation 
        and entrenchment ratio all represent minima unless bkfW is unbounded, in which case the
        entrenchment ratio is neither a minimum nor a maximum.
        
        If bkfEl is None, the values in the dict will also be done.
        """
        leftMax = max(self.elevations[:self.thwIndex+1])
        rightMax = max(self.elevations[self.thwIndex:])
        
        boundDict = {'bkfWidth':None,'floodproneWidth':None}
        if self.bkfEl:
            fpEl = self.flood_prone_elevation()
            boundDict['bkfWidth'] = (self.bkfEl<=leftMax,self.bkfEl<=rightMax)
            boundDict['floodproneWidth'] = (fpEl<=leftMax,fpEl<=rightMax)
        self.boundTruths = boundDict
    
    def area(self):
        """
        Calculates the area under a given elevation. Only calculates area in
        primary channel (as defined by min el) by default.
        """
        return(sm.get_area(self.bStations,self.bElevations))
    
    def wetted_perimeter(self):
        """
        Calculates the wetted perimeter under a given elevation.
        """
        segmentLengths = []
        for i in range(0,len(self.bStations)-1):
            p1 = (self.bStations[i],self.bElevations[i])
            p2 = (self.bStations[i+1],self.bElevations[i+1])
            length = sm.length_of_segment((p1,p2))
            segmentLengths.append(length)
        return(sum(segmentLengths))
    
    def hydraulic_radius(self):
        """
        Calculates the hydraulic radius given an elevation.
        """
        return(self.area() / self.wetted_perimeter())
    
    def shear_stress(self):
        """
        Calculates the shear stress at bkf. If metric, units are N/m^2.
        If imperial, units are lbs/ft^2
        """
        return(self.unitDict['gammaWater']*self.mean_depth()*self.waterSlope)
        
    def shear_velocity(self):
        """
        Calculates the shear velocity at bkf.
        """
        return((self.unitDict['g']*self.hydraulic_radius()*self.waterSlope)**(1/2))
        
    def stream_power(self):
        """
        Calculates the stream power at bkf.
        """
        return(self.unitDict['gammaWater']*self.discharge_rate()*self.waterSlope*self.width())
        
    def threshhold_particle(self):
        """
        Calculates the diameter of the biggest particles that could be entrained at the bankfull flow.
        """
        raise NotImplementedError('This method has not yet been implemented.')
        
    def flood_prone_elevation(self):
        """
        Calculates the elevation of the floodprone area. This elevation is at twice the bkf max depth by default.
        """
        return(min(self.bElevations) + 2*self.max_depth())
        
    def flood_prone_width(self):
        """
        Calculates the width of the floodprone area.
        """
        broken = sm.break_at_bankfull(self.stations,
                                      self.elevations,
                                      self.flood_prone_elevation(),
                                      self.thwIndex)
        floodSta = broken[0]
        return(sm.max_width(floodSta))
        
    def calculate_entrenchment_ratio(self):
        """
        Calculates the entrenchment ratio - the flood prone width divided by the bankfull width
        """
        return(self.flood_prone_width() / self.width())
    
    def bank_height_ratio(self):
        """
        The height of the top of bank above the channel thalweg divided by the 
        height of bankfull above the thalweg.
        """
        minEl = min(self.bElevations)
        bkfHeight = self.bkfEl - minEl
        tobHeight = self.tobEl - minEl
        return(tobHeight / bkfHeight)
    
    def mean_depth(self):
        """
        Calculates the mean depth given a certain elevation.
        """
        return(sm.get_mean_depth(self.bStations,self.bElevations,self.bkfEl,True))
        
    def max_depth(self):
        """
        Calculates the max depth given a certain elevation.
        """
        return(sm.max_depth(self.bElevations,self.bkfEl))

    def width(self):
        """
        Calculates the bankfull width given a certain elevation.
        """
        return(sm.max_width(self.bStations))
        
    def flow_velocity(self):
        """
        Calculates the bkf discharge velocity, given a bkf elevation, ws slope and manning's n.
        Units are ft/s or meters/s.
        """
        manNum = self.unitDict['manningsNumerator']
        return((manNum/self.manN)*self.hydraulic_radius()**(2/3)*self.waterSlope**(1/2))
    
    def discharge_rate(self):
        """
        Calculates the volumetric flow given a bkf elevation, ws slope and manning's n.
        Units are cubic ft/s or cubic meters/s.
        """
        return(self.area()*self.flow_velocity())
            
    def widthdepth_ratio(self):
        """
        Calculates the ratio of the bankfull width to the mean bankfull depth.
        """
        return(self.width() / self.mean_depth())
        
    def froude(self):
        """
        Calculates the Froude number at the cross section.
        """
        return(self.flow_velocity()/(self.unitDict['g']*self.mean_depth())**(1/2))
    
    @_bkf_savestate
    def attribute_list(self, attributeMethod, deltaEl = 0.1):
        """
        Returns two arrays: a list of elevations and a corresponding list of the channel attribute if bkf
        were at that elevation.
        
        Args:
            deltaEl: the granularity of the change in elevation by which the area will be calculated.
            attribute: a string that references an attribute such as bkfW or bkfA

        Returns:
            Two lists; a list of elevations and a list that relates that elevation to the desired statistic
            
        Raises:
            None.
        """
        elArray = []
        attrArray = []
        
        minEl = self.elevations[self.thwIndex]
        self.bkfEl = minEl + deltaEl
        
        while self.bkfEl <= max(self.elevations):
            elArray.append(self.bkfEl)
            attrArray.append(attributeMethod())
            self.bkfEl += deltaEl
        
        return(elArray,attrArray)
      
    @_bkf_savestate
    def get_attr(self,attributeMethod,elevation):
        """
        Returns an attribute (that is a function of bkf elevation) at a given bkf elevation.
        
        Deprecated.
        """
        self.bkfEl = elevation
        at = attributeMethod()
        return(at)
    
    def _attr_nthderiv(self,attributeMethod,n,elevation,delta = 0.01):
        """
        This is the same as attr_nthderiv(), but does not have the @_bkf_savestate decorator to save
        on computation time when called multiple times. Does not save the bankfull state.
        """
        checkList, change = sm.build_deriv_exes(elevation,n,delta)
        atList = []
        for el in checkList:
            self.bkfEl = el
            at = attributeMethod()
            atList.append(at)
        result = sm.diffreduce(atList,change)
        return(result)
      
    @_bkf_savestate
    def attr_nthderiv(self,attributeMethod,n,elevation,delta = 0.01):
        """
        Finds the central nth numerical derivative of an attribute with respect to elevation at a given elevation.
        
        Args:
            attribute: the attribute to find the derivative for.
            n: the order of the derivative.
            elevation: the elevation to find the derivative at.
            delta: the granularity of the change in elevation by which the derivative will be calculated.

        Returns:
            The elevation where d(dA/dh) is maximized.
            
        Raises:
            None.
        """
        result = self._attr_nthderiv(attributeMethod,n,elevation,delta)
        return(result)
        
    @_bkf_savestate
    def find_floodplain_elevation(self, attribute = 'width', returns = 'lower', delta = None):
        """
        Estimates the elevation of the floodplain by maximizing a target function that is evaluated
        at each possible survey elevation.
        
        Args:
            attribute: the attribute used to find flow relief. Can be 'area' or 'width' (preferred)
                If bkfA, the target function is the third derivative of bankfull area with respect to bankfull elevation.
                If bkfW, the target function is the second derivative of bankfull width with respect to bankfull elevation.
                Note that bkfW seems to return the expected result more reliably as the derivative is uncentered
                    for derivatives with an odd order.
            returns: which floodplain value to return.
                'left' - the left floodplain elevation is returned
                'right' - the right floodplain elevation is returned
                'lower' - the lower of the left and right floodplains is returned
                'upper' - the higher of the left and right floodplains is returned
                'min' - the floodplain where less flow release is returned
                'max' - the floodplain where more flow release is returned
                'mean' - the mean floodplain elevation is returned            
            deltaEl: the granularity of the change in elevation by which the area will be calculated.
                Note that any points within delta/2 of the thalweg will not be evaluated
                as the target function would not be able to be evaluated. If no delta is specified,
                then delta will be set to 10% of the difference between the lowest and highest points
                in the survey.
        Returns:
            The elevation where the target function is maximized.
            
        Raises:
            InputError: if the arguments passed to the method or attribute parameters are invalid.   
        
        TODO: Right now the whole cross section is tested for flow release.
        However if eg there are two points on opposing sides with similar els,
        The same floodplain could be returned even if each side has a different
        true fp el.
        """
        if delta is None:
            delta = (max(self.elevations)-min(self.elevations))*0.1
        
        if returns not in ['lower','upper','min','max','left','right','mean']:
            raise streamexceptions.InputError("Invalid method. Method must be one of 'lower','upper','left','right','mean'.")
        
        if attribute == 'area':
            attributeMethod = self.area
            deriv = 3
            logging.warn('Floodplain estimation by the third derivative of bkf area is unstable.')
        elif attribute == 'width':
            attributeMethod = self.width
            deriv = 2
        else:
            raise streamexceptions.InputError("Invalid attribute. Attribute must be 'area' or 'width'.")
        
        leftEls = sm.make_monotonic(self.elevations[self.thwIndex-1::-1],removeDuplicates=True)
        rightEls = sm.make_monotonic(self.elevations[self.thwIndex+1:],removeDuplicates=True)
        
        els = [None,None]
        # we need to filter out elevations within delta/2 of the thalweg elevation
        els[0] = [el for el in leftEls if el > (self.elevations[self.thwIndex] + delta/2)]
        els[1] = [el for el in rightEls if el > (self.elevations[self.thwIndex] + delta/2)]
        
        funcResults = [None,None]
        for i,side in enumerate(els):
            funcResults[i] = [self._attr_nthderiv(attributeMethod,deriv,el,delta) for el in side]
        maxes = [max(funcResults[0]),max(funcResults[1])]
        inds = [sm.find_max_index(funcResults[0]),sm.find_max_index(funcResults[1])]
        winEls = [els[0][inds[0]],els[1][inds[1]]]
        
        maxSide = sm.find_max_index(maxes) # 0 for left, 1 for right
        minSide = maxSide^1 # flips the bit
        
        highSide = sm.find_max_index(winEls) # 0 for left, 1 for right
        lowSide = highSide^1
        
        resultDict = {'min':winEls[minSide],'max':winEls[maxSide],'left':winEls[0],
                      'right':winEls[1],'lower':winEls[lowSide],'upper':winEls[highSide],
                      'mean':np.mean(winEls)}
        return(resultDict[returns])
       
    @_bkf_savestate
    def bkf_brute_search(self, attributeMethod, target, delta = 0.1):
        """
        Finds the most ideal bkf elevation by performing a brute force search,
        looking for a target value of a specified attribute. The attribute need
        not increase monotonically with bkf elevation. After exiting the
        algorithm, bankfull statistics will be recalculated for whatever the
        bkfEl was when entering the method. The algorithm only checks betwee
        the thw elevation and the maximum surveyed elevation.
        
        Args:
            attribute: a string that references an attribute such as bkfW that 
                       is dependent on bkf el.
            target: the ideal value of attribute.
            delta: the elevation interval between statistics calculations
            epsilon: the desired maximum absolute deviation from the target
                     attribute.
            terminateOnSufficient: a boolean indicating if the first result
                                   within the tolerance should be returned
        
        Returns:
            The ideal bkf elevation.
            
        Raises:
            None.
        """
        
        arrays = self.attribute_list(attributeMethod = attributeMethod, 
                                     deltaEl = delta)
        elevations = arrays[0]
        attributes = np.asarray(arrays[1])
        dists = np.abs(attributes-target)
        
        bestIndex = sm.find_min_index(dists)
        
        return(elevations[bestIndex])
    
    @_bkf_savestate
    def bkf_binary_search(self,
                          attributeMethod,
                          target,
                          epsilon = None,
                          returnFailed = False):
        """
        Finds the most ideal bkf elevation by performing a binary-esque search,
        looking for a target value of a specified attribute. This runs much
        quicker than bkf_brute_search() but is restricted to attributes that
        increase monotonically with bkfEl.
        
        Args:
            attributeMethod: a method that calculates an attribute that is
                             MONOTONICALLY dependent on bkf el. Results are not
                             guaranteed to be accurate if the function that
                             relates the attribute to bkf elevation is not
                             monotonic increasing.
            target: the ideal value of attribute.
            epsilon: the maximum acceptable absolute deviation from the target
                     attribute.
                
        Returns:
            The ideal bkf elevation.
            
        Raises:
            None.
        """
        
        if epsilon is None:
            epsilon = target/1000 
            # by default the tolerance is 0.1% of the target value.
        
        bottom = min(self.elevations)
        top = max(self.elevations)
        
        if self.thwStation:
            thwEl = self.elevations[self.thwIndex]
            if thwEl > bottom:
                bottom = thwEl        
        """
        The above nested if is meant to handle when a secondary channel
        contains the thw. But if the thwInd indicates a point in the main
        channel that is NOT the true thw then this will cause the algorithm to
        start with an incorrectly high bottom.
        """
        
        found = False
        foundUpperBound = False
        n = 0
        
        while not found and n < 1000:
            n += 1
            self.bkfEl = (bottom + top)/2
            calculatedValue = attributeMethod()
            if np.isclose(calculatedValue,target,atol=epsilon):
                found = True
            else:
                if calculatedValue > target:
                    # if we have overestimated the bkf el
                    top = self.bkfEl
                    foundUpperBound = True
                else: # if we have underestimated the bkf el
                    bottom = self.bkfEl
                    if not foundUpperBound:
                        top = top * 2
                        # in case the target cannot be found within the
                        # confinements of the surveyed channel
                        if top >= max(self.elevations)*10**2:
                            print('Target too great for channel ' + str(self) +
                                  '. Breaking.')
                            break
        
        foundEl = self.bkfEl # save the best result we found       
        
        if found:
            print('Converged in ' + str(n) + ' iterations.')
            return(foundEl)
        else:
            print('Could not converge in ' + str(n) + ' iterations.')
            if returnFailed:
                return(foundEl)
            else:
                return(None)
