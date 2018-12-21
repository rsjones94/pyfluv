"""
Contains the CrossSection class, which stores and processes stream geometry (cross sections)

"""
import logging

import matplotlib.pyplot as plt
import numpy as np

import streamconstants as sc
import streammath as sm


class CrossSection(object):
    
    """
    A generic geomorphic cross section.
        Lengths are expressed in terms of feet or meters.
        Time is expressed in terms of seconds.
        Mass is express in terms of kilograms or slugs.
        Weights are express in terms of newtons or pounds.
    
    Attributes:
        name(str): the name of the XS
        metric(bool): whether the survey units are feet (False) or meters (True)
        exes(:obj:'list' of :obj:'float'): the surveyed x (easting or similar) vals of the cross section
        whys(:obj:'list' of :obj:'float'): the surveyed y (northing or similar) vals of the cross sections
        zees(:obj:'list' of :obj:'float'): the surveyed z (elevation) vals of the cross section
        rawSta(float): the stationing of the cross section
        rawEl(float): the elevations of the cross section
        stations(float): the stationing of the cross section with overhangs removed (may be equivalent to rawSta)
        elevations(float): the elevations of the cross section with overhangs removed (may be equivalent to rawEl)
        bStations(float): the stationing of the channel that is filled at bkf
        bElevations(float): the elevations corresponding to bStations
        _bkfEl(float): the bankfull elevation at the XS
        thwStation(float): the station of the thalweg
        thwIndex(int): the index of the thalweg in bSta
        waterSlope(float): dimensionless slope of the water surface at the cross section
        project(bool): whether the stationing should be calculated along the XS's centerline (True) or not (False)
        hasOverhangs(bool): whether or not overhangs are present in the raw survey
        fillFraction(float): a float between 0 or 1 that specifies how overhangs are to be removed.
            0 indicates that the overhangs will be cut, 1 indicates they will be filled, and intermediate values are some mix of cut and fill.
        bkfA(float): the area of the XS at bankfull
        bkfW(float): the width of the XS at the bankfull
        bkfQ(float): the flow rate of the XS at bankfull
        bkfMeanD(float): the mean depth at bankfull
        bkfMaxD(float): the max depth at bankfull
        bkfWetP(float): the wetted perimeter at bankfull
        bkfHydR(float): the hydraulic radius at bankfull
        bkfStress(float): shear stress at bankfull
        entrainedParticleSize(float): diameter of the biggest particles entrained at bankfull
        floodProneEl(float): the flood prone elevation
        floodProneWidth(float): the width of the flood prone area
        manN(float): manning's N
        sizeDist(GrainSizeDistribution): an object of the class GrainSizeDistribution
        unitDict(dict): a dictionary of unit values and conversion ratios; values depend on value of self.metric
        triggerRecalc(bool): True if you wish for statistics to be recalculated every time you reset bkfEl, waterSlope or manN
        """
    
    def __init__(self, exes, whys, zees, name = None, metric = False, manN = None, waterSlope = None, project = True, bkfEl = None, thwStation = None, triggerRecalc = True, fillFraction = 1):
        """
        Method to initialize a CrossSection.
        
        Args:
            exes: the surveyed x (easting or similar) vals of the cross section as a list
            whys: the surveyed y (northing or similar) vals of the cross sections as a list
            zees: the surveyed z (elevation) vals of the cross section as a list
            name: the name of the XS
            metric: whether the survey units are feet (False) or meters (True)
            project: whether the stationing should be calculated along the XS's centerline (True) or not (False)
            bkfEl: the bankfull elevation at the XS
            thwStation: the station of the thalweg. If not specified, the deepest point in the given XS is assumed
                        If the XS cuts across multiple channels or the channel is raised, this assumption may not be correct.
            fillFraction: float between 0 or 1 that specifies how overhangs are to be removed.
                          0 indicates that the overhangs will be cut, 1 indicates they will be filled
                          and intermediate values are some mix of cut and fill.
                
        Raises:
            Exception: If the geometry of the cross section is not simple (non self-intersecting)
        """
        self.name = name
        self.exes = exes.copy()
        self.whys = whys.copy()
        self.zees = zees.copy()
        self.project = project
        self.metric = metric
        if self.metric:
            self.unitDict = sc.METRIC_CONSTANTS
        elif not(self.metric):
            self.unitDict = sc.IMPERIAL_CONSTANTS
        self.thwStation = thwStation
        self.manN = manN
        self.waterSlope = waterSlope
        self.fillFraction = fillFraction
        self.triggerRecalc = triggerRecalc

        self.hasOverhangs = False
        
        self.create_2d_form()
        self.validate_geometry()
        self.check_sta_and_el()
        
        self._bkfEl = bkfEl
        self.calculate_bankfull_statistics() # this calls set_bankfull_stations_and_elevations() within it
    
    def alter_bankfull_properties(self,value,recalculate = None):
        """
        This method sets the bankfull elevation, creates bStations and bElevations and recalculates statistics
        """
        self._bkfEl = value
        if recalculate == None:
            recalculate = self.triggerRecalc
        if recalculate:
            self.set_bankfull_stations_and_elevations()
            self.calculate_bankfull_statistics()
            
    def get_bkfEl(self):
        return(self._bkfEl)
    
    bkfEl = property(fget = get_bkfEl,fset = alter_bankfull_properties) # whenever a user calls (CrossSection).bkfEl, the bkfEl will be set AND statistics recalculated
    
    def __str__(self):
        """
        Prints the name of the CrossSection object. If the name attribute is None, prints "UNNAMED".
        """
        if self.name:
            return(self.name)
        else:
            return("UNNAMED")
    
    def qplot(self,showBkf=False,showCutSection=False):
        """
        Uses matplotlib to create a quick plot of the cross section.
        """
        plt.figure()
        if showCutSection:
            plt.plot(self.rawSta,self.rawEl, color="tomato")
            
        plt.plot(self.stations,self.elevations, color="black")
        
        if showBkf:
            bkfExes = [self.bStations[0],self.bStations[len(self.bStations)-1]]
            bkfWhys = [self.bElevations[0],self.bElevations[len(self.bStations)-1]]
            plt.plot(bkfExes,bkfWhys, color="blue")
            plt.scatter(bkfExes,bkfWhys, color="blue")
            
    def planplot(self, showProjections = True):
        """
        Uses matplotlib to create a quick plot of the planform of the cross section.
        
        Args:
            showProjections: If True, shows the where each shot was projected to.
        """
        plt.figure()
        plt.plot(self.exes,self.whys)
        
        if showProjections:
            projected = self.get_centerline_shots()
            projX = projected[0]
            projY = projected[1]
            plt.scatter(projX,projY)
            for i in range(len(projX)):
                px = (self.exes[i],projX[i])
                py = (self.whys[i],projY[i])
                plt.plot(px,py)
                
    def get_centerline_shots(self):
        """
        Returns two lists representing the projection of the original exes and whys onto the centerline
        
        Args:
            None
        
        Returns:
            A tuple of lists.
            
        Raises:
            None
        """
        return(sm.centerline_series(self.exes,self.whys))
        
    def create_2d_form(self):
        """
        Uses the survey x,y,z data to create stationing and elevation data.
            Defines rawSta and rawEl, representing stationing and elevation.
        """
        self.rawSta = sm.get_stationing(self.exes,self.whys,project = self.project)
        self.rawEl = self.zees
    
    def validate_geometry(self):
        """
        Checks if a cross section is self-intersecting (always illegal) and if it has 
            overhangs (okay, but changes data processing).
        """
        noOverhangs = not(sm.monotonic_increasing(self.rawSta))
        if noOverhangs:
            self.hasOverhangs = True
            logging.warning('Overhangs present in geometry.')
        
        simplicity = sm.is_simple(self.rawSta,self.rawEl)
        if not(simplicity[0]): # if the geometry is not simple
            raise Exception('Error: geometry is self-intersecting on segments ' + str(simplicity[1]) + ' and ' + str(simplicity[2]))
    
    def set_thw_index(self):
        """
        Finds the index of the thw in bSta. If user didn't specify thwSta, then we guess it by finding the index of
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
    
    def set_bankfull_stations_and_elevations(self):
        """
        Sets bStations and bElevations.
        """
        if self._bkfEl:
            if self._bkfEl <= min(self.elevations):
                raise Exception('Bankfull elevation is at or below XS bottom.')
            self.set_thw_index()
            if self.elevations[self.thwIndex] >= self.bkfEl:
                raise Exception('Thw index (' + str(self.thwIndex) + ') is at or above bankfull elevation.')
            broken = sm.break_at_bankfull(self.stations,self.elevations,self._bkfEl,self.thwIndex)
            self.bStations = broken[0]
            self.bElevations = broken[1]
        else:
            self.bStations = None
            self.bElevations = None
    
    def check_sta_and_el(self):
        """
        Checks the raw sta and el to make sure there are no overhangs. If there are, removes them.
            Either way this method sets self.sta and self.el.
        """
        if not(self.hasOverhangs):
            self.stations = self.rawSta
            self.elevations = self.rawEl
        elif self.hasOverhangs:
            removed = sm.remove_overhangs(self.rawSta,self.rawEl,method='fill',adjustY=True) # remove_overhangs will change soon; this will need to be updated
            self.stations = removed[0]
            self.elevations = removed[1]
    
    def calculate_bankfull_statistics(self):
        """
        Recalculate all variables dependent on bkf el
        """
        self.set_bankfull_stations_and_elevations()
        
        if self._bkfEl > self.elevations[0] or self._bkfEl > self.elevations[len(self.elevations)-1]:
            logging.warning('Bankfull elevation exceeds surveyed channel. Some calculated statistics may represent lower bounds.')
        
        self.calculate_area()
        self.calculate_mean_depth()
        self.calculate_max_depth()
        self.calculate_width()
        self.calculate_wetted_perimeter()
        self.calculate_hydraulic_radius()
        self.calculate_shear_stress()
        self.calculate_max_entrained_particle()
        self.calculate_floodprone_elevation()
        self.calculate_floodprone_width()
        self.calculate_flow()
    
    def calculate_area(self):
        """
        Calculates the area under a given elevation. Only calculates area in primary channel
        (as defined by min el) by default.
        """
        if self._bkfEl:
            area = sm.get_area(self.bStations,self.bElevations)
            self.bkfA = area
        else:
            self.bkfA = None
    
    def calculate_wetted_perimeter(self):
        """
        Calculates the wetted perimeter under a given elevation.
        """
        if self._bkfEl:
            segmentLengths = []
            for i in range(0,len(self.bStations)-1):
                p1 = (self.bStations[i],self.bElevations[i])
                p2 = (self.bStations[i+1],self.bElevations[i+1])
                length = sm.length_of_segment((p1,p2))
                segmentLengths.append(length)
            self.bkfWetP = sum(segmentLengths)
        else:
            self.bkfWetP = None
    
    def calculate_hydraulic_radius(self):
        """
        Calculates the hydraulic radius given an elevation.
        """
        if self._bkfEl:
            self.bkfHydR = self.bkfA / self.bkfWetP
        else:
            self.bkfHydR = None
    
    def calculate_shear_stress(self):
        """
        Calculates the shear stress. If metric, units are N/m^2. If imperial, units are lbs/ft^2
        """
        
        if self.waterSlope and self._bkfEl: # if we don't have a waterslope set, we can't calculate this.
            gammaWater = self.unitDict('gammaWater')
            stress = gammaWater * self.bkfMeanD * self.waterSlope
            self.bkfStress = stress
        else:
            self.bkfStress = None
        
    def calculate_max_entrained_particle(self):
        """
        Calculates the diameter of the biggest particles that could be entrained at the bankfull flow.
        """
        pass
        
    def calculate_floodprone_elevation(self):
        """
        Calculates the elevation of the floodprone area. This elevation is twice the bkf elevation by default.
        """
        pass
        
    def calculate_floodprone_width(self):
        """
        Calculates the width of the floodprone area.
        """
        pass
        
    def calculate_mean_depth(self):
        """
        Calculates the mean depth given a certain elevation.
        """
        if self._bkfEl:
            meanDepth = sm.get_mean_depth(self.bStations,self.bElevations,self._bkfEl,True)
            self.bkfMeanD = meanDepth
        else:
            self.bkfMeanD = None
        
    def calculate_max_depth(self):
        """
        Calculates the max depth given a certain elevation.
        """
        if self._bkfEl:
            maxDepth = sm.max_depth(self.bElevations,self._bkfEl)
            self.bkfMaxD = maxDepth
        else:
            self.bkfMaxD = None

    def calculate_width(self):
        """
        Calculates the bankfull width given a certain elevation.
        """
        if self._bkfEl:
            self.bkfW = sm.max_width(self.bStations)
        else:
            self.bkfW = None
        
    def calculate_flow(self):
        """
        Calculates the volumetric flow given a bkf elevation, ws slope and manning's n.
        """
        if self.waterSlope and self.manN and self._bkfEl: # need all of these to calculate this
            manNum = self.unitDict['manningsNumerator']
            flow = (manNum/self.manN)*self.bkfA*self.bkfHydR**(2/3)*self.waterSlope**(1/2)
            self.bkfQ = flow
        else:
            self.bkfQ = None
    
    def bkf_by_flow_release(self):
        """
        Estimates the bankfull elevation by finding the elevation where the rate of flow release (dq/dh) is maximized.
        """
        pass
