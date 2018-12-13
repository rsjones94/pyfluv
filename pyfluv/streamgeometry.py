"""
Contains the CrossSection class, which stores and processes stream geometry (cross sections)

"""
import logging

import matplotlib.pyplot as plt
import numpy as np

import streammath as sm


class CrossSection(object):
    
    """
    A generic geomorphic cross section.
        Spatial variables are expressed in terms of feet or meters
        Time variables are expressed in terms of seconds
    
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
        bkfEl(float): the bankfull elevation at the XS
        thwStation(float): the station of the thalweg
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
        floodProneEl(float): the flood prone elevation
        floodProneWidth(float): the width of the flood prone area
        manN(float): manning's N
        sizeDist(GrainSizeDistribution): an object of the class GrainSizeDistribution
    """
    
    def __init__(self, exes, whys, zees, name = None, metric = False, project = True, bkfEl = None, thwStation = None, fillFraction = 1):
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
        self.thwStation = thwStation
        self.fillFraction = fillFraction
        self.bkfEl = bkfEl
        
        self.hasOverhangs = False
        
        self.create_2d_form()
        self.validate_geometry()
        self.check_sta_and_el()
        if self.bkfEl: # if a bankful elevation has been specified
            self.set_bankfull_stations_and_elevation()
        
    def __str__(self):
        """
        Prints the name of the CrossSection object. If the name attribute is None, prints "UNNAMED"
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
        plt.plot(self.stations,self.elevations)
        
        if showBkf:
            plt.scatter(self.bStations,self.bElevations)
            bkfExes = [self.bStations[0],self.bStations[len(self.bStations)-1]]
            bkfWhys = [self.bElevations[0],self.bElevations[len(self.bStations)-1]]
            plt.plot(bkfExes,bkfWhys)
            
        if showCutSection:
            plt.plot(self.rawSta,self.rawEl)
    
    def planplot(self, showProjections = True):
        """
        Uses matplotlib to create a quick plot of the planform of the cross section.
        
        Args:
            showProjections: If True, shows the where each shot was projected to
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
            overhangs (okay, but changes data processing)
        """
        
        noOverhangs = not(sm.monotonic_increasing(self.rawSta))
        if noOverhangs:
            self.hasOverhangs = True
            logging.warning('Overhangs present in geometry')
        
        simplicity = sm.is_simple(self.rawSta,self.rawEl)
        if not(simplicity[0]): # if the geometry is not simple
            raise Exception('Error: geometry is self-intersecting on segments ' + str(simplicity[1]) + ' and ' + str(simplicity[2]))
    
    def set_bankfull_stations_and_elevation(self):
        """
        Sets bStations and bElevations.
        """
        if not(self.thwStation): # if the user didn't specify this, we need to guess it. If the min value is not unique, the leftmost value is used.
            thwInd = sm.find_min_index(self.elevations)
        broken = sm.break_at_bankfull(self.stations,self.elevations,self.bkfEl,thwInd)
        self.bStations = broken[0]
        self.bElevations = broken[1]
    
    def check_sta_and_el(self):
        """
        Checks the raw sta and el to make sure there are no overhangs. If there are, removes them.
            Either way this method sets self.sta and self.el
        """
        if not(self.hasOverhangs):
            self.stations = self.rawSta
            self.elevations = self.rawEl
        elif self.hasOverhangs:
            removed = sm.remove_overhangs(self.rawSta,self.rawEl,method='fill',adjustY=True) # remove_overhangs will change soon; this will need to be updated
            self.stations = removed[0]
            self.elevations = removed[1]
    
    def calculate_statistics():
        """
        Recalculate all variables dependent on bkf el
        """
        pass
    
    def calculate_area():
        """
        Calculates the area under a given elevation. Only calculates area in primary channel
        (as defined by min el) by default
        """
        pass
    
    def calculate_wetted_perimeter():
        """
        Calculates the wetted perimeter under a given elevation
        """
        pass
    
    def calculate_hydraulic_radius():
        """
        Calculates the hydraulic radius given an elevation
        """
        pass
        
    def calculate_floodprone_elevation():
        """
        Calculates the elevation of the floodprone area. This elevation is twice the bkf elevation by default
        """
        pass
        
    def calculate_floodprone_width():
        """
        Calculates the width of the floodprone area
        """
        pass
        
    def calculate_mean_depth():
        """
        Calculates the mean depth given a certain elevation
        """
        pass
    
    def calculate_max_depth():
        """
        Calculates the max depth given a certain elevation
        """
        pass

    def calculate_width():
        """
        Calculates the bankfull width given a certain elevation
        """
        pass
        
    def calculate_flow():
        """
        Calculates the volumetric flow given a certain elevation, ws slope and manning's N
        """
        pass
    
    def bkf_by_flow_release():
        """
        Estimates the bankfull elevation by finding the elevation where the rate of flow release (dq/dh) is maximized
        """
        pass
