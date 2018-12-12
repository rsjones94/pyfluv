"""
Contains the CrossSection class, which stores and processes stream geometry (cross sections)

"""
import logging

import streammath as sm


class CrossSection(object):
    
    """
    A generic geomorphic cross section.
        Spatial variables are expressed in terms of feet or meters
        Time variables are expressed in terms of seconds
    
    Attributes:
        name(str): the name of the XS
        metric(bool): whether the survey units are feet (False) or meters (True)
        exes(:obj:'list' of :obj:'float'): the surveyed x vals of the cross section (planform)
        whys(:obj:'list' of :obj:'float'): the surveyed y vals of the cross sections (planform)
        zees(:obj:'list' of :obj:'float'): the surveyed z vals of the cross section (elevation)
        rawSta(float): the stationing of the cross section
        rawEl(float): the elevations of the cross section
        sta(float): the stationing of the cross section with overhangs removed (may be equivalent to rawSta)
        el(float): the elevations of the cross section with overhangs removed (may be equivalent to rawEl)
        bkfEl(float): the bankfull elevation at the XS
        project(bool): whether the stationing should be calculated along the XS's centerline (True)
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
        sizeDist(GrainDistribution): an object of the class GrainSizeDistribution
    """
    
    def __init__(self, exes, whys, zees, name = None, metric = False, project = True, bkfEl = None, fillFraction = 1):
        self.name = name
        self.exes = exes
        self.whys = whys
        self.zees = zees
        self.project = project
        self.bkfEl = bkfEl
        self.hasOverhangs = False
        
        self.create_2d_form()
        self.validate_geometry()
        self.check_sta_and_el()
        
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
    
    def check_sta_and_el(self):
        """
        Checks the raw sta and el to make sure there are no overhangs. If there are, removes them.
            Either way this method sets self.sta and self.el
        """
        if not(self.hasOverhangs):
            self.sta = self.rawSta
            self.el = self.rawEl
        elif self.overHangs:
            removed = sm.remove_overhangs(self.rawSta,self.rawEl,adjustY=True) # remove_overhangs will change soon; this will need to be updated
            self.sta = removed[0]
            self.el = removed[1]
    
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
