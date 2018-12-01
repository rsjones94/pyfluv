"""
Contains the CrossSection class, which stores and processes stream geometry (cross sections)

"""
import warnings

import streammath as sm


class CrossSection(object):
    
    """
    A generic geomorphic cross section.
    
    Attributes:
        name(str): the name of the XS
        exes(:obj:'list' of :obj:'float'): the raw x vals of the cross section (planform)
        whys(:obj:'list' of :obj:'float'): the raw y vals of the cross sections (planform)
        zees(:obj:'list' of :obj:'float'): the raw z vals of the cross section (elevation)
        raw2dX(float): the stationing of the cross section
        raw2dY(float): the elevations of the cross section
        bkfEl(float): the bankfull elevation at the XS
        hasOverhangs(bool): whether or not overhangs are present in the raw survey

    """
    
    def __init__(self, exes, whys, zees, name = None, bkfEl = None):
        self.name = name
        self.exes = exes
        self.whys = whys
        self.zees = zees
        self.bkfEl = bkfEl
        self.hasOverhangs = False
        
        self.create_2d_form()
        self.validate_geometry()
        
        # need to check to make sure exes and whys are lists of equal length with only numbers
        
    def create_2d_form(self,project=True):
        """
        Uses the survey x,y,z data to create stationing and elevation data.
        
        Args:
            project: a boolean telling whether or not to project x,y data onto a centerline
            
        Returns:
            A tuple of two lists, one representing stationing and another representing elevation.
            
        Raises:
            None.
        """
        pass
    
    def validate_geometry(self):
        """
        Checks if a cross section is self-intersecting (always illegal) and if it has overhangs (okay, but changes data processing)
        """
        x = self.raw2dX
        y = self.raw2dY
        
        overhangs = sm.get_cuts(x,y,'overhang') # list of points that are overhangs
        if overhangs: # pythonic way to check if a list is not empty
            self.hasOverhangs = True
            warnings.warn('Overhangs present in geometry')
        
        simplicity = sm.is_simple(x,y)
        if not(simplicity[0]): # if the geometry is not simple
            raise Exception('Error: geometry is self-intersecting on segments ' + str(simplicity[1]) + ' and ' + str(simplicity[2]))
    
    def set_bankfull_elevation():
        """
        Sets the elevation of the stream's bankfull and recalculate dependent variables
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
