"""
Contains the CrossSection class, which stores and processes stream geometry (cross sections)

"""
import warnings

import mathtools as mt


class CrossSection(object):
    
    def __init__(self, exes, whys, name = None, bkfEl = None, xsType = None):
        self.name = name # the name of the cross section
        self.exes = exes # the stationing (x, or "ex") values of the total station survey
        self.whys = whys # the elevation (y, or "why") values of the total statoin survey
        self.bkfEl = bkfEl # the elevation of the bankfull at the cross section
        self.xsType = xsType
        self.hasOverhangs = False
        
        self.validate_geometry()
        
        # need to check to make sure exes and whys are lists of equal length with only numbers
    
    def validate_geometry(self):
        """
        Checks if a cross section is self-intersecting (always illegal) and if it has overhangs (okay, but changes data processing)
        """
        x = self.exes
        y = self.whys
        
        overhangs = mt.get_cuts(x,y,'overhang') # list of points that are overhangs
        if overhangs: # pythonic way to check if a list is not empty
            self.hasOverhangs = True
            warnings.warn('Overhangs present in geometry')
        
        simplicity = mt.is_simple(x,y)
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
