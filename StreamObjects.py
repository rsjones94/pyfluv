# a collection of objects that, together, model a stream's XS, profile and planform from total station survey data

def CrossSection():
    
    def __init__(self, name = None, exes, whys, bkfEl = None):
        self.name = name # the name of the cross section
        self.exes = exes # the stationing (x, or "ex") values of the total station survey
        self.whys = whys # the elevation (y, or "why") values of the total statoin survey
        self.bkfEl = bkfEl # the elevation of the bankfull at the cross section
        
        # need to check to make sure exes and whys are lists of equal length with only numbers
        
    def setBankfullEl():
        """sets the elevation of the stream's bankfull and recalculate dependent variables
        """
        pass
    
    def calculateArea():
        """calculates the area under a given elevation. Only calculates area in primary channel
        (as defined by min el) by default
        """
       pass
    
    def calculateWettedPerimeter():
        """calculates the wetted perimeter under a given elevation
        """
        pass
    
    def calculateHydraulicRadius():
        """calculates the hydraulic radius given an elevation
        """
        pass
        
    def calculateFloodproneEl():
        """calculates the elevation of the floodprone area. This elevation is twice the bkf elevation by default
        """
        pass
        
    def calculateFloodproneWidth():
        """calculates the width of the floodprone area
        """
        pass
        
    def calculateMeanDepth():
        """calculates the mean depth given a certain elevation
        """
        pass
    
    def calculateMaxDepth():
        """calculates the max depth given a certain elevation
        """
        pass

    def calculateWidth():
        """calculates the bankfull width given a certain elevation
        """
        pass
        
    def calculateFlow():
        """calculates the volumetric flow given a certain elevation, ws slope and manning's N
        """
        pass
    
    def bkfByFlowRelease():
        """estimates the bankfull elevation by finding the elevation where the rate of flow release (dq/dh) is maximized
        """
        pass