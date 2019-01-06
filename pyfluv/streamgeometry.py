"""
Contains the CrossSection class, which stores and processes stream geometry (cross sections).
"""
import logging

import matplotlib.pyplot as plt
import numpy as np

import streamconstants as sc
import streammath as sm


class CrossSection(object):
    
    """
    A generic geomorphic cross section.
        Lengths are expressed in terms of meters or feet.
        Time is expressed in terms of seconds.
        Mass is express in terms of kilograms or slugs.
        Forces are expressed in terms of newtons or pounds.
    
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
        morphType(string): Denotes the morphological type of the XS - 'Ri', 'Ru', 'Po', 'Gl'
        bkfEl(float): the bankfull elevation at the XS
        wsEl(float): the elevation of the water surface
        thwStation(float): the station of the thalweg
        thwIndex(int): the index of the thalweg in stations
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
        sizeDist(GrainDistribution): an object of the class GrainDistribution
        unitDict(dict): a dictionary of unit values and conversion ratios; values depend on value of self.metric
        boundTruths(dict): a dictionary that stores whether an attribute (such as bkfW) is exact or represents a minimum
        """
    
    def __init__(self, exes, whys, zees, name = None, morphType = None, metric = False, manN = None, waterSlope = None, project = True, bkfEl = None, wsEl = None, tobEl = None, thwStation = None, sizeDist = None, fillFraction = 1):
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
            thwStation: the station of the thalweg. If not specified, the deepest point in the given XS is assumed.
                        If the XS cuts across multiple channels or the channel is raised, this assumption may not be correct.
                        However unless you are certain you do not want to use the deepest surveyed point as the thw
                        it is suggested that this parameter is left unspecified.
            fillFraction: float between 0 or 1 that specifies how overhangs are to be removed.
                          0 indicates that the overhangs will be cut, 1 indicates they will be filled
                          and intermediate values are some mix of cut and fill.
                
        Raises:
            Exception: If the geometry of the cross section is not simple (non self-intersecting)
            Exception: If any of exes, whys or zees don't have the same length as the others
        """
        if not(len(exes) == len(whys) == len(zees)):
            raise Exception('exes, whys and zees must all have the same length.')
        
        self.name = name
        self.exes = exes.copy()
        self.whys = whys.copy()
        self.zees = zees.copy()
        self.morphType = morphType
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

        self.hasOverhangs = False # may be altered when validate_geometry() is called
        
        self.create_2d_form()
        self.validate_geometry()
        self.check_sta_and_el()
        self.set_thw_index()
        
        self.wsEl = wsEl
        self.tobEl = tobEl
        self.bkfEl = bkfEl
        self.calculate_bankfull_statistics() # this calls set_bankfull_stations_and_elevations() within it
    
    def __str__(self):
        """
        Prints the name of the CrossSection object. If the name attribute is None, prints "UNNAMED".
        """
        attachDict = {'Ri':', Riffle','Ru':', Run','Po':', Pool','Gl':', Glide',None:''}
        if self.name:
            printname = self.name + attachDict[self.morphType]
            return(printname)
        else:
            return("UNNAMED")
    
    def qplot(self, showBkf=True, showWs = True, showTob = True, showFloodEl = True, showCutSection=False):
        """
        Uses matplotlib to create a quick plot of the cross section.
        """
        plt.figure()
        if showCutSection:
            plt.plot(self.rawSta,self.rawEl, color="tomato", linewidth = 2)
            
        plt.plot(self.stations,self.elevations, color="black", linewidth = 2)
        plt.title(str(self))
        plt.xlabel('Station (' + self.unitDict['lengthUnit'] + ')')
        plt.ylabel('Elevation (' + self.unitDict['lengthUnit'] + ')')
        
        
        if showBkf and self.bkfEl:
            broken = sm.break_at_bankfull(self.stations,self.elevations,self.bkfEl,self.thwIndex)
            exes = [broken[0][0],broken[0][-1]]
            whys = [broken[1][0],broken[1][-1]]
            plt.plot(exes,whys, color="red", linewidth = 2)
            plt.scatter(exes,whys, color="red")
        
        if showWs and self.wsEl:
            broken = sm.break_at_bankfull(self.stations,self.elevations,self.wsEl,self.thwIndex)
            exes = [broken[0][0],broken[0][-1]]
            whys = [broken[1][0],broken[1][-1]]
            plt.plot(exes,whys, "b--", linewidth = 2)
            plt.scatter(exes,whys, color="b")
            
        if showTob and self.tobEl:
            broken = sm.break_at_bankfull(self.stations,self.elevations,self.tobEl,self.thwIndex)
            exes = [broken[0][0],broken[0][-1]]
            whys = [broken[1][0],broken[1][-1]]
            plt.plot(exes,whys, color="magenta", linewidth = 2)
            plt.scatter(exes,whys, color="magenta")
            
        if showFloodEl and self.floodproneEl:
            broken = sm.break_at_bankfull(self.stations,self.elevations,self.floodproneEl,self.thwIndex)
            exes = [broken[0][0],broken[0][-1]]
            whys = [broken[1][0],broken[1][-1]]
            plt.plot(exes,whys, color="green", linewidth = 2)
            plt.scatter(exes,whys, color="green")
            
    def planplot(self, showProjections = True):
        """
        Uses matplotlib to create a quick plot of the planform of the cross section.
        
        Args:
            showProjections: If True, shows the where each shot was projected to.
        """
        plt.figure()
        plt.plot(self.exes,self.whys)
        plt.title(str(self) + ' (Planform)')
        plt.xlabel('Easting (' + self.unitDict['lengthUnit'] + ')')
        plt.ylabel('Northing (' + self.unitDict['lengthUnit'] + ')')
        
        if showProjections and self.project:
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
    
    def set_bankfull_stations_and_elevations(self):
        """
        Sets bStations and bElevations.
        """
        if self.bkfEl:
            if self.bkfEl <= min(self.elevations):
                raise Exception('Bankfull elevation is at or below XS bottom.')
            if self.elevations[self.thwIndex] >= self.bkfEl:
                raise Exception('Thw index (' + str(self.thwIndex) + ') is at or above bankfull elevation.')
            
            broken = sm.break_at_bankfull(self.stations,self.elevations,self.bkfEl,self.thwIndex)
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
            
    def determine_bounding_truths(self):
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
        
        boundDict = {'bkfW':None,'floodproneWidth':None}
        if self.bkfEl:
            boundDict['bkfW'] = (self.bkfEl<=leftMax,self.bkfEl<=rightMax)
            boundDict['floodproneWidth'] = (self.floodproneEl<=leftMax,self.floodproneEl<=rightMax)
        self.boundTruths = boundDict
    
    def calculate_bankfull_statistics(self):
        """
        Recalculate all statistics. Note that if bkfEl is None, then the attributes set by these methods will be done.
        Also note that if bkfEl exceeds the maximum elevation of the surveyed channel then somme attributes
        may represent a lower bound rather than the actual value.
        """
        self.set_bankfull_stations_and_elevations()
        
        #note that the order you call these in DOES matter
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
        self.calculate_entrenchment_ratio()
        self.calculate_bank_height_ratio()
        self.calculate_velocity()
        self.calculate_flow()
        
        self.determine_bounding_truths()
    
    def calculate_area(self):
        """
        Calculates the area under a given elevation. Only calculates area in primary channel
        (as defined by min el) by default.
        """
        if self.bkfEl:
            area = sm.get_area(self.bStations,self.bElevations)
            self.bkfA = area
        else:
            self.bkfA = None
    
    def calculate_wetted_perimeter(self):
        """
        Calculates the wetted perimeter under a given elevation.
        """
        if self.bkfEl:
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
        if self.bkfEl:
            self.bkfHydR = self.bkfA / self.bkfWetP
        else:
            self.bkfHydR = None
    
    def calculate_shear_stress(self):
        """
        Calculates the shear stress at bkf. If metric, units are N/m^2. If imperial, units are lbs/ft^2
        """
        
        if self.waterSlope and self.bkfEl: # if we don't have a waterslope set, we can't calculate this.
            gammaWater = self.unitDict['gammaWater']
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
        Calculates the elevation of the floodprone area. This elevation is at twice the bkf max depth by default.
        """
        if self.bkfEl:
            minEl = min(self.bElevations)
            self.floodproneEl = minEl + 2*self.bkfMaxD
        else:
            self.floodproneEl = None
        
    def calculate_floodprone_width(self):
        """
        Calculates the width of the floodprone area.
        """
        if self.bkfEl:
            broken = sm.break_at_bankfull(self.stations,self.elevations,self.floodproneEl,self.thwIndex)
            floodSta = broken[0]
            self.floodproneWidth = sm.max_width(floodSta)
        else:
            self.floodproneWidth = None
        
    def calculate_entrenchment_ratio(self):
        """
        Calculates the entrenchment ratio - the flood prone width divided by the bankfull width
        """
        if self.bkfEl:
            self.entrenchmentRatio = self.floodproneWidth / self.bkfW
        else:
            self.entrenchmentRatio = None
    
    def calculate_bank_height_ratio(self):
        """
        The height of the top of bank above the channel thalweg divided by the height of bankfull above the thalweg.
        """
        if self.bkfEl and self.tobEl:
            minEl = min(self.bElevations)
            bkfHeight = self.bkfEl - minEl
            tobHeight = self.tobEl - minEl
            self.bankHeightRatio = tobHeight / bkfHeight
        else:
            self.bankHeightRatio = None
    
    def calculate_mean_depth(self):
        """
        Calculates the mean depth given a certain elevation.
        """
        if self.bkfEl:
            meanDepth = sm.get_mean_depth(self.bStations,self.bElevations,self.bkfEl,True)
            self.bkfMeanD = meanDepth
        else:
            self.bkfMeanD = None
        
    def calculate_max_depth(self):
        """
        Calculates the max depth given a certain elevation.
        """
        if self.bkfEl:
            maxDepth = sm.max_depth(self.bElevations,self.bkfEl)
            self.bkfMaxD = maxDepth
        else:
            self.bkfMaxD = None

    def calculate_width(self):
        """
        Calculates the bankfull width given a certain elevation.
        """
        if self.bkfEl:
            self.bkfW = sm.max_width(self.bStations)
        else:
            self.bkfW = None
        
    def calculate_velocity(self):
        """
        Calculates the bkf discharge velocity, given a bkf elevation, ws slope and manning's n.
        Units are ft/s or meters/s.
        """
        if self.waterSlope and self.manN and self.bkfEl: # need all of these to calculate this
            manNum = self.unitDict['manningsNumerator']
            vel = (manNum/self.manN)*self.bkfHydR**(2/3)*self.waterSlope**(1/2)
            self.bkfV = vel
        else:
            self.bkfV = None
    
    def calculate_flow(self):
        """
        Calculates the volumetric flow given a bkf elevation, ws slope and manning's n.
        Units are cubic ft/s or cubic meters/s.
        """
        if self.waterSlope and self.manN and self.bkfEl: # need all of these to calculate this
            flow = self.bkfA*self.bkfV
            self.bkfQ = flow
        else:
            self.bkfQ = None
    
    def attribute_list(self, attribute, deltaEl = 0.1):
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
        saveEl = self.bkfEl # will use this to revert state at end of algorithm
        
        elArray = []
        attrArray = []
        
        minEl = self.elevations[self.thwIndex]
        self.bkfEl = minEl
        
        while self.bkfEl <= max(self.elevations):
            self.bkfEl += deltaEl
            self.calculate_bankfull_statistics()

            elArray.append(self.bkfEl)
            attrArray.append(getattr(self, attribute))
                
        self.bkfEl = saveEl # this line and next line reverts to initial bkfEl state
        self.calculate_bankfull_statistics()
        
        return(elArray,attrArray)
    
    def find_floodplain_elevation(self, deltaEl = 0.1):
        """
        Finds the elevation where the second derivative of the channel area with respect to bkf elevation is maximized.
        
        Args:
            deltaEl: the granularity of the change in elevation by which the area will be calculated.

        Returns:
            The elevation where d(dA/dh) is maximized.
            
        Raises:
            None.
            
        Notes:
            The current algorithm just checks every elevation between the thalweg and highest surveyed point.
            A much faster algorithm is possible that just checks at elevations that exist in the survey.
        """
        arrays = self.attribute_list(attribute = 'bkfA', deltaEl = deltaEl)
        elevations = arrays[0]
        areas = arrays[1]
        
        dAreas = np.diff(areas)/deltaEl # units: length^2 / length
        ddAreas = np.diff(dAreas)/deltaEl # units: length^2 / length / length
        #ddAreas[i] corresponds with elevations[i+1]
        
        mindex = sm.find_max_index(ddAreas)
        floodEl = elevations[mindex+1]
        return(floodEl)
    
    def bkf_brute_search(self, attribute, target, delta = 0.1):
        """
        Finds the most ideal bkf elevation by performing a brute force search, looking for a target value of a specified attribute.
        The attribute need not increase monotonically with bkf elevation.
        After exiting the algorithm, bankfull statistics will be recalculated for whatever the bkfEl was when entering the method.
        The algorithm only checks betwee the thw elevation and the maximum surveyed elevation.
        
        Args:
            attribute: a string that references an attribute such as bkfW that is dependent on bkf el.
            target: the ideal value of attribute.
            delta: the elevation interval between statistics calculations
            epsilon: the desired maximum absolute deviation from the target attribute.
            terminateOnSufficient: a boolean indicating if the first result within the tolerance should be returned
        
        Returns:
            The ideal bkf elevation.
            
        Raises:
            None.
        """
        
        # first save the current bkfEl, if any
        saveEl = self.bkfEl

        bottom = min(self.elevations)
        top = max(self.elevations)
        
        if self.thwStation:
            thwEl = self.elevations[self.thwIndex]
            if thwEl > bottom:
                bottom = thwEl        
        """
        The above nested if is meant to handle when a secondary channel contains the thw.
        But if the thwInd indicates a point in the main channel that is NOT the true thw then
        this will cause the algorithm to start with an incorrectly high bottom.
        """
        self.bkfEl = bottom
        best = self.bkfEl
        bestDistance = float('inf')
        while self.bkfEl <= top:
            self.bkfEl += delta
            self.calculate_bankfull_statistics()
            calculatedValue = getattr(self, attribute)
            distance = np.abs(calculatedValue-target)
            if distance < bestDistance:
                bestDistance = distance
                best = self.bkfEl
        
        foundEl = best # save the best result we found       
        self.bkfEl = saveEl # this line and next line reverts to initial bkfEl state
        self.calculate_bankfull_statistics()
        
        return(foundEl)
    
    def bkf_binary_search(self, attribute, target, epsilon = None, returnFailed = False):
        """
        Finds the most ideal bkf elevation by performing a binary-esque search, looking for a target value of a specified attribute.
        After exiting the algorithm, bankfull statistics will be recalculated for whatever the bkfEl was when entering the method.
        This runs much quicker than bkf_brute_search() but is restricted to attributes that increase monotonically with bkfEl.
        
        Args:
            attribute: a string that references an attribute such as bkfW that is MONOTONICALLY dependent on bkf el.
                       Results are not guaranteed to be accurate if the function that relates the attribute to bkf elevation is not monotonic increasing.
            target: the ideal value of attribute.
            epsilon: the maximum acceptable absolute deviation from the target attribute.
                
        Returns:
            The ideal bkf elevation.
            
        Raises:
            None.
        """
        # first save the current bkfEl, if any
        saveEl = self.bkfEl
        
        if epsilon is None:
            epsilon = target/1000 # by default the tolerance is 0.1% of the target.
        
        bottom = min(self.elevations)
        top = max(self.elevations)
        
        if self.thwStation:
            thwEl = self.elevations[self.thwIndex]
            if thwEl > bottom:
                bottom = thwEl        
        """
        The above nested if is meant to handle when a secondary channel contains the thw.
        But if the thwInd indicates a point in the main channel that is NOT the true thw then
        this will cause the algorithm to start with an incorrectly high bottom.
        """
        
        found = False
        foundUpperBound = False
        n = 0
        
        while not found and n < 1000:
            n += 1
            self.bkfEl = (bottom + top)/2
            self.calculate_bankfull_statistics()
            calculatedValue = getattr(self, attribute)
            if np.isclose(calculatedValue,target,atol=epsilon):
                found = True
            else:
                if calculatedValue > target: # if we have overestimated the bkf el
                    top = self.bkfEl
                    foundUpperBound = True
                else: # if we have underestimated the bkf el
                    bottom = self.bkfEl
                    if not foundUpperBound:
                        top = top * 2 # in case the target cannot be found within the confinements of the surveyed channel
                        if top >= max(self.elevations)*10**2:
                            print('Target too great for channel ' + str(self) + '. Breaking.')
                            break
        
        foundEl = self.bkfEl # save the best result we found       
        self.bkfEl = saveEl # this line and next line reverts to initial bkfEl state
        self.calculate_bankfull_statistics()
        
        if found:
            print('Converged in ' + str(n) + ' iterations.')
            return(foundEl)
        else:
            print('Could not converge in ' + str(n) + ' iterations.')
            if returnFailed:
                return(foundEl)
            else:
                return(None)
