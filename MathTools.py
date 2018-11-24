# simple functions for finding intersections and areas between lines
import matplotlib.pyplot as plt
import numpy as np
#import shapely
# TODO: integrate shapely for some geometric operations

depth = 3

method = 'cut'
adjustY = True

if method == 'cut':
    findType = 'overhang'
    pareType = 'max'
elif method == 'fill':
    findType = 'undercut'
    pareType = 'min'
else:
    raise Exception('Invalid method. Method must be "cut" or "fill"')

line1 = (0,depth)

lineX = [0,1,3,2,5,7,9,7.5,10,10,12,15,14,16,14,13.5,12.5,17]
lineY = [1,3,1,4,3,5,5,4,2,-1,.5,0,2,4,3,4,4.5,5]

###

def lineFromPoints(p1,p2):
    """Returns the slope and intercept of a line defined by two points
    Takes two tuples or lists of length two and returns a tuple (m,b) where m is the slope and b is the intercept
    of a line y = mx + b that passes through both points
    If a line is vertical, slope is 'inf' and the intercept given is the horizontal intercept
    If the two points are the same, then the slope is 'inf' and the x intercept is the points' x value
    """
    rise = p2[1] - p1[1]
    run = p2[0] - p1[0]
    
    try:
        slope = rise / run # rise over run
        intercept = p1[1] - p1[0]*slope
    except ZeroDivisionError: # we might expect that a line is vertical
        slope = float('inf')
        intercept = p1[0] # in this case, this is actually the horizontal intercept
    
    return((slope,intercept))

def yFromEquation(x,equation):
    """Returns the y value for a line given an x and a tuple containing (slope, intercept)
    """
    if equation[0] != float('inf'):
        y = equation[0] * x + equation[1]
    elif equation[0] == float('inf'):
        y = 'Undefined'
    
    return(y)
    
def intersectionOfLines(l1,l2):
    """Returns the x and y coordinates of the intersection of two lines
    Takes two tuples of lists of length two of form (slope, intercept) that define the two lines
    """
    isVert1 = False # flags to note of lines are vertical, as such lines will need special code to check for intersections
    isVert2 = False
    
    m1 = l1[0]
    m2 = l2[0]
    b1 = l1[1]
    b2 = l2[1]
    
    if m1 == float('inf'):
        isVert1 = True
    if m2 == float('inf'):
        isVert2 = True
        
    # we might expect that one or both lines are vertical
    if isVert1 and isVert2:
        return(None) # note that if the two lines are the same line, then there are infinite intersections. This case is not handled
    elif isVert1:
        x = b1
        y = yFromEquation(x,l2) # if the first line is vertical, then the intercept given is the horizontal intercept and l2 will cross at this intercept
    elif isVert2:
        x = b2
        y = yFromEquation(x,l1)
    else: # if neither are vertical then they intersect, unless they're parallel
        try:
            x = (b2-b1)/(m1-m2)
            y = yFromEquation(x,l1)
        except ZeroDivisionError: # we might expect that the lines are parallel
            return(None)

    return(x,y)

def doesIntersect(s1,s2):
    """Returns whether two line segments [[x1,y1],[x2,y2]] intersect one another
    """
    # evaluate the y range
    y1a = s1[0][1]
    y1b = s1[1][1]
    y1Range = [y1a,y1b]
    y1Range.sort()
    
    y2a = s2[0][1]
    y2b = s2[1][1]
    y2Range = [y2a,y2b]
    y2Range.sort()
    
    if (y2Range[1] < y1Range[0] and y2Range[0] < y1Range[0]) or (y2Range[1] > y1Range[0] and y2Range[0] > y1Range[0]):
        return(False) # can't have any overlap if there is no overlap in the y ranges

    yRange = y1Range+y2Range
    yRange.sort()
    yRange = yRange[1:3] # we just want to evaluate on the overlapping parts of ranges
    
    
    #evaluate the x range
    x1a = s1[0][0]
    x1b = s1[1][0]
    x1Range = [x1a,x1b]
    x1Range.sort()
    
    x2a = s2[0][0]
    x2b = s2[1][0]
    x2Range = [x2a,x2b]
    x2Range.sort()
    
    if (x2Range[1] < x1Range[0] and x2Range[0] < x1Range[0]) or (x2Range[1] > x1Range[0] and x2Range[0] > x1Range[0]):
        return(False) # can't have any overlap if there is no overlap in the x ranges

    xRange = x1Range+x2Range
    xRange.sort()
    xRange = xRange[1:3]
    
    # we only want to check for intersections on the most restrictive combination of the two ranges
    
    l1 = lineFromPoints(s1[0],s1[1])
    l2 = lineFromPoints(s2[0],s2[1])
    
    if intersectsOnInterval(yRange,l1,l2,vertical=True) and intersectsOnInterval(xRange,l1,l2,vertical=False):
        return(True)
    else:
        return(False)
    

def intersectsOnInterval(interval,l1,l2,vertical=False):
    """determines if two lines intersection on a given interval (inclusive). By default this is a range of x values but setting vertical to True will use a range of y values instead
    Takes tuples of length two - an interval, and two line-defining tuples of form (slope, intercept)
    """
    
    try:
        inter = intersectionOfLines(l1,l2)
        xint = inter[0] # x value of the intersection
        yint = inter[1]
    except TypeError: # if the lines are parallel then trying the index in the line above will throw a TypeError (as the result of intersectionofLines() will be a Nonetype)
        return(False)
    
    if not(vertical):
        if (xint >= min(interval) and (xint <= max(interval))):
            return(True)
    elif vertical:
        if (yint >= min(interval) and (yint <= max(interval))):
            return(True)

    return(False)
    
def isFloatIn(checkTuple,tupleList):
    """Equivalent to "is in" keyword combo, but will correct for floating point errors
    """
    n = len(tupleList)
    
    for i in range(0,n):
        if np.allclose(checkTuple,tupleList[i]):
            return(True)
        
    return(False)
    
    
def getIntersections(seriesX,seriesY,line):
    """returns a list of points where a line intersects a series of linear line segments.
    The line should be a tuple of form (slope, intercept) and the x and y values are lists of equal length
    Intersections are returned in order of occurrence from top to bottom of the two lists along with a list that indicates the index that the corresponding intersection occurs after
    
    Will not return intersections for lines that are identical - this is technically undefined
    """
    
    intersectsX = []
    intersectsY = []
    
    n = len(seriesX)
    
    intersectionIndex = []
    for i in range(0,n-1):
        x1 = seriesX[i]
        x2 = seriesX[i+1]
        y1 = seriesY[i]
        y2 = seriesY[i+1]
        
        isVert = False
        interval = (x1,x2)
        if (x1==x2):
            isVert = True # we'll use the isVert flag to indicate if the intersection algorithm needs to use its vertical functionality
            interval = (y1,y2)
        
        testLine = lineFromPoints((x1,y1),(x2,y2)) # create a line based on two adjacent points in the lists

        if intersectsOnInterval(interval,line,testLine, vertical = isVert): # we only care if the lines intersect where the line segment actually exists
            intersection = intersectionOfLines(line,testLine)
            intersectsX.append(intersection[0])
            intersectsY.append(intersection[1])
            intersectionIndex.append(i)
       
    # remove any identical intersect points
    checkList = list(zip(intersectsX,intersectsY))
    newList = []
    newIntersectList = []
    for i in range(0,len(checkList)):
        if not isFloatIn(checkList[i],newList):
            newList.append(checkList[i])
            newIntersectList.append(intersectionIndex[i])
    try:
        intersectsX, intersectsY = zip(*newList) # remake the intersectsX/intersectY list with the unique points
    except ValueError: # when there are no intersections a ValueError is thrown
        intersectsX, intersectsY = [], []
        
    return(intersectsX,intersectsY,newIntersectList)
    
def insertPointsInSeries(origSeriesX,origSeriesY,insertions):
    """Merges the points in a list to a cross section
    Takes two lists indicating x and y values and a tuple that has three lists - insertX, insertY and the indices that the points should be inserted after
    
    Returns three lists - the merged X values and merged Y values and a list that flags if a point is an intersection (0 for no, 1 for yes)
    If an insertion point already exists in the series, it will still be merged
    TODO: Make it so an insertion point will be ignored iff its x-y coords are equal to the point it is to be inserted after
    """
    seriesX = origSeriesX.copy()
    seriesY = origSeriesY.copy()
    
    intersX = insertions[0]
    intersY = insertions[1]
    indices = insertions[2]
    
    flagger = []
    for i in range(len(seriesX)):
        flagger.append(0) # making a list of flags for if a point is an intersection - initially none are
    
    offset = 1
    for i in range(0,len(indices)): # inserting points
        seriesX.insert(indices[i]+offset,intersX[i])
        seriesY.insert(indices[i]+offset,intersY[i])
        flagger.insert(indices[i]+offset,1)
        offset += 1
        
    return(seriesX,seriesY,flagger)
    
def aboveBelow(point,line):
    """Calculates if a point is above, below or on a line
    takes a point (px,py) and a line (m,b)
    Returns 1 for above, -1 for below, 0 for on
    If the line is vertical (m = inf) no result will be given
    """
    y = yFromEquation(point[0],line)
    
    try:
        if point[1] == y:
            ans = 0
        elif point[1] > y:
            ans = 1
        elif point[1] < y:
            ans = -1
    except TypeError: # when the line is vertical, a TypeError is thrown
        ans = None
        
    return(ans)
        
def scalpSeries(seriesX,seriesY,equation, above = True):
    """Returns a list where are all points above (default) or below a non-vertical line are removed
    """
    
    if above:
        matchVal = 1
    elif not(above):
        matchVal = -1
    
    newX, newY = [],[]
    
    for i in range(0,len(seriesX)):
        x = seriesX[i]
        y = seriesY[i]
        if not(aboveBelow((x,y),equation) == matchVal):
            newX.append(x)
            newY.append(y)
    
    return(newX,newY)

def removeSide(seriesX,seriesY,xVal,leftRight):
    """Accepts a two lists (x and y coords) and remove all points that are either to the left or right of a given x
    leftRight = 'left' will remove all points to left
    leftRight = 'right' will remove all point to right
    """
    
    newX, newY = [],[]
    
    for i in range(0,len(seriesX)):
        x = seriesX[i]
        y = seriesY[i]
        if leftRight == 'left':
            if x >= xVal:
                newX.append(x)
                newY.append(y)
        elif leftRight == 'right':
            if x <= xVal:
                newX.append(x)
                newY.append(y)
    
    return(newX,newY)
    
def keepRange(seriesX,seriesY,xRange):
    """Keeps only the points that fall within a range (inclusive)
    """
    
    xCopy,yCopy = removeSide(seriesX,seriesY,xRange[0],leftRight='left')
    xCopy,yCopy = removeSide(xCopy,yCopy,xRange[1],leftRight='right')
    
    return(xCopy,yCopy)
    

def triArea(a,b,c):
    """Returns the area of a triangle defined by three points (px,py)
    """
    Ax = a[0]
    Ay = a[1]
    Bx = b[0]
    By = b[1]
    Cx = c[0]
    Cy = c[1]
    
    area = abs((Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By)) / 2)
    return(area)
    
def getNearestIntersectBounds(seriesX,seriesY,flag,searchStart):
    """Finds the nearest intersection point (distance defined as array distance, not euclidean distance or delta-x/y) to left and right of a given index
    Will be used to define bankfull limits and left/right chopping
    flag is an array that indicates if a corresponding (x,y) point is an intersection
    """
    xBounds = []
    yBounds = []
    indices = []
    
    #xArray = np.asarray(seriesX)
    #idx = (np.abs(xArray - searchStart)).argmin() # index of the station that is closest to searchStart
    #these two lines above were used to start a search from a given station, but a result is that overhangs can sometimes be closer to the station specified than the thalweg
    idx = searchStart
    
    i = idx
    while True: # search to the left first
        #print(i)
        if flag[i] == 1 or i == 0:
            xBounds.append(seriesX[i])
            yBounds.append(seriesY[i])
            indices.append(i)
            break
        i = i - 1
        
    i = idx
    while True: # search to the right
        #print(i)
        i = i + 1 # we add to the index first because if the starting point is an intersection the above loop will have already found it
        if flag[i] == 1 or i == (len(flag)-1):
            xBounds.append(seriesX[i])
            yBounds.append(seriesY[i])
            indices.append(i)
            break
        
    return (xBounds,yBounds,indices)

def prepareCrossSection(seriesX,seriesY,line, thw = None):
    """Takes a cross section and returns the shape under the XS and between the main channel (defined as having the lowest thalweg)
    Can specify index of the thalweg; otherwise the function assumes it's the deepest point (if multiple points of equal depth exist, the leftmost is used)
    """
    
    intersects = getIntersections(seriesX,seriesY,line) # get where the line intersects
    withAddedIntersects = insertPointsInSeries(seriesX,seriesY,intersects) # merge the intersecting points
    
    if thw == None:
        mainChannelIndex = np.asarray(withAddedIntersects[1]).argmin() # find the index of the deepest point in the XS. If there are multiple points of equal depth, the leftmost is selected
    else:
        mainChannelIndex = thw
    
    # find the left and right bounds of the channel (nearest [by index] intersecting points to the thw)
    channelBounds = getNearestIntersectBounds(withAddedIntersects[0],withAddedIntersects[1],withAddedIntersects[2],mainChannelIndex)
    # get the stations of points between the intersect bounds - we'll use it to set the keepRange limits (to preserve undercuts)
    betweenX = (withAddedIntersects[0][channelBounds[2][0]:channelBounds[2][1]+1])
    minX = min(betweenX)
    maxX = max(betweenX)
    
    #remove points outside of channel
    chopped = keepRange(withAddedIntersects[0],withAddedIntersects[1],(minX,maxX)) #if you use channelBounds[0] you will remove undercuts in some cases that should be preserved
    #remove points above channel
    scalped = scalpSeries(chopped[0],chopped[1],line,above=True)
    
    # what's left is the fundamental shape of the channel
    return(scalped[0],scalped[1])
    
def shoelaceArea(seriesX,seriesY):
    """An implementation of the showlace formula for finding area of an irregular, simple polygon
    Modified from code by user Mahdi on Stack Exchange
    Take two list of x and y coords and returns the area enclosed by the points (assuming the last point connects to the first)
    
    The area returned is signed
    """
    x = seriesX
    y = seriesY
    
    area = 0.5*(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return(area)

def getArea(seriesX,seriesY):
    """A wrapper for shoelaceArea(). Returns the absolute (not signed) area
    TODO: Make sure that when this is used to calculate XS area under an elevation that the left and right points have elevations == to the elevation you are calculating area under
    """
    
    area = np.abs(shoelaceArea(seriesX,seriesY))
    return(area)
    
    
def isCut(index,seriesX,seriesY,findType):
    """Determines if point in a series is part of an overhang or an undercut
    findType must be 'overhang' or 'undercut'
    """
    
    if findType is not 'overhang' and findType is not 'undercut':
        raise Exception('Invalid findType value. findType must be "overhang" or "undercut"')
    
    pointX = seriesX[index]
    pointY = seriesY[index]
    intersections = getIntersections(seriesX,seriesY,(float('inf'),pointX)) # returns the x and y coordinates of any intersections of a vertical line with h-intercept of pointX and the cross section
    
    intersectionsX = intersections[0]
    intersectionsY = intersections[1]
    
    xs = list(map(list, zip(seriesX, seriesY)))
    
    for i in range(0,len(intersectionsX)):
        testPoint = (intersectionsX[i],intersectionsY[i])
        if findType == 'overhang':
            if (intersectionsY[i] < pointY) and (not isFloatIn(testPoint,xs)): # if the intersection elevation is below the testing point and the intersection point is actually a  point in the cross section (which implies that the the test point and intersection are on a vertical line together and do not constitute an overhang)
                return(True)
        elif findType == 'undercut':
            if (intersectionsY[i] > pointY) and (not isFloatIn(testPoint,xs)): # if the intersection elevation is below the testing point and the intersection point is actually a  point in the cross section (which implies that the the test point and intersection are on a vertical line together and do not constitute an overhang)
                return(True)
    
    return(False)
    

def getCuts(seriesX,seriesY,findType):
    """Returns the a list of indices of overhangs or undercuts in a cross sections.
    findType must be 'overhang' or 'undercut'
    """
    
    if findType is not 'overhang' and findType is not 'undercut':
        raise Exception('Invalid findType value. findType must be "overhang" or "undercut"')
    
    cuts = []
    
    for i in range(0,len(seriesX)):
        if isCut(i,seriesX,seriesY,findType):
            cuts.append(i)
    
    return(cuts)
    
    
def findContiguousSequences(numbers):
    """Takes a list of numbers and returns a list of lists of numbers that are form contiguous sequences in that list
    """
    masterList = []
    tackList = [numbers[0]] # the first number will always be in a list
    
    for i in range(1,len(numbers)): # skip the first
        if numbers[i] == numbers[i-1] + 1:
            tackList.append(numbers[i])
            if i == len(numbers) - 1: # if we reach the end of the list, we need to append the tackList no matter what
                masterList.append(tackList)
        else:
            masterList.append(tackList)
            tackList = [numbers[i]]
            
    return(masterList)
        
    
def pareContiguousSequences(sequences,seriesY,minOrMax=None):
    """Given a list of lists of contiguous sequences corresponding to XS shots and a list of elevations, returns only the indices with the max elevation in each sequence (peak of the overhang) or min (bottom of an undercut)
    """
    if minOrMax is not 'min' and minOrMax is not 'max':
        raise Exception('Invalid minOrMax value. minOrMax must be "min" or "max"')
    
    keepList = []
    
    for seq in sequences:
        currentY = seriesY[seq[0]]
        currentWinner = seq[0]
        for i in range(0,len(seq)):
            if seriesY[seq[i]] > currentY and minOrMax == 'max':
                currentWinner = seq[i]
            if seriesY[seq[i]] < currentY and minOrMax == 'min':
                currentWinner = seq[i]
        keepList.append(currentWinner)
        
    return(keepList)
            

def removeOverhangs(seriesX,seriesY,method,adjustY=True):
    """Returns new XS x and y coordinates that have had overhangs removed either by cutting them off or filling under them
    Valid values for method are "cut" and "fill"
    
    KNOWN BUG: If the XS is such that all points that are substrate (points x[i] where x[i] > x[i-1])
    are covered by overhangs, and there are by fore- and backhangs, then the algorithm will fail
    """
    if method == 'cut':
        findType = 'overhang'
        pareType = 'max'
    elif method == 'fill':
        findType = 'undercut'
        pareType = 'min'
    else:
        raise Exception('Invalid method. Method must be "cut" or "fill"')
    
    overhangs = getCuts(seriesX,seriesY,findType)
    contigOverhangs = findContiguousSequences(overhangs)
    pareOverhangs = pareContiguousSequences(contigOverhangs,seriesY, minOrMax = pareType)
    
    pointsNotEssential = [] # points that are overhangs or undercuts but not the peak or base
    for element in overhangs:
        if element not in pareOverhangs:
            pointsNotEssential.append(element)
            
    pointsEssential = []
    for i in range(0,len(seriesX)):
        if i not in pointsNotEssential:
            pointsEssential.append(i)
    
    newX = seriesX[:]
    newY = seriesY[:]
    
    for i in range(0,len(pareOverhangs)):
        
        peakIndex = pareOverhangs[i]
        contigArray = contigOverhangs[i] # the continuous sequence that the peak belongs to
        nextInd = contigArray[len(contigArray)-1] + 1 # index of the point following the continuous sequence
        prevInd = contigArray[0] - 1 # index of the point preceding the continuous sequnece

        try:
            if newX[peakIndex] > newX[nextInd]: # if it's a backhang
                anchorX = newX[peakIndex] # save the x
                newX[peakIndex] = newX[nextInd]
                if adjustY:
                    # we need to calculate the elevation of the adjusted point - it will fall on line segment behind it
                    p1 = (newX[prevInd],newY[prevInd])
                    p2 = (anchorX,newY[prevInd+1])
                    theLine = lineFromPoints(p1,p2)
                    newY[peakIndex] = yFromEquation(newX[peakIndex],theLine)
        except IndexError: # will happen when the last point is a forehang
            pass
        
        try:
            if newX[peakIndex] < newX[prevInd]: # if it's a forehang
                anchorX = newX[peakIndex] # save the x
                newX[peakIndex] = newX[prevInd]
                if adjustY:
                    # we need to calculate the elevation of the adjusted point - it will fall on line segment ahead of it
                    p1 = (anchorX,newY[nextInd-1])
                    p2 = (newX[nextInd],newY[nextInd])
                    theLine = lineFromPoints(p1,p2)
                    newY[peakIndex] = yFromEquation(newX[peakIndex],theLine)
        except IndexError: # will happen when the first point is a backhang
            pass
            
    newX = [newX[i] for i in pointsEssential]
    newY = [newY[i] for i in pointsEssential]
           
                
    return(newX,newY)
 

    
def getMeanElevation(seriesX,seriesY,ignoreCeilings=True): # gives weird results if overhangs are present - should remove first to prevent couble counting surfaces (when there is more than one depth for a given x)
    """Takes a cross section and returns the mean elevation of the points.
    By default, ignores ceilings (any line segment [x1,y1],[x2,y2] where x1>=x2)
    """
    els = [] # array of mean elevation for each qualifying segment
    weights = [] # horizontal component of length of each qualifying segment
    for i in range(0,len(seriesX)-1):
        x1 = seriesX[i]
        x2 = seriesX[i+1]
        y1 = seriesY[i]
        y2 = seriesY[i+1]
        
        if ignoreCeilings:
            condition = x2>x1
        else: 
            condition = True
            
        if condition:
            segmentEl = np.mean([y1,y2])
            segmentLength = x2-x1
            #print('El = ' + str(segmentEl))
            #print('Length = ' + str(segmentLength))
            els.append(segmentEl)
            weights.append(segmentLength)
        
    normWeights = [x / sum(weights) for x in weights] #normalize the weights
    
    meanEl = np.dot(els,normWeights)
    return(meanEl)  
    
def getMeanDepth(seriesX,seriesY,bkfDepth,ignoreCeilings=True):
    """A wrapper for getMeanElevation, but will subtract the bkf depth for you
    """
    meanDepth = bkfDepth - getMeanElevation(seriesX,seriesY,ignoreCeilings)
    return(meanDepth)
    

def getCentroid(seriesX,seriesY):
    """Calculates the centroid of a non-intersecting polygon. Modified from code by Robert Fontino at UCLA.
    Args:
        poly: a list of points, each of which is a list of the form [x, y].
    Returns:
        the centroid of the polygon in the form [x, y].
    Raises:
        ValueError: if poly has less than 3 points or the points are not
                    formatted correctly.
    """
    
    poly = list(map(list, zip(seriesX, seriesY)))
    
    # Make sure poly is formatted correctly
    if len(poly) < 3:
        raise ValueError('polygon has less than 3 points')
    for point in poly:
        if type(point) is not list or 2 != len(point):
            raise ValueError('point is not a list of length 2')
    # Calculate the centroid from the weighted average of the polygon's
    # constituent triangles
    area_total = 0
    centroid_total = [float(poly[0][0]), float(poly[0][1])]
    for i in range(0, len(poly) - 2):
        # Get points for triangle ABC
        a, b, c = poly[0], poly[i+1], poly[i+2]
        # Calculate the signed area of triangle ABC
        area = ((a[0] * (b[1] - c[1])) +
                (b[0] * (c[1] - a[1])) +
                (c[0] * (a[1] - b[1]))) / 2.0;
        # If the area is zero, the triangle's line segments are
        # colinear so we should skip it
        if 0 == area:
            continue
        # The centroid of the triangle ABC is the average of its three
        # vertices
        centroid = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0]
        # Add triangle ABC's area and centroid to the weighted average
        centroid_total[0] = ((area_total * centroid_total[0]) +
                             (area * centroid[0])) / (area_total + area)
        centroid_total[1] = ((area_total * centroid_total[1]) +
                             (area * centroid[1])) / (area_total + area)
        area_total += area
    return(centroid_total)
    
    
def maxDepth(seriesY,bkfEl):
    """Returns the maximum depth of a cross section. Takes an array of elevations and a water/bkf depth
    """
    minEl = min(seriesY)
    depth = bkfEl - minEl
    
    return(depth)
    
def maxWidth(seriesX):
    """Returns the difference of the leftmost and rightmost points in a cross section. Takes an array of stations/x coords
    Note that this is just one definition of max width
    """
    width = max(seriesX) - min(seriesX)
    return(width)
    
def lengthOfOverlap1d(s1,s2):
    """Calculates the length of overlap of two 1d line line segments
    Takes s1 and s2 where each is a tuple (x1,x2) of coordinates
    Returns the unsigned length of overlap between them
    """
    min1 = min(s1)
    min2 = min(s2)
    max1 = max(s1)
    max2 = max(s2)
    overlap =  max(0, min(max1, max2) - max(min1, min2))
    return overlap

def lengthOfOverlap2d(s1,s2):
    """Calculates the length of the overlap of two line 2d segments iff their associated lines are identical
    Takes two line segments of form [[x1,y1],[x2,y2]] (MUST BE IN A LIST, NOT TUPLE) and returns an unsigned length
    """
    
    s1.sort()
    s2.sort()
        
    l1 = lineFromPoints(s1[0],s1[1])
    l2 = lineFromPoints(s2[0],s1[1])
    
    lineIsSame = np.allclose(l1,l2)
    
    if not(lineIsSame):
        return(0)
    else:
        s1proj = (s1[0][0],s1[1][0]) # range of s1, or the projection of s1 onto the x axis represented as a range of x
        s2proj = (s2[0][0],s2[1][0]) # range of s2, or the projection of s2 onto the y axis represented as a range of x
        # we need to get the range of the overlap of the x values
        xOverlap = lengthOfOverlap1d(s1proj,s2proj)
        slope = l1[0]
        lengthOfOverlap = (xOverlap**2 + (xOverlap*slope)**2)**0.5 # Pythagorean theorem
        return(lengthOfOverlap)
        
def lengthOfSegment(segment):
    """Returns the length of a line segment [[x1,y1],[x2,y2]]
    """
    deltaX = segment[1][0] - segment[0][0]
    deltaY = segment[1][1] - segment[0][1]
    
    length = (deltaX**2  + deltaY**2)**0.5
    return(length)
        
def wettedPerimeter(childX,childY,parentX,parentY):
    """Calculates the total wetted perimeter of a prepared cross section by comparing it its parent XS
    It can be assumed that all segments that are not horizontal are wetted, but a horizontal segment is wetted iff it overlaps with a segment in the parents XS
    Returns an unsigned length
    """
    length = 0
    for i in range(1,len(childX)):
        segment = [[childX[i-1],childY[i-1]],[childX[i],childY[i]]]
        line = lineFromPoints(segment[0],segment[1])
        if not(np.isclose(0,line[0])): # if the line isn't flat, we know it's fully wetted
            length += lengthOfSegment(segment)
        else: # otherwise, we can only add the length of the segment that overlaps with a portion of the original XS; it could be a ceiling but it (more likely) represents a water surface
            for j in range(1,len(parentX)):
                checkSeg = [[parentX[i-1],parentY[i-1]],[parentX[i],parentY[i]]]
                lapLength = lengthOfOverlap2d(segment,checkSeg)
                length += lapLength
                
    return(length)
    
def isSimple(seriesX,seriesY,verbose=False):
    """Given two lists of x and y coords for a polyon, determines if the shape is simple (non self-intersecting) or not
    Algorithms exist that solve this in O(nlogn) and even O(n), but for simplicity of implementation a straightforward algorithm that runs in O(n^2) is used
    The algorithm terminates upon finding the first self-intersection. If verbose is true, then the indices of the offending segments are printed.
    """
    for i in range(1,len(seriesX)):
        segment = [[seriesX[i-1],seriesY[i-1]],[seriesX[i],seriesY[i]]]
        # check if the segment intersects with any segments in the series, EXCEPT itself or the segments immediately before/after (these will always intersect by their definition)
        for j in range(1,len(seriesX)):
            if j in range(i-1,i+2):
                next
            else:
                checkSeg = [[seriesX[j-1],seriesY[j-1]],[seriesX[j],seriesY[j]]]
                if doesIntersect(segment,checkSeg):
                    if verbose:
                        print('Not simple on ' + str(i) + ' and ' + str(j))
                    return(False)
    return(True)
    
    
def blendPolygons():
    """
    Takes two polygons (represented as an array of X-Y coordinates) and returns one polygon that represents a weighted average of the two shapes
    
    TODO: WHAT THE HELL DOES IT MEAN TO AVERAGE SHAPES. This will be used to transition between riffles, pools and reaches smoothly
    """

    
    
inters = getIntersections(lineX,lineY,line1)

myPlot = plt.plot(lineX,lineY)
plt.scatter(inters[0],inters[1])

merged = insertPointsInSeries(lineX,lineY,inters)
plt.plot(merged[0],merged[1])

prepared = prepareCrossSection(lineX,lineY,line1,thw=None) 
plt.plot(prepared[0],prepared[1], linewidth = 3)

print('Area = ' + str(round(getArea(prepared[0],prepared[1]),2)))
print('Mean Depth = ', str(round(getMeanDepth(prepared[0],prepared[1],depth),2)))

ov = getCuts(lineX,lineY,findType)
ovHangsX = [lineX[i] for i in ov]
ovHangsY = [lineY[i] for i in ov]
plt.scatter(ovHangsX,ovHangsY,s=100)

overhangSeqs = findContiguousSequences(ov)
pareHangs = pareContiguousSequences(overhangSeqs,lineY,pareType)
topHangsX = [lineX[i] for i in pareHangs]
topHangsY = [lineY[i] for i in pareHangs]
plt.scatter(topHangsX,topHangsY,s=200)

cut = removeOverhangs(lineX,lineY,method,adjustY)
plt.plot(cut[0],cut[1], linewidth = 4)

cent = getCentroid(prepared[0],prepared[1])
plt.scatter(cent[0],cent[1],s=250)

wetLength = wettedPerimeter(prepared[0],prepared[1],lineX,lineY)
print(wetLength)


sampX = [1,3,2,4,5,5,6,6,5.5,5,2,1]
sampY = [1,4,5,5,2,3,3,2,2,1,1.5,3]
plt.figure()
plt.plot(sampX,sampY)
print('Simple: ' + str(isSimple(sampX,sampY)))