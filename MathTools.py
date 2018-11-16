# simple functions for finding intersections and areas between lines
import matplotlib.pyplot as plt
import numpy as np

depth = 4

a = (1,2)
b = (3,1)

#line1 = (1,0)
line1 = (0,depth)
#line1 = (0,3)
#line1 = (float('inf'),9)
#line1 = (float('inf'),3)

lineX = [0,1,3,4,5,7,12,8,10,10,15,15,18]
lineY = [1,3,1,3,3,5,3,4,2,0,0,2,6]

def lineFromPoints(p1,p2):
    """Returns the slope and intercept of a line defined by two points
    Takes two tuples or lists of length two and returns a tuple (m,b) where m is the slope and b is the intercept
    of a line y = mx + b that passes through both points
    If a line is vertical, slope is 'inf' and the intercept given is the horizontal intercept
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

def intersectsOnInterval(interval,l1,l2,vertical=False):
    """determines if two lines intersection on a given interval (inclusive)
    Takes tuples of length two - an interval, and two line-defining tuples of form (slope, intercept)
    Assume neither line is vertical. If one is, set vertical to True and use a y interval
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
        
def getIntersections(seriesX,seriesY,line):
    """returns a list of points where a line intersects a series of linear line segments.
    The line should be a tuple of form (slope, intercept) and the x and y values are lists of equal length
    Intersections are returned in order of occurrence from top to bottom of the two lists along with a list that indicates the index that the corresponding intersection occurs after
    
    Will not return intersections for lines that are identical - this is technically undefined
    If addToSeries is True, the function will 
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
        if checkList[i] not in newList:
            newList.append(checkList[i])
            newIntersectList.append(intersectionIndex[i])
    try:
        intersectsX, intersectsY = zip(*newList) # remake the intersectsX/intersectY list with the unique points
    except ValueError: # when there are no intersections a ValueError is thrown
        intersectsX, intersectsY = [], []
        
    return(intersectsX,intersectsY,newIntersectList)
    
def addIntersectionsToSeries(origSeriesX,origSeriesY,intersections):
    """Merges the intersections from getIntersections() to the original series
    Takes two lists indicating x and y values and a tuple that has three lists - intersectionX, intersectionY and the indices that the intersections should be inserted after
    
    Returns three lists - the merged X values and merged Y values and a list that flags if a point is an intersection (0 for no, 1 for yes)
    """
    seriesX = origSeriesX.copy()
    seriesY = origSeriesY.copy()
    
    intersX = intersections[0]
    intersY = intersections[1]
    indices = intersections[2]
    
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
    Can specify index of the thalweg; otherwise the function assumes it's the deepest point (if multiple points of equal depth assist, the leftmost is used)
    """
    
    intersects = getIntersections(seriesX,seriesY,line) # get where the line intersects
    withAddedIntersects = addIntersectionsToSeries(seriesX,seriesY,intersects) # merge the intersecting points
    
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
    
def getShoelaceArea(seriesX,seriesY):
    """An implementation of the showlace formula for finding area of an irrgular, simple polygon
    Modified from code by user Mahdi on Stack Exchange
    Take two list of x and y coords and returns the area enclosed by the points (assuming the last point connects to the first)
    """
    x = seriesX
    y = seriesY
    
    area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return(area)
    
def getMeanElevation(seriesX,seriesY,ignoreCeilings=True): # gives weird results if overhangs are present - should remove first
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
        
    print(els)
    print(weights)
        
    normWeights = [x / sum(weights) for x in weights] #normalize the weights
    
    meanEl = np.dot(els,normWeights)
    return(meanEl)
    
def getMeanDepth(seriesX,seriesY,bkfDepth,ignoreCeilings=True):
    """A wrapper for getMeanElevation, but will subtract the bkf depth for you
    """
    meanDepth = bkfDepth - getMeanElevation(seriesX,seriesY,ignoreCeilings)
    return(meanDepth)
    
    
    
def removeOverhangs(seriesX,seriesY,shrinking = True): # NOT DONE
    """Remove all overhangs from a cross section.
    By default, the algorithm will result in a XS with smaller area than original (expanding)
    If shrinking is set to False, the new XS will have a greater area (expanding)
    """
    newX, newY = [seriesX],[seriesY]
    
    for i in range(0,len(seriesX)):
        pointX = newX[i]
        pointY = newY[i]
        
        # determine if it's forehang
            #handle
        #determine if it's a backhang
            #handle

    pass
        
    
    
# figuring out wetted perimeter will be a challenge
    
    
    
inters = getIntersections(lineX,lineY,line1)

myPlot = plt.plot(lineX,lineY)
plt.scatter(inters[0],inters[1])

merged = addIntersectionsToSeries(lineX,lineY,inters)
plt.plot(merged[0],merged[1])

prepared = prepareCrossSection(lineX,lineY,line1,thw=None) 
plt.plot(prepared[0],prepared[1], linewidth = 3)

print('Area = ' + str(round(getShoelaceArea(prepared[0],prepared[1]),2)))
print('Mean Depth = ', str(round(getMeanDepth(prepared[0],prepared[1],depth),2)))
