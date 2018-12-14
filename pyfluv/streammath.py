"""
Simple functions for processing stream survey data
"""
import numpy as np

def line_from_points(p1,p2):
    """
    Creates a line (slope,intercept) based on two points that fall on the line.
    
    Args:
        p1: A list or tuple of form (x,y)
        p2: A list or tuple of form (x,y)
        
    Returns:
        A tuple (slope,intercept) based on p1 and p2. If the line is not vertical, then the intercept is the y -intercept.
        If the line is vertical, the slope is float('inf') and the intercept is the x-intercept.
        
    Raises:
        None.
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


def y_from_equation(x,equation):
    """
    Given an equation representing a line (slope,intercept) and an x-coordinate, returns the corresponding y-value.
    
    Args:
        x: The x-coordinate, an int or float
        equation: A tuple or list of form (slope,intercept). The intercept is assumed to be a y-intercept.
    
    Returns:
        An int or float representing the y-coordinate on the line at x. If the line is vertical, returns the string 'Undefined'.
    
    Raises:
        None.
    """
    if equation[0] != float('inf'):
        y = equation[0] * x + equation[1]
    elif equation[0] == float('inf'):
        y = 'Undefined'
    return(y)
    
def x_from_equation(y,equation):
    """
    Given an equation representing a line (slope,intercept) and a y-coordinate, returns the corresponding x-value.
    
    Args:
        y: The y-coordinate, an int or float
        equation: A tuple or list of form (slope,intercept). The intercept is assumed to be a y-intercept.
    
    Returns:
        An int or float representing the x-coordinate on the line at y. If the line is vertical, returns the string 'Undefined'.
    
    Raises:
        None.
    """
    if equation[0] != float('inf'):
        x = (y - equation[1]) / equation[0]
    elif equation[0] == float('inf'):
        x = 'Undefined'
    return(x)
    
    
def intersection_of_lines(l1,l2):
    """
    Returns the x and y coordinates of the intersection of two lines.
    
    Args:
        l1: The first line, a tuple or list of form (slope,intercept). If slope is float('inf'), the intercept is taken
            to be a x-intercept. Otherwise, the slope is the y-intercept.
        l2: The first line, a tuple or list of form (slope,intercept). If slope is float('inf'), the intercept is taken
            to be a x-intercept. Otherwise, the slope is the y-intercept.
    
    Returns:
        A tuple (x,y) of the intersection between the lines if they do intersect.
        If there is no intersection, None is returned.
    
    Raises:
        None.
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
        y = y_from_equation(x,l2) # if the first line is vertical, then the intercept given is the horizontal intercept and l2 will cross at this intercept
    elif isVert2:
        x = b2
        y = y_from_equation(x,l1)
    else: # if neither are vertical then they intersect, unless they're parallel
        try:
            x = (b2-b1)/(m1-m2)
            y = y_from_equation(x,l1)
        except ZeroDivisionError: # we might expect that the lines are parallel
            return(None)
    return(x,y)


def does_intersect(s1,s2):
    """
    Determines if two line segments intersect.
    
    Args:
        s1: A line segment in the form ((x1,y1),(x2,y2)). May be any combination of lists and tuples.
        s2: A line segment in the form ((x1,y1),(x2,y2)). May be any combination of lists and tuples.
    
    Returns:
        Returns True if the segments intersect, else returns False.
    
    Raises:
        None.
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
    
    l1 = line_from_points(s1[0],s1[1])
    l2 = line_from_points(s2[0],s2[1])
    
    if intersects_on_interval(yRange,l1,l2,vertical=True) and intersects_on_interval(xRange,l1,l2,vertical=False):
        return(True)
    else:
        return(False)
    

def intersects_on_interval(interval,l1,l2,vertical=False):
    """
    Determines if two lines intersect on a given interval (inclusive).
    
    Args:
        interval: A tuple or list (x1,x2) that representes the inclusive range that an intersection will be checked for. By default, this range is a range of x values.
        s2: A tuple or list in the form (slope,intercept). If the slope is not float('inf'), then the intercept is a y-intercept. Otherwise it is an x intercept.
        s2: A tuple or list in the form (slope,intercept). If the slope is not float('inf'), then the intercept is a y-intercept. Otherwise it is an x intercept.
        vertical: A boolean. If True, then interval represents a range of x values. If False, represent a range of y values.
    
    Returns:
        True if the segments intersect, else returns False.
    
    Raises:
        None.
    """
    try:
        inter = intersection_of_lines(l1,l2)
        xint = inter[0] # x value of the intersection
        yint = inter[1]
    except TypeError: # if the lines are parallel then trying the index in the line above will throw a TypeError (as the result of intersection_of_lines() will be a Nonetype)
        return(False)
    
    if not(vertical):
        if (xint >= min(interval) and (xint <= max(interval))):
            return(True)
    elif vertical:
        if (yint >= min(interval) and (yint <= max(interval))):
            return(True)
    return(False)
    
    
def is_float_in(checkTuple,tupleList):
    """
    Like the "is in" keyword combo, but corrects for floating point errors.
    
    Args:
        checkTuple: A tuple or list of values or a single numeric value that you want to check is in tupleList. 
        tupleList: A a list or tuple containing lists or tuples that have the same length as checkTuple or a list or tuple of numeric values.
        
    Returns:
        True if checkTuple is in tupleList. False otherwise.
    
    Raises:
        None.
    """
    n = len(tupleList)
    
    for i in range(0,n):
        if np.allclose(checkTuple,tupleList[i]):
            return(True) 
    return(False)
    
    
def indices_of_equivalents(checkTuple,tupleList):
    """
    Finds all indices in a list of tuples that are equal to a tuple.
    
    Args:
        checkTuple: A tuple or list of values that you want to check for equivalency in a list of tuples.
        tupleList: A a list or tuple containing lists or tuples that have the same length as checkTuple.
        
    Returns:
        A list of values that indicate the indices of tupleList that containing a tuple equivalent to checkTuple.
    
    Raises:
        None.
    """
    n = len(tupleList)
    eq = []
    
    for i in range(0,n):
        if np.allclose(checkTuple,tupleList[i]):
            eq.append(i)
    return(eq)


def segment_length(p1,p2):
    """
    Returns the length of a line segment defined by two points.
    
    Args:
        p1: A list or tuple of form (x,y)
        p2: A list or tuple of form (x,y)
        
    Returns:
        The length of the line segment as an int or float.
    
    Raises:
        None.
    """
    p1x = p1[0]
    p1y = p1[1]
    p2x = p2[0]
    p2y = p2[1]
    
    length = ((p1x - p2x)**2 + (p1y - p2y)**2)**0.5
    return(length)
    
def angle_by_points(p1,p2,p3):
    """
    Gets the angle defined by three points in radians, assuming p2 is the hinge/on the bisecting line
    
    Args:
        p1: A list or tuple of form (x,y)
        p2: A list or tuple of form (x,y)
        p3: A list or tuple of form (x,y)
        
    Returns:
        Returns the angle define by the three points in radians as an int or float.
    
    Raises:
        None.
    """
    P12 = segment_length(p1,p2)
    P13 = segment_length(p1,p3)
    P23 = segment_length(p2,p3)
    
    angle = np.arccos((P12**2 + P13**2 - P23**2) / (2 * P12 * P13))
    return(angle)
    
def on_line_together(index1,index2,seriesX,seriesY, tol = 10e-5):
    """
    Tests if two points in a series lay on an uninterupted line segment together.
    
    Args:
        index1: The index of the first point (seriesX[index1],seriesY[index1])
        index2: The index of the second point (seriesX[index2],seriesY[index2])
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        tol: The maximum angular deviation concurrent line segments can have and still be considered on the same uninterupted segment.
            By default this is 10e-5 rads.
        
    Returns:
        True if the two points are the same line segment. False otherwise.
    
    Raises:
        ValueError: If index2 is not greater than index1.
    """
    if (index2 - index1) == 1:
        return(True)
    elif not((index2 - index1) > 1):
        raise ValueError('Error: must specify two indices where index2 is > index1')

    for i in range(index1+1,index2): # skip the first  two indices specified, stop one short of the last
        p1 = [seriesX[index1],seriesY[index1]]
        p2 = [seriesX[i],seriesY[i]]
        p3 = [seriesX[index2],seriesY[index2]]
        angle = angle_by_points(p1,p2,p3)
        angleIsNought = np.isclose(0,angle,atol=tol)
        if not(angleIsNought):
            return(False)
    return(True)
    
    
def get_intersections(seriesX,seriesY,line):
    """
    Returns a list of points where a line intersects a series of linear line segments.
    
    Args:
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        line: a line represented as tuple or list of form (slope,intercept).
            If slope is float('inf'), then the intercept is assumed to be an x-intercept. Otherwise, it is a y-intercept.
        
    Returns:
        A tuple of lists representing the intersections with the form (x,y). These are guaranteed to be unique.
            The form is (x-coordinates,y-coordinates,intersectionIndex). intersectionIndex is a list whose entries represent
            what index an intersection comes after in seriesX and seriesY
        
    Raises:
        None.
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
        
        testLine = line_from_points((x1,y1),(x2,y2)) # create a line based on two adjacent points in the lists

        if intersects_on_interval(interval,line,testLine, vertical = isVert): # we only care if the lines intersect where the line segment actually exists
            intersection = intersection_of_lines(line,testLine)
            intersectsX.append(intersection[0])
            intersectsY.append(intersection[1])
            intersectionIndex.append(i)
       
    # remove any identical intersect points
    checkList = list(zip(intersectsX,intersectsY))
    newList = []
    newIntersectList = []
    for i in range(0,len(checkList)):
        if not is_float_in(checkList[i],newList):
            newList.append(checkList[i])
            newIntersectList.append(intersectionIndex[i])
    try:
        intersectsX, intersectsY = zip(*newList) # remake the intersectsX/intersectY list with the unique points
    except ValueError: # when there are no intersections a ValueError is thrown
        intersectsX, intersectsY = [], [] 
    return(intersectsX,intersectsY,newIntersectList)
    
    
def insert_points_in_series(origSeriesX,origSeriesY,insertions):
    """
    Adds points to a series of x-y values. The indices the points are to be added to are specified by the user.
    
    Args:
        origSeriesX: A list of x coordinates.
        origSeriesY: A list of corresponding y coordinates.
        insertions: A tuple of lists of form (x-coordinates,y-coordinates,insertionIndices). This is the form of the output of streammath.get_intersections().
            
    Returns:
        A list of three lists of form (newXVals,newYVals,flagger). Flagger is a list that flags whether or not a given point
            in newXVals / newYVals was inserted into a series of points (1 for True, 0 for False).
        
    Raises:
        None.
        
    Todo:
        Make it so an insertion point will be ignored iff its x-y coords are equal to the point it is to be inserted after
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
    
    
def above_below(point,line):
    """
    Calculates if a point is above, below or on a line. Does not work for vertical lines.
    
    Args:
        point: A tuple or list of for (x,y)
        line: A line of form (slope,intercept)
        
    Returns:
        1 if the point is above the line.
        0 if the point is on the line.
        -1 if the point is below the line.
        
    Raises:
        None.
    """
    y = y_from_equation(point[0],line)
    
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
      
    
def scalp_series(seriesX,seriesY,equation, above = True):
    """
    Returns a list where are all points above (default) or below a non-vertical line are removed. Never removes points on the line.
    
    Args:
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        equation: A list or tuple representing a line of form (slope,y-intercept)
        above: True (default) if you want to remove all points above the line. False if you want to remove all points below the line.
        
    Returns:
        A tuple of two lists representing the scalped series.
        
    Raises:
        None.
    """
    if above:
        matchVal = 1
    elif not(above):
        matchVal = -1
    
    newX, newY = [],[]
    
    for i in range(0,len(seriesX)):
        x = seriesX[i]
        y = seriesY[i]
        if not(above_below((x,y),equation) == matchVal):
            newX.append(x)
            newY.append(y)
    return(newX,newY)


def remove_side(seriesX,seriesY,xVal,leftRight):
    """
    Removes all points in a series of x and y vals that are either to the left or right of a vertical line.
    
    Args:
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        xVal: The x intercept of the vertical line.
        leftRight: A string specifying which side to remove points from. 'left' to remove points to the left of xVal, 'right' to remove points to the right.
        
    Returns:
        A tuple of two lists representing the modified series.
        
    Raises:
        Exception: if leftRight is neither 'left' nor 'right'
    """
    if leftRight is not 'left' and leftRight is not 'right':
        raise Exception('Invalid leftRight value. findType must be "left" or "right"')
        
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
    
    
def keep_range(seriesX,seriesY,xRange):
    """
    Removes all points from a series of x-y points that do not fall within a range of x values (inclusive)
    
    Args:
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        xRange: The range of x values to keep points in (inclusive) as a list or tuple of form (x1,x2).
        
    Returns:
        A tuple of two lists representing the modified series.
        
    Raises:
        None.
    """
    xCopy,yCopy = remove_side(seriesX,seriesY,xRange[0],leftRight='left')
    xCopy,yCopy = remove_side(xCopy,yCopy,xRange[1],leftRight='right')
    return(xCopy,yCopy)
    

def tri_area(a,b,c):
    """
    Gets the area of a triangle defined by three points.
    
    Args:
        a: A point represented as a tuple or list of form (x,y)
        b: A point represented as a tuple or list of form (x,y)
        c: A point represented as a tuple or list of form (x,y)
        
    Returns:
        An int or float of the area of the triangle.
        
    Raises:
        None.
    """
    Ax = a[0]
    Ay = a[1]
    Bx = b[0]
    By = b[1]
    Cx = c[0]
    Cy = c[1]
    
    area = abs((Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By)) / 2)
    return(area)
    
    
def get_nearest_intersect_bounds(seriesX,seriesY,flag,searchStart):
    """
    Finds the nearest intersection point (distance defined as array distance, not euclidean distance or delta-x or delta-y) to the left and right
        of a given index in a series of x-y points.
    
    Args:
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        flag: A list indicating what points in a series are intersection points. 1 for True, 0 for False.
        searchStart: the index in seriesX/seriesY to begin the search from.
        
    Returns:
        A tuple of three lists. The first list is the x coords of the nearest intersections.
            The second is the y coords of the nearest intersections.
            The third is the indices of the nearest intersections.
        
    Raises:
        None.
    """
    xBounds = []
    yBounds = []
    indices = []
    
    #xArray = np.asarray(seriesX)
    #idx = (np.abs(xArray - searchStart)).argmin() # index of the station that is closest to searchStart
    #these two lines above were used to start a search from a given station, but a result is that overhangs can sometimes be closer to the station specified than the thalweg
        # so if we want to use this functionality, we must remove overhangs first
    idx = searchStart
    
    i = idx
    while True: # search to the left first
        if flag[i] == 1 or i == 0:
            xBounds.append(seriesX[i])
            yBounds.append(seriesY[i])
            indices.append(i)
            break
        i = i - 1
        
    i = idx
    while True: # search to the right
        i = i + 1 # we add to the index first because if the starting point is an intersection the above loop will have already found it
        if flag[i] == 1 or i == (len(flag)-1):
            xBounds.append(seriesX[i])
            yBounds.append(seriesY[i])
            indices.append(i)
            break
    return (xBounds,yBounds,indices)


def prepare_cross_section(seriesX,seriesY,line, thw = None):
    """
    Takes a cross section and returns the shape under the XS and between the main channel (defined as having the lowest thalweg by default)
    
    Args:
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        line: A tuple of form (slope,intercept) that represents the line that will be used to crop the channel. Cannot be vertical.
        thw: the index of the thalweg. By default this is None, and the function will assume the index is where seriesY is lowest.
        
    Returns:
        A tuple of two lists. The first is the prepared x coordinates. The second is the prepared y coordinates.
        
    Raises:
        None.
    """
    
    intersects = get_intersections(seriesX,seriesY,line) # get where the line intersects
    withAddedIntersects = insert_points_in_series(seriesX,seriesY,intersects) # merge the intersecting points
    
    if thw == None:
        mainChannelIndex = np.asarray(withAddedIntersects[1]).argmin() # find the index of the deepest point in the XS. If there are multiple points of equal depth, the leftmost is selected
    else:
        mainChannelIndex = thw
    
    # find the left and right bounds of the channel (nearest [by index] intersecting points to the thw)
    channelBounds = get_nearest_intersect_bounds(withAddedIntersects[0],withAddedIntersects[1],withAddedIntersects[2],mainChannelIndex)
    # get the stations of points between the intersect bounds - we'll use it to set the keep_range limits (to preserve undercuts)
    betweenX = (withAddedIntersects[0][channelBounds[2][0]:channelBounds[2][1]+1])
    minX = min(betweenX)
    maxX = max(betweenX)
    
    #remove points outside of channel
    chopped = keep_range(withAddedIntersects[0],withAddedIntersects[1],(minX,maxX)) #if you use channelBounds[0] you will remove undercuts in some cases that should be preserved
    #remove points above channel
    scalped = scalp_series(chopped[0],chopped[1],line,above=True)
    # what's left is the fundamental shape of the channel
    return(scalped[0],scalped[1])
    
    
def shoelace_area(seriesX,seriesY):
    """
    An implementation of the showlace formula for finding the area of an irregular, simple polygon - modified from code by user Mahdi on Stack Exchange
    
    Args:
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
    Returns:
        The signed area enclosed by the polygon defined by seriesX and seriesY. If the polygon is not simple (not self-intersecting)
            then the area may not be correct.
        
    Raises:
        None.
    """
    x = seriesX
    y = seriesY
    
    area = 0.5*(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return(area)


def get_area(seriesX,seriesY):
    """
    A wrapper for streammath.shoelace_area(). Returns the absolute (not signed) area.
    
    Args:
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        
    Returns:
        The unsigned area enclosed by the polygon defined by seriesX and seriesY. If the polygon is not simple (not self-intersecting)
            then the area may not be correct.
        
    Raises:
        None.
        
    Todo: 
        Make sure that when this is used to calculate XS area under an elevation that the left and right points have elevations == to the elevation you are calculating area under
    """
    area = np.abs(shoelace_area(seriesX,seriesY))
    return(area)
    
    
def is_cut(index,seriesX,seriesY,findType):
    """
    Determines if point in a series is part of an overhang or an undercut.
    
    Args:
        index: index of the point in question.
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        findType: A string ("overhang" or "undercut") indicating if you want to identify if the point is an overhang or an undercut.
        
    Returns:
        True if the point is the findType specified, False otherwise.
        
    Raises:
        Exception: if findType is not "overhang" or "undercut".
    """
    #print('newcall: checking index ' + str(index))
    if findType is not 'overhang' and findType is not 'undercut':
        raise Exception('Invalid findType value. findType must be "overhang" or "undercut"')
    #print(index)
    pointX = seriesX[index]
    pointY = seriesY[index]
    intersections = get_intersections(seriesX,seriesY,(float('inf'),pointX)) # returns the x and y coordinates of any intersections of a vertical line with h-intercept of pointX and the cross section
    
    intersectionsX = intersections[0]
    intersectionsY = intersections[1]
    
    if len(intersectionsX) == 1: # if there is only one intersection, then we know the point can not be either an overhang or an undercut
        return(False)
    
    for i in range(0,len(intersectionsX)): # we need to check each intersection to see if it's really an overhang/undercut
        sectX = intersectionsX[i]
        sectY = intersectionsY[i]
        eqs = indices_of_equivalents(sectX,seriesX) # indices of all natural points in the series that have the same x value as the intersection point

        if findType == 'overhang':
                elCheck = sectY < pointY
        elif findType == 'undercut':
                elCheck = sectY > pointY
                
        if len(eqs) == 1: # if eqs is length 1 (meaning the only eq point is the actual point we're checking), we just have to check if the intersection is appropriately above or below
            if elCheck:
               return(True)
        else: # if eqs is not length 1, then we need to make sure that that at least one equivalent point is not on the same line as the point we're checking (i.e., they are not on the same vertical line segment).
              # we also should only check equivalent points where the elevation is above or below (depending on the findType) the check point
            for j in range(0,len(eqs)):
                #print('i is ' + str(i) + ' and j is ' + str(j))
                
                eqIndex = eqs[j]
                eqY = seriesY[eqIndex]
                
                eqYisBelow = eqY < pointY
                if eqYisBelow and findType == 'undercut': # if looking for undercuts, we only should check eqpoints that are above
                    next
                elif not(eqYisBelow) and findType == 'overhang': # if looking for overcuts, we only should check eqpoints that are below
                    next

                segmentRange = [index,eqIndex]
                segmentRange.sort()
                #print('Checking indices ' + str(segmentRange))
                if segmentRange[0] == segmentRange[1]:
                    pass
                else:
                    onSameLine = on_line_together(segmentRange[0],segmentRange[1],seriesX,seriesY)
                    if elCheck and not(onSameLine):
                        return(True)
                
    return(False)
    

def get_cuts(seriesX,seriesY,findType):
    """
    Determines if which points in a series are part of an overhang or an undercut.
    
    Args:
        seriesX: A list of x coordinates.
        seriesY: A list of corresponding y coordinates.
        findType: A string ("overhang" or "undercut") indicating if you want to identify is an overhangs or undercuts.
        
    Returns:
        A list of indices representing which points in the series are of the findType specified.
        
    Raises:
        Exception: If findType is not "overhang" or "undercut"
    """
    if findType is not 'overhang' and findType is not 'undercut':
        raise Exception('Invalid findType value. findType must be "overhang" or "undercut"')
    
    cuts = []
    
    for i in range(0,len(seriesX)):
        if is_cut(i,seriesX,seriesY,findType):
            cuts.append(i)
    return(cuts)
    
    
def find_contiguous_sequences(numbers):
    """
    Takes a list of numbers and returns a list of lists of numbers that are form contiguous always-increasing sequences in that list.
    
    Args:
        numbers: A list of numbers.
            
    Returns:
        A list of lists where each sublist is a sequence of numbers found in the paremeter numbers where
            each subsequent element is equal to the previous element + 1.
        
    Raises:
        None.
    """
    masterList = []
    tackList = [numbers[0]] # the first number will always be in a list
    
    for i in range(1,len(numbers)): # skip the first
        if numbers[i] == numbers[i-1] + 1:
            tackList.append(numbers[i])
        else:
            masterList.append(tackList)
            tackList = [numbers[i]]
        if i == len(numbers) - 1: # if we reach the end of the list, we need to append the tackList no matter what
            masterList.append(tackList)
            
    if len(numbers) == 1: # when there's only one number in numbers, we have to tack this on by hand as the loop is never entered
        masterList.append(tackList)
        
    return(masterList)
        
    
def pare_contiguous_sequences(sequences,seriesY,minOrMax=None):
    """
    Given a list of lists of contiguous sequences corresponding to XS shots and a list of elevations, returns only the indices with the max elevation in each sequence (peak of the overhang) or min (bottom of an undercut)
    
    Args:
        sequences: A list of lists that represent the indices of overhangs or undercuts.
        seriesY: A list of y-values that sequences maps onto.
        minOrMax: Whether to find the highest or lowest points in each sublist of sequences. Must be "min" or "max"
            
    Returns:
        A list of indices that represent the highest or lowest point in each sublist of sequences.
        
    Raises:
        Exception: If minOrMax is not "min" or "max".
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
            

def remove_overhangs(seriesX,seriesY,method,adjustY=True):
    """
    Returns new cross section x and y coordinates that have had overhangs removed either by cutting them off or filling under them
    
    Args:
        seriesX: A list of x-coordinates.
        seriesY: A list of corresponding y-coordinates.
        method: What method to use for removing overhangs. Must be "cut" or "fill".
        adjustY: A boolean that specifies if y-values at the edges of removed overhangs should be adjusted. Default is True.
            
    Returns:
        A tuple of lists representing the new x and y coordinates of the section.
        
    Raises:
        Exception: If method is not "cut" or "fill"
    """
    if method == 'cut':
        findType = 'overhang'
        pareType = 'max'
    elif method == 'fill':
        findType = 'undercut'
        pareType = 'min'
    else:
        raise Exception('Invalid method. Method must be "cut" or "fill"')
    
    overhangs = get_cuts(seriesX,seriesY,findType)
    contigOverhangs = find_contiguous_sequences(overhangs)
    pareOverhangs = pare_contiguous_sequences(contigOverhangs,seriesY,minOrMax = pareType)
    
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
                    theLine = line_from_points(p1,p2)
                    newY[peakIndex] = y_from_equation(newX[peakIndex],theLine)
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
                    theLine = line_from_points(p1,p2)
                    newY[peakIndex] = y_from_equation(newX[peakIndex],theLine)
        except IndexError: # will happen when the first point is a backhang
            pass
            
    newX = [newX[i] for i in pointsEssential]
    newY = [newY[i] for i in pointsEssential]
    return(newX,newY)
 

def get_mean_elevation(seriesX,seriesY,ignoreCeilings=True): # gives weird results if overhangs are present - should remove first to prevent couble counting surfaces (when there is more than one depth for a given x)
    """
    Takes a series of x and y points and returns the weighted mean elevation of the line segments. 
        By default, ignores ceilings (any line segment [x1,y1],[x2,y2] where x1>=x2).
    
    Args:
        seriesX: A list of x-coordinates.
        seriesY: A list of corresponding y-coordinates.
        ignoreCeilings: True by default. A boolean that specifies if segments that are vertical or ceilings should be ignored.
        
    Returns:
        The mean weighed mean elevation of the line segments.
        
    Raises:
        None.
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
    
    
def get_mean_depth(seriesX,seriesY,bkfEl,ignoreCeilings=True):
    """
    A wrapper for streammath.get_mean_elevation(), but will subtract a number (usually bankfull depth) for you.
    
    Args:
        seriesX: A list of x-coordinates.
        seriesY: A list of corresponding y-coordinates.
        bkfEl: the elevation of bankfull
        ignoreCeilings: True by default. A boolean that specifies if segments that are vertical or ceilings should be ignored.
        
    Returns:
        The mean weighed mean depth of the line segments.
        
    Raises:
        None.
    """
    
    meanDepth = bkfEl - get_mean_elevation(seriesX,seriesY,ignoreCeilings)
    return(meanDepth)
    

def get_centroid(seriesX,seriesY):
    """
    Calculates the centroid of a non-intersecting polygon. Modified from code by Robert Fontino at UCLA.
    
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
    
    
def max_depth(seriesY,bkfEl):
    """
    Returns the maximum depth of a cross section.
    
    Args:
        seriesY: a list of y-coordinates in the cross section.
        bkfEl: the elevation of bankfull.
        
    Returns:
        The max depth of the channel.
        
    Raises:
        None.
    """
    minEl = min(seriesY)
    depth = bkfEl - minEl
    return(depth)
    
    
def max_width(seriesX):
    """
    Returns the difference of the leftmost and rightmost points in a cross section. Note that this is just one definition of max width.
    
    Args:
        seriesX: a list of x-coordinates in the cross section.
        
    Returns:
        The max width of the channel.
        
    Raises:
        None.
    """
    width = max(seriesX) - min(seriesX)
    return(width)
    
    
def length_of_overlap_1d(s1,s2):
    """
    Calculates the length of overlap of two 1d line segments.
    
    Args:
        s1: a tuple or list representing a 1d line segment defined by two x-coordinates (x1,x2)
        s2: a tuple or list representing a 1d line segment defined by two x-coordinates (x1,x2)
        
    Returns:
        The unsigned length of the overlap between the line segments.
        
    Raises:
        None.
    """
    min1 = min(s1)
    min2 = min(s2)
    max1 = max(s1)
    max2 = max(s2)
    overlap =  max(0, min(max1, max2) - max(min1, min2))
    return overlap


def length_of_overlap_2d(s1,s2):
    """
    Calculates the length of overlap of two 2d line segments.
    
    Args:
        s1: a tuple or list of tuples or lists representing a line segment of form ((x1,y2),(x2,y2))
        s2: a tuple or list of tuples or lists representing a line segment of form ((x1,y2),(x2,y2))
        
    Returns:
        The unsigned length of the overlap between the line segments.
        
    Raises:
        None.
    """
    
    s1.sort()
    s2.sort()
        
    l1 = line_from_points(s1[0],s1[1])
    l2 = line_from_points(s2[0],s1[1])
    
    lineIsSame = np.allclose(l1,l2)
    
    if not(lineIsSame):
        return(0)
    else:
        s1proj = (s1[0][0],s1[1][0]) # range of s1, or the projection of s1 onto the x axis represented as a range of x
        s2proj = (s2[0][0],s2[1][0]) # range of s2, or the projection of s2 onto the y axis represented as a range of x
        # we need to get the range of the overlap of the x values
        xOverlap = length_of_overlap_1d(s1proj,s2proj)
        slope = l1[0]
        lengthOfOverlap = (xOverlap**2 + (xOverlap*slope)**2)**0.5 # Pythagorean theorem
        return(lengthOfOverlap)
       
        
def length_of_segment(segment):
    """
    Calculates the length of a line segment.
    
    Args:
        segment: a tuple or list of tuples or lists representing a line segment of form ((x1,y2),(x2,y2))
        
    Returns:
        The unsigned length of the overlap between the line segments.
        
    Raises:
        None.
    """
    deltaX = segment[1][0] - segment[0][0]
    deltaY = segment[1][1] - segment[0][1]
    
    length = (deltaX**2  + deltaY**2)**0.5
    return(length)
        
    
def wetted_perimeter(childX,childY,parentX,parentY):
    """
    Calculates the total wetted perimeter of a prepared cross section by comparing it to its parent XS.
        It can be assumed that all segments that are not horizontal are wetted, but a horizontal segment
        is wetted for only its length that overlaps with a segment in the parent XS.
    
    Args:
        childX: a list of prepared x coordinates.
        childY: a list of prepared y coordinates.
        parentX: a list of x coordinates that childX originated from.
        parentY: a list of y coordinates that childY originated from.
        
    Returns:
        The wetted perimeter of the cross section.
        
    Raises:
        None.
    """
    length = 0
    for i in range(1,len(childX)):
        segment = [[childX[i-1],childY[i-1]],[childX[i],childY[i]]]
        line = line_from_points(segment[0],segment[1])
        if not(np.isclose(0,line[0])): # if the line isn't flat, we know it's fully wetted
            length += length_of_segment(segment)
        else: # otherwise, we can only add the length of the segment that overlaps with a portion of the original XS; it could be a ceiling but it (more likely) represents a water surface
            for j in range(1,len(parentX)):
                checkSeg = [[parentX[i-1],parentY[i-1]],[parentX[i],parentY[i]]]
                lapLength = length_of_overlap_2d(segment,checkSeg)
                length += lapLength    
    return(length)
    
    
def is_simple(seriesX,seriesY):
    """
    Given two lists of x and y coords for a multipoint line, determines if the series of line segments is simple (non self-intersecting) or not.
    
    Args:
        seriesX: a list of x coordinates.
        seriesY: a list of y coordinates.
        
    Returns:
        (True,-1,-1) if the series is simple. If the series self-intersections, returns a tuple where the first value
            is False, and the second and third are the indices of the line segments that intersect eachother.
        
    Raises:
        None.
    """
    
    bad1 = -1
    bad2 = -1
    for i in range(1,len(seriesX)):
        segment = [[seriesX[i-1],seriesY[i-1]],[seriesX[i],seriesY[i]]]
        # check if the segment intersects with any segments in the series, EXCEPT itself or the segments immediately before/after (these will always intersect by their definition)
        for j in range(1,len(seriesX)):
            if j in range(i-1,i+2):
                next
            else:
                checkSeg = [[seriesX[j-1],seriesY[j-1]],[seriesX[j],seriesY[j]]]
                if does_intersect(segment,checkSeg):
                    bad1 = i
                    bad2 = j
                    return(False,bad1,bad2)
    return(True,bad1,bad2)
    
    
def project_point(a,b):
    """
    Project a vector onto another vector
    
    Args:
        a: a vector as a list or tuple of form (x,y) to be projected onto b
        a: a vector as a list or tuple of form (x,y) which a is to be projected on
        
    Returns:
        The projection of a onto b as a numpy array (x,y)
 
    Raises:
        None.
    """
    maga = (sum(np.multiply(a,a)))**0.5
    #print("maga = " + str(maga))
    magb = (sum(np.multiply(b,b)))**0.5
    #print("magb = " + str(magb))
    
    unitb = np.divide(b,magb)
    #print("unitb = " + str(unitb))
    costheta = np.matmul(a,b) / (maga*magb)
    #print("costheta = " + str(costheta))
    a1 = maga*costheta
    #print("a1 = " + str(a1))
  
    proj = np.multiply(unitb,a1)
    return(proj)
    
def centerline_series(seriesX,seriesY):
    """
    Project series of x and y coords onto the centerline defined by the first and last points in the series
    
    Args:
        seriesX: a list or tuple of x coordinates.
        seriesY: a list or tuple of y coordinates.
        
    Returns:
        A tuple of numpy arrays that represent the projection of (seriesX,seriesY) onto its centerline.
        
    Raises:
        None.
    """
    # first define the origin as the first (x,y) point
    origX = seriesX[0]
    origY = seriesY[0]
    
    # then subtract the origin from the series so everything starts at (0,0)
    rmX = np.subtract(seriesX,origX)
    rmY = np.subtract(seriesY,origY)
    
    # get the point that will be defining the centerline
    centerlineX = rmX[len(rmX)-1]
    centerlineY = rmY[len(rmY)-1]
    centerlinePoint = (centerlineX,centerlineY)
    
    projX = [0]
    projY = [0]
    
    for i in range(1,len(seriesX)): # running project_point() on the origin will give NaNs, so we prepopulated it
        originalPoint = (rmX[i],rmY[i])
        projected = project_point(originalPoint,centerlinePoint)
        projX.append(projected[0])
        projY.append(projected[1])
        
    # add the origin we subtracted out earlier back in
    projX = np.add(projX,origX)
    projY = np.add(projY,origY)
    
    return(projX,projY)
  
def get_stationing(seriesX,seriesY,project = False):
    """
    Get stationing given survey [x,y] (planform) data. Note that overhangs are impossible if project = False.
    
    Args:
        seriesX: a list of x (planform) coordinates.
        seriesY: a list of y (planform) coordinates.
        project: a boolean indicating if points should be projected onto the centerline
                             (defined by the first and last points) before stationing is calculated.
        
    Returns:
        A list containing the stationing corresponding to each (x,y) pair.
 
    Raises:
        None.
    """
    if project:
        projected = centerline_series(seriesX,seriesY)
        workingX = projected[0]
        workingY = projected[1]
        
        stationList = [0]
        for i in range(1,len(seriesX)):
            p1 = (workingX[0],workingY[0])
            p2 = (workingX[i],workingY[i])
            length = segment_length(p1,p2)
            station = length
            stationList.append(station)
    else:
        workingX = seriesX
        workingY = seriesY
    
        stationList = [0]
        for i in range(1,len(seriesX)):
            p1 = (workingX[i-1],workingY[i-1])
            p2 = (workingX[i],workingY[i])
            length = segment_length(p1,p2)
            station = stationList[i-1] + length
            stationList.append(station)
        
    return(stationList)
    
def monotonic_increasing(x):
    """
    Determines if an array is monotonically increasing. Modified from code by Autoplectic on SO
    
    Args:
        x: a list
        
    Returns:
        True if x is increasing monotinically, False otherwise
 
    Raises:
        None
    """
    dx = np.diff(x)
    return np.all(dx >= 0)

def crawl_to_elevation(seriesY,elevation,startInd):
    """
    Find the indices of the first points the the left and right of a starting point that exceeds an elevation
    
    Args:
        seriesY: a list of elevation points (implicitly ordered by station)
        elevation: the threshhold elevation
        startInd: the index to begin the search
        
    Returns:
        A tuple (leftIndex,rightIndex) that indicates the first points that go above the elevation threshhold
        If there is no index on a particular side meeting this, None will be returned in lieu of an index
    
    Raises:
        Exception: if the elevation of the initial index is at or above the threshhold elevation
    """
    initialEl = seriesY[startInd]
    if initialEl >= elevation:
        raise Exception('Search start elevation of ' + str(initialEl) + ' is at or above threshhold of ' + str(elevation))
    
    listLen = len(seriesY)
    indices = []
    
    # first search to the left
    for i in range(startInd,-1,-1):
        if seriesY[i] >= elevation:
            indices.append(i)
            break
        elif i == 0:
            indices.append(None)
        
    # then search to the right
    for i in range(startInd,listLen,1):
        if seriesY[i] >= elevation:
            indices.append(i)
            break
        elif i == listLen - 1:
            indices.append(None)
    return(indices)
    
def find_min_index(seriesY):
    """
    Finds the index of the minimum in an array
    """
    winIndex = 0
    winValue = seriesY[0]
    for i in range(1,len(seriesY)):
        if seriesY[i] < winValue:
            winValue = seriesY[i]
            winIndex = i
    
    return(winIndex)
    
def break_at_bankfull(seriesX,seriesY,bkfEl,startInd):
    """
    Take a cross section and cuts it at the bankfull elevation. XS should be free of overhangs.
        If the bkf elevation is unbounded on either side, adds a point at the bkf elevation.
        
    Args:
        seriesX: a list of stationing
        seriesY: a list of elevations
        startInd: index of the center of the channel. Does not need to be exactly the center.
        bkfEl: the bankfull elevation
        
    Returns:
        A new Xs as two list that defines the channel at bankfull flow.
        
    Raises:
        Exception: if your start index has an elevation above bankfull elevation.
    """
    
    cutIndices = crawl_to_elevation(seriesY,bkfEl,startInd)
    
    exesToInsert = []
    whysToInsert = []
    yInsert = bkfEl
    for i in [0,1]: # finding the stationing of the points we're inserting, first left then right
        try:
            if i == 0:
                p1 = (seriesX[cutIndices[i]],seriesY[cutIndices[i]])
                p2 = (seriesX[cutIndices[i]+1],seriesY[cutIndices[i]+1])
            elif i == 1:
                p1 = (seriesX[cutIndices[i]-1],seriesY[cutIndices[i]-1])
                p2 = (seriesX[cutIndices[i]],seriesY[cutIndices[i]])
            line = line_from_points(p1,p2)
            xInsert = x_from_equation(bkfEl,line)
        except TypeError: #if there is the value of cutIndices[i] is None
            if i == 0:
                xInsert = seriesX[0]
            elif i == 1:
                xInsert = seriesX[len(seriesX)-1]
        exesToInsert.append(xInsert)
        whysToInsert.append(yInsert)
        
    # next we'll trim down the original XS. If cutIndices are None, we need to replace with the limits
    if cutIndices[0] == None:
        cutIndices[0] = -1
    if cutIndices[1] == None:
        cutIndices[1] = len(seriesX)
    
    cutX = seriesX[cutIndices[0]+1:cutIndices[1]]
    cutY = seriesY[cutIndices[0]+1:cutIndices[1]]
    
    cutX.insert(0,exesToInsert[0])
    cutX.append(exesToInsert[1])
    
    cutY.insert(0,whysToInsert[0])
    cutY.append(whysToInsert[1])
    
    return(cutX,cutY)
    
def blend_polygons():
    """
    Takes two polygons (represented as an array of X-Y coordinates) and returns one polygon that represents a weighted average of the two shapes.
    
    Args:
        
    Returns:
 
    Raises:

    Todo:
        WHAT THE HELL DOES IT MEAN TO AVERAGE SHAPES. This will be used to transition between riffles, pools and reaches smoothly
    """
    
