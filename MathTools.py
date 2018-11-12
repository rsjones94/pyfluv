# simple functions for finding intersections and areas between lines
import matplotlib.pyplot as plt

a = (1,2)
b = (3,1)

line1 = (1,0)
#line1 = (0.5,-1)
#line1 = (float('inf'),2)
#line1 = (float('inf'),3)

lineX = [-1,2,2,3,4,6,7,8,9,10]
lineY = [-1,2.5,3,2,0,-2,6,1,1.0,1]

### CURRENT ISSUES - INTERSECTIONS BETWEEN LINES THAT ARE THE SAME LINE NOT WORKING RIGHT - NEED TO GET BEGINNING AND END OF THE LINE SEGMENT

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
        
def findIntersections(seriesX,seriesY,line):
    """returns a list of points where a line intersects a series of linear line segments.
    The line should be a tuple of form (slope, intercept) and the x and y values are lists of equal length
    Intersections are returned in order of occurrence from top to bottom of the two lists.
    """
    
    intersectsX = []
    intersectsY = []
    
    n = len(seriesX)
    
    for i in range(0,n-1):
        x1 = seriesX[i]
        x2 = seriesX[i+1]
        y1 = seriesY[i]
        y2 = seriesY[i+1]
        
        isVert = False
        interval = (x1,x2)
        if (x1==x2):
            isVert = True
            interval = (y1,y2)
        
        testLine = lineFromPoints((x1,y1),(x2,y2)) # create a line based on two adjacent points in the lists

        if intersectsOnInterval(interval,line,testLine, vertical = isVert): # we only care if the lines intersect where the line segment actually exists
            intersection = intersectionOfLines(line,testLine)
            intersectsX.append(intersection[0])
            intersectsY.append(intersection[1])
            
    return(intersectsX,intersectsY)
    
    
inters = findIntersections(lineX,lineY,line1)

myPlot = plt.plot(lineX,lineY)
plt.scatter(inters[0],inters[1])
    