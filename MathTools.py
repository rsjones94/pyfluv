# simple functions for finding intersections and areas between lines

a = (1,2)
b = (3,1)

def lineFromPoints(p1,p2):
    """Returns the slope and intercept of a line defined by two points
    Takes two tuples or lists of length two and returns a tuple (m,b) where m is the slope and b is the intercept
    of a line y = mx + b that passes through both points
    """
    rise = p2[1] - p1[1]
    run = p2[0] - p1[0]
    
    slope = rise / run # rise over run
    intercept = p1[1] - p1[0]*slope
    
    return((slope,intercept))

def yFromEquation(x,equation):
    """Returns the y value for a line given an x and a tuple containing (slope, intercept)
    """
    y = equation[0] * x + equation[1]
    
    return(y)
    
def intersectionOfLines(l1,l2):
    """Returns the x and y coordinates of the intersection of two lines
    Takes two tuples of lists of length two of form (slope, intercept) that define the two lines
    
    y = m*x + b
    
    m1*x + b1 = m2*x + b2
    
    m1*x - m2*x + b1 = b2
    x(m1-m2) + b1 = b2
    x(m1-m2) = b2 - b1
    x = (b2-b1)/(m1-m2)
    
    """
    m1 = l1[0]
    m2 = l2[0]
    b1 = l1[1]
    b2 = l2[1]
    
    try:
        x = (b2-b1)/(m1-m2)
    except ZeroDivisionError: # we might expect that the lines are parallel
        return(None)
        
    y = yFromEquation(x,l1)
    
    return(x,y)

def intersectsOnInterval(interval,l1,l2):
    """determines if two lines intersection on a given interval (inclusive)
    Takes tuples of length two - an interval, and two line-defining tuples of form (slope, intercept)
    """
    
    try:
        xint = intersectionOfLines(l1,l2)[0] # x value of the intersection
    except TypeError: # if the lines are parallel then trying the index in the line above will throw a TypeError (as the result of intersectionofLines() will be a Nonetype)
        return(False)
    
    if (xint >= min(interval) and (xint <= max(interval))):
        return(True)
    else:
        return(False)