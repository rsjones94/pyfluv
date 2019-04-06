from . import streamexceptions
from . import streammath as sm

"""
An implementation of the famous Visvalingam's algorithm, which simplifies line
geometry.
"""

def eArea(n, exes, whys):
    """
    Effective area of a point on a line
    """
    pLeft = (exes[n-1], whys[n-1])
    p = (exes[n], whys[n])
    pRight = (exes[n+1], whys[n+1])

    return sm.tri_area(pLeft, p, pRight)

def eArea_alt(n, exes, whys, origArea):
    """
    An alternate definition of the effective area of the point on a line:
        the absolute value of the area of the shape without the point minus the
        original area
    """
    exes = exes.copy()
    whys = whys.copy()
    
    exes.pop(n)
    whys.pop(n)
    newArea = sm.get_area(exes, whys)
    diff = abs(origArea-newArea)

    return diff

def visvalingam(exes, whys, nKeep=None, nRemove=None):
    """
    Removes/keeps n points in a line. Never removes the first or last point.

    Args:
        exes: A list of x coords
        whys: A list of y coords
        nKeep: Number of points to keep. Must specify nKeep OR nRemove
        nRemove: Number of points to remove. Must specify nKeep OR nRemove

    Returns:
        A tuple of lists containing the coordinates of the simplified line.
    """
    exes = exes.copy()
    whys = whys.copy()

    nPoints = len(exes)
    if nKeep and nRemove:
        raise streamexceptions.InputError('nKeep and nRemove cannot both be specified.')
    elif nKeep:
        nToRemove = nPoints - nKeep
    elif nRemove:
        nToRemove = nRemove

    toRemove = []
    xCopy = exes.copy()
    yCopy = whys.copy()
    indices = [i for i in range(len(xCopy))][1:-1]

    while True:
        try:
            eAreas = [eArea(i, xCopy, yCopy) for i in range(len(xCopy))[1:-1]]
            mindex = sm.find_min_index(eAreas) # the min index in eAreas
            addex = indices[mindex] # the actual index that mindex corresponds to
            toRemove.append(addex)
            xCopy.pop(mindex)
            yCopy.pop(mindex)
            indices.pop(mindex)
        except IndexError:
            break

    for i in range(nToRemove):
        ind = toRemove[i]
        exes[ind] = None
        whys[ind] = None
        
    exes = [i for i in exes if i is not None]
    whys = [i for i in whys if i is not None]

    return exes, whys

def visvalingamish(exes, whys, nKeep=None, nRemove=None):
    """
    Removes/keeps n points in a line. Never removes the first or last point.
    Like visvalingam's algorithm, the effective area of a point is difference of
    the area of the shape excluding that point and the original shape

    Args:
        exes: A list of x coords
        whys: A list of y coords
        nKeep: Number of points to keep. Must specify nKeep OR nRemove
        nRemove: Number of points to remove. Must specify nKeep OR nRemove

    Returns:
        A tuple of lists containing the coordinates of the simplified line.
    """
    exes = exes.copy()
    whys = whys.copy()

    nPoints = len(exes)
    if nKeep and nRemove:
        raise streamexceptions.InputError('nKeep and nRemove cannot both be specified.')
    elif nKeep:
        nToRemove = nPoints - nKeep
    elif nRemove:
        nToRemove = nRemove

    toRemove = []
    xCopy = exes.copy()
    yCopy = whys.copy()
    indices = [i for i in range(len(xCopy))][1:-1]
    
    origArea = sm.get_area(xCopy,yCopy)
    while True:
        try:
            eAreas = [eArea_alt(i, xCopy, yCopy, origArea) for i in range(len(xCopy))[1:-1]]
            mindex = sm.find_min_index(eAreas) # the min index in eAreas
            addex = indices[mindex] # the actual index that mindex corresponds to
            toRemove.append(addex)
            xCopy.pop(mindex)
            yCopy.pop(mindex)
            indices.pop(mindex)
        except IndexError:
            break

    for i in range(nToRemove):
        ind = toRemove[i]
        exes[ind] = None
        whys[ind] = None
        
    exes = [i for i in exes if i is not None]
    whys = [i for i in whys if i is not None]

    return exes, whys





