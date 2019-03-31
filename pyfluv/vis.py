from . import streamexceptions

"""
An implementation of the famous Visvalingam's algorithm, which simplifies line
geometry.
"""

def visvalingam(exes, whys, nKeep=None, nRemove=None, keepLow=False):
    """
    Removes/keeps n points in a line
    
    Args:
        exes: A list of x coords
        whys: A list of y coords
        nKeep: Number of points to keep. Must specify nKeep OR nRemove
        nRemove: Number of points to remove. Must specify nKeep OR nRemove
        keepLow: If True the algorithm will not remove the point with this
                 lowest y coordinate
                 
    Returns:
        A tuple of lists containing the coordinates of the simplified line.
    """
    exes = exes.copy()
    whys = whys.copy()
    
    nPoints = len(exes)
    if nKeep and nRemove:
        raise streamexceptions.InputError('nKeep and nRemove cannot both be specified.')
    elif nKeep:
        nToKeep = nKeep
    elif nRemove:
        nToKeep = nPoints - nRemove
        
    pass