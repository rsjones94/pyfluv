"""
A brute force segmentation algorithm. Reduces a series of x-y coords to n linear segments.
"""

from itertools import combinations

import numpy as np

from . import streammath as sm

def combos(L, n):
    """
    Returns all combos using n elements of L.
    """
    return list(combinations(L, n))

def projected_error(a,l):
    """
    Find the distance between a point a and a line l where l is defined by
    two points.
    
    Args:
        a: a vector as a list or tuple of form (x,y) to be projected onto l
        l: a line defined as a tuple of two tuples

    Returns:
        The the distance of the projection of a onto l

    Raises:
        None.
    """
    
    s1 = np.array(l[0])
    s2 = np.array(l[1])
    a = np.array(a)
    
    zero = s1.copy() # use this to make the origin zero
    
    s1 -= zero
    s2 -= zero # this is now the vector we'll project a onto
    a -= zero
    return sm.projected_magnitude(a,s2)

def segment(exes, whys, n):
    """
    Partitions a series into n segments with the minimum possible sum of
    projected error. Always fits the first and last point perfectly.
    
    Args:
        exes: a list of x coords
        whys: a list of y coords
        n: the number of segments to create. Should be less than len(exes)-1
    """
    indices = [i for i in range(len(exes))][1:-1]
    possibles = combos(indices, n-1)
    
    bestIndices, bestError = None, np.inf
    for pos in possibles:
        sumError = 0
        
        inds = [0]
        inds.extend(pos)
        inds.extend([indices[-1]+1])
        
        indPoints = zip(inds[0:-1], inds[1:]) # indices of the endpoints of the segments
        for seg in indPoints:
            s1 = exes[seg[0]], whys[seg[0]]
            s2 = exes[seg[1]], whys[seg[1]]
            error = [projected_error((exes[ind],whys[ind]),(s1,s2)) for ind in range(seg[0],seg[1]+1)]
            sumError += np.sum(error)
            
        if sumError < bestError:
            bestError = sumError
            bestIndices = inds
            
    return bestIndices, bestError
        