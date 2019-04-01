"""
A brute force segmentation algorithm.
"""

from itertools import combinations

def combos(L, n):
    """
    Returns all combos using n elements of L.
    """
    return list(combinations(L, n)) 

def segment(exes, whys, n):
    """
    Partitions a series into n segments with the minimum possible sum of squared error.
    Always fits the first and last point perfectly.
    
    Args:
        exes: a list of x coords
        whys: a list of y coords
        n: the number of segments to create. Should be less than len(exes)-1
    """
    indices = [i for i in range(len(exes))][1:-1]
    possibles = combos(indices, n)
    bestIndices, inverseError = None, 0
    for pos in possibles:
        inds = [0]
        inds.extend(pos)
        inds.extend([indices[-1]+1])
        print(inds)