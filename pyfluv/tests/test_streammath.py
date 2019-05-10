"""
Tests for streammath.py using pytest
"""
import numpy as np
import pandas as pd
import pytest

from ..streamexceptions import InputError
from .. import streammath as sm

def test_line_from_points():

    p1 = (0, 0)
    p2 = [1, 1]
    p3 = [2, 1]
    p4 = (0, -1)
    p5 = (-1, 3)

    assert sm.line_from_points(p1, p2) == (1, 0)
    assert sm.line_from_points(p2, p1) == (1, 0)
    assert sm.line_from_points(p1, p3) == (0.5, 0)
    assert sm.line_from_points(p2, p3) == (0, 1)
    assert sm.line_from_points(p1, p4) == (float('inf'), 0)
    assert sm.line_from_points(p5, p2) == (-1, 2)

def test_y_from_equation():

    eq1 = (2, 0)
    eq2 = (float('inf'), 1)

    p1 = 1
    p2 = 3

    assert sm.y_from_equation(p1, eq1) == 2
    assert sm.y_from_equation(p2, eq1) == 6
    assert sm.y_from_equation(p1, eq2) == 'Undefined'
    assert sm.y_from_equation(p2, eq2) == 'Undefined'

def test_x_from_equation():

    eq1 = (2, 0)
    eq2 = (float('inf'), 1)

    p1 = 2
    p2 = 4

    assert sm.x_from_equation(p1, eq1) == 1
    assert sm.x_from_equation(p2, eq1) == 2
    assert sm.x_from_equation(p1, eq2) == 1
    assert sm.x_from_equation(p2, eq2) == 1

def test_get_populated_indices():

    testList = [2,1,0,None,None,None,2,3,None,2,None,10,10,1,None,None]

    assert sm.get_populated_indices(testList, 2) == [2,2]
    assert sm.get_populated_indices(testList, 3) == [2,6]
    assert sm.get_populated_indices(testList, 4) == [2,6]
    assert sm.get_populated_indices(testList, 8) == [7,9]

def test_get_nearest_value():

    testList1 = [2,1,0,None,None,None,2,3,None,2,None,10,10,1,None,None]
    testList2 = [None,None,None]

    assert sm.get_nearest_value(testList1, 1) == 1
    assert sm.get_nearest_value(testList1, 3) == 0
    assert sm.get_nearest_value(testList1, 4) == 2
    with pytest.raises(InputError) as e_info:
        e = sm.get_nearest_value(testList2, 1)

def test_interpolate_value():

    exes = [-3,-1,0,2,4,5,7]
    whys = [None,None,1,None,5,6,None]

    assert sm.interpolate_value(exes, whys, 2) == 1
    assert sm.interpolate_value(exes, whys, 3) == 3
    assert sm.interpolate_value(exes, whys, 0) == 1
    assert sm.interpolate_value(exes, whys, 6) == 6

def test_interpolate_series():

    exes = [-3,-1,0,2,4,5,7]
    whys = [None,None,1,None,5,6,None]

    assert sm.interpolate_series(exes, whys) == [1,1,1,3,5,6,6]

def test_intersection_of_lines():

    l1 = (1,0)
    l2 = (-1,2)
    l3 = (float('inf'),2)
    l4 = (float('inf'),3)

    assert sm.intersection_of_lines(l1, l2) == (1,1)
    assert sm.intersection_of_lines(l1, l3) == (2,2)
    assert sm.intersection_of_lines(l3, l4) == None

def test_does_intersect():
    """
    Companion function sm.ccw() untested
    """

    seg1 = ((0,0),(1,1))
    # seg2 = ((0,2),(2,0)) # barely touching - technically intersects
    seg3 = ((0,0.5),(2,0.25)) # obviously intersects
    seg4 = ((-1,0),(0,1)) # parallel
    seg5 = ((-1,0.25),(0,0.25)) # would intersect if extended further

    # assert sm.does_intersect(seg1, seg2) == True # test fails - is this behavior desired?
    assert sm.does_intersect(seg1, seg3) == True
    assert sm.does_intersect(seg1, seg4) == False
    assert sm.does_intersect(seg1, seg5) == False

def test_is_float_in():

    n1 = 3.21
    m1 = 4.34
    l1 = [3.21000000001, 4.39, 5]

    n2 = (3.21, 4.34)
    m2 = (3.21, 4.39)
    l2 = ((4.18,5.2),(3.21000000001, 4.34),(2,2.9))

    assert sm.is_float_in(n1, l1) == True
    assert sm.is_float_in(m1, l1) == False

    assert sm.is_float_in(n2,l2) == True
    assert sm.is_float_in(m2,l2) == False

def test_indices_of_equivalents():

    t1 = (2,1)
    l1 = [(2,1),(2,1),(3,2),(2,2),(2.00000001,0.99999999),(3,1)]

    assert sm.indices_of_equivalents(t1, l1) == [0,1,4]

def test_segment_length():

    p1 = (0,0)
    p2 = (1,1)
    p3 = (2,2)
    p4 = (2,1)

    assert np.isclose(sm.segment_length(p1, p2), 2**(1/2))
    assert np.isclose(sm.segment_length(p1, p3), 2*2**(1/2))
    assert np.isclose(sm.segment_length(p2, p4), 1)

def test_angle_by_points():

    p1 = (0,1)
    p2 = (0,0)
    p3 = (1,0)
    p4 = (1,1)
    p5 = (2,2)
    p6 = (-1,0)
    p7 = (0,-1)

    assert np.isclose(sm.angle_by_points(p1, p2, p3), np.pi/2)
    assert np.isclose(sm.angle_by_points(p4, p2, p3), np.pi/4)
    assert np.isclose(sm.angle_by_points(p5, p2, p3), np.pi/4)
    assert np.isclose(sm.angle_by_points(p6, p2, p3), np.pi)
    assert np.isclose(sm.angle_by_points(p7, p2, p3), np.pi/2)

def test_on_line_together():

    exes = [0,1,2,3,4,5,6,7]
    whys = [0,1,2,3,3,4,6,6]

    assert sm.on_line_together(0, 1, exes, whys) == True
    with pytest.raises(ValueError) as e_info:
        e = sm.on_line_together(1, 0, exes, whys)
    assert sm.on_line_together(0, 3, exes, whys) == True
    assert sm.on_line_together(2, 4, exes, whys) == False
    assert sm.on_line_together(4, 6, exes, whys) == False
    assert sm.on_line_together(4, 7, exes, whys) == False

def test_get_intersections():

    exes = [0,1,2,3,4,5,6,7,8,9]
    whys = [5,4,2,1,0,2,3,6,3,1]

    l1 = (0,4)
    l2 = (-1,4)
    l3 = (float('inf'),4)

    expectedA = ((1.0, 6.333333333333333, 7.666666666666667), (4.0, 4.0, 4.0), [0, 6, 7])
    expectedB = ((2.0, 4.0), (2.0, 0.0), [1, 4])
    expectedC = ((4,), (0.0,), [3])

    assert np.allclose(sm.get_intersections(exes, whys, l1), expectedA)
    assert np.allclose(sm.get_intersections(exes, whys, l2), expectedB)
    assert np.allclose(sm.get_intersections(exes, whys, l3), expectedC)

def test_insert_points_in_series():

    exes = [0,1,2,3,4,5,6,7,8,9]
    whys = [5,4,2,1,0,2,3,6,3,1]

    toInsert = ((1.0, 6.333333333333333, 7.666666666666667), (4.0, 4.0, 4.0), [0, 6, 7])

    expectedX = [0,1.0,1,2,3,4,5,6,6.33333333333333333,7,7.666666666666667,8,9]
    expectedY = [5,4.0,4,2,1,0,2,3,4.0,6,4.0,3,1]
    expectedFlag = [0,1,0,0,0,0,0,0,1,0,1,0,0]

    assert np.allclose(sm.insert_points_in_series(exes, whys, toInsert),
                       [expectedX, expectedY, expectedFlag])

def test_above_below():

    l1 = (0,1)
    l2 = (-1,3)

    p1 = (1,1)
    p2 = (2,2)
    p3 = (-2,0.5)

    assert sm.above_below(p1, l1) == 0
    assert sm.above_below(p2, l1) == 1
    assert sm.above_below(p3, l1) == -1
    assert sm.above_below(p1, l2) == -1
    assert sm.above_below(p2, l1) == 1
    assert sm.above_below(p3, l1) == -1

def test_scalp_series():

    exes = [0,1,2,3,4,5]
    whys = [5,4,2,5,6,-3]

    l1 = (-1,4)

    assert sm.scalp_series(exes, whys, l1, above=True) == ([2,5],[2,-3])
    assert sm.scalp_series(exes, whys, l1, above=False) == ([0,1,2,3,4],[5,4,2,5,6])

def test_remove_side():

    exes = [0,1,2,3,4,5]
    whys = [5,4,2,5,6,-3]

    x1 = 3

    assert sm.remove_side(exes, whys, x1, 'right') == ([0,1,2,3],[5,4,2,5])
    assert sm.remove_side(exes, whys, x1, 'left') == ([3,4,5],[5,6,-3])

def test_keep_range():

    exes = [0,1,2,3,4,5]
    whys = [5,4,2,5,6,-3]

    r1 = (1.5,4)

    assert sm.keep_range(exes, whys, r1) == ([2,3,4],[2,5,6])

def test_tri_area():

    p1 = (0,0)
    p2 = (1,1)
    p3 = (2,0)

    assert sm.tri_area(p1, p2, p3) == sm.tri_area(p2, p3, p1) == sm.tri_area(p3, p1, p2) == 1

def test_get_nearest_intersect_bounds():

    exes = [0,1,1,2,3,4,5,6,6.33333333333333333,7,7.666666666666667,8,9]
    whys = [5,4,4,2,1,0,2,3,4,6,4,3,1]
    flag = [0,1,0,0,0,0,0,0,1,0,1,0,0]

    assert np.allclose(sm.get_nearest_intersect_bounds(exes, whys, flag, 3),
                       ([1,6.333333333],[4,4],[1,8]))

def test_prepare_cross_section():

    exes = [0,1,2,3,4,5,6,7,8,9,10]
    whys = [5,4,2,1,0,2,3,6,3,1,6]

    l1 = (0,4)
    altThw = 9

    assert np.allclose(sm.prepare_cross_section(exes, whys, l1, thw=None),
                       ([1.0, 1, 2, 3, 4, 5, 6, 6.333333333333333], [4.0, 4, 2, 1, 0, 2, 3, 4.0]))
    assert np.allclose(sm.prepare_cross_section(exes, whys, l1, thw=altThw),
                       ([7.666666666666667, 8, 9, 9.6], [4.0, 3, 1, 4.0]))

def test_shoelace_area():

    x1 = [0,1,2]
    y1 = [1,0,1]

    x2 = [0,0,1,1]
    y2 = [1,0,0,1]

    x3 = [0,2,4,6,6,0]
    y3 = [2,0,0,2,3,3]

    assert np.isclose(sm.shoelace_area(x1, y1), -1)
    assert np.isclose(sm.shoelace_area(x2, y2), -1)
    assert np.isclose(sm.shoelace_area(x3, y3), -14)

def test_get_area():

    x1 = [0,1,2]
    y1 = [1,0,1]

    x2 = [0,0,1,1]
    y2 = [1,0,0,1]

    x3 = [0,2,4,6,6,0]
    y3 = [2,0,0,2,3,3]

    assert np.isclose(sm.get_area(x1, y1), 1)
    assert np.isclose(sm.get_area(x2, y2), 1)
    assert np.isclose(sm.get_area(x3, y3), 14)

def test_is_cut():

    exes = [0,1,2,3,3,4,5,4.5,6,7,8,8,8,9,7.5,10]
    whys = [10,9,8,5,2,2,3,2,4,5,8,8.5,9,10,1,3]

    assert sm.is_cut(0, exes, whys, 'overhang') == False
    assert sm.is_cut(0, exes, whys, 'undercut') == False
    assert sm.is_cut(3, exes, whys, 'undercut') == False
    assert sm.is_cut(3, exes, whys, 'overhang') == False
    assert sm.is_cut(4, exes, whys, 'undercut') == False
    assert sm.is_cut(4, exes, whys, 'overhang') == False
    assert sm.is_cut(6, exes, whys, 'undercut') == False
    assert sm.is_cut(6, exes, whys, 'overhang') == True
    assert sm.is_cut(7, exes, whys, 'undercut') == True
    assert sm.is_cut(7, exes, whys, 'overhang') == False
    assert sm.is_cut(10, exes, whys, 'undercut') == False
    assert sm.is_cut(10, exes, whys, 'overhang') == False
    assert sm.is_cut(11, exes, whys, 'undercut') == False
    assert sm.is_cut(11, exes, whys, 'overhang') == False

def test_get_cuts():

    exes = [0,1,2,3,3,4,5,4.5,6,7,8,9,6.5,10]
    whys = [10,9,8,5,2,2,3,2,4,5,9,10,1,3]

    assert sm.get_cuts(exes, whys, 'overhang') == [6,9,10,11]
    assert sm.get_cuts(exes, whys, 'undercut') == [7,12]

    """
    exes = [0,1,2,3,3,4,5,4.5,6,7,8,8,8,9,6.5,10]
    whys = [10,9,8,5,2,2,3,2,4,5,8,8.5,9,10,1,3]
    
    # fails - bug where vertical walls erroneously inhibit identification as overhang/undercut
    assert sm.get_cuts(exes, whys, 'overhang') == [6,9,10,11,12,13] assert sm.get_cuts(exes, whys, 'undercut') == [7,12]
    """
    
def test_find_contiguous_sequences():

    nums = [1,2,3,5,4,5,4,7,8,10,11,12,1,2]
    assert sm.find_contiguous_sequences(nums) == [[1,2,3],[5],[4,5],[4],[7,8],[10,11,12],[1,2]]

def test_pare_contiguous_sequences():

    whys = [10,9,8,5,2,2,3,2,4,5,9,10,1,3]
    seq = [[4,5],[7],[8,9,10]]
    
    assert sm.pare_contiguous_sequences(seq, whys, minOrMax='max') == [4,7,10]
    assert sm.pare_contiguous_sequences(seq, whys, minOrMax='min') == [4,7,8]
    
def test_remove_overhangs():
    
    x1 = [0,1,2,3,4,5]
    y1 = [5,4,3,3,4,5]
    with pytest.raises(IndexError) as e_info:
        e = sm.remove_overhangs(x1, y1, method = 'cut', adjustY = True)
    
    x2 = [0,1,2,2,3,4,5]
    y2 = [5,4,3,2,2,3,5]
    with pytest.raises(IndexError) as e_info:
        e = sm.remove_overhangs(x2, y2, method = 'cut', adjustY = True)
        
    x3 = [0,1,2,1.5,3,4,3.5,5]
    y3 = [5,4,3,2,2,3,3,5]
    assert np.allclose(sm.remove_overhangs(x3, y3, method = 'cut', adjustY = True),
                       ([0, 1, 1.5, 1.5, 3, 4, 4, 5], [5, 4, 3.5, 2, 2, 3, 3.666666666666667, 5]))
    assert np.allclose(sm.remove_overhangs(x3, y3, method = 'cut', adjustY = False),
                       ([0, 1, 1.5, 1.5, 3, 4, 4, 5], [5, 4, 3, 2, 2, 3, 3, 5]))
    assert np.allclose(sm.remove_overhangs(x3, y3, method = 'fill', adjustY = True),
                       ([0, 1, 2, 2, 3, 3.5, 3.5, 5], [5, 4, 3, 2.0, 2, 2.5, 3, 5]))
    assert np.allclose(sm.remove_overhangs(x3, y3, method = 'fill', adjustY = False),
                       ([0, 1, 2, 2, 3, 3.5, 3.5, 5], [5, 4, 3, 2, 2, 3, 3, 5]))
    
    """
    This fails; overhangs that contain a vertical wall within it are not totally removed
    x4 = [0,3,2,2,1,4,5,6]
    y4 = [5,4,3,2,1,1,5,6]
    assert np.allclose(sm.remove_overhangs(x4, y4, method = 'cut', adjustY = False),
                       ([0, 1, 1, 1, 1, 4, 5, 6], [5, 4, 3, 2, 1, 1, 5, 6]))
    """
    
def test_get_mean_elevation():
    
    exes = [0,2,4,5]
    whys = [2,0,0,1]
    expectedResult = 1*(2/5) + 0*(2/5) + 0.5*(1/5)
    
    assert np.isclose(sm.get_mean_elevation(exes, whys, ignoreCeilings=True), expectedResult)
    
def test_get_mean_depth():
    
    exes = [0,2,4,5]
    whys = [2,0,0,1]
    expectedResult = 1*(2/5) + 0*(2/5) + 0.5*(1/5)
    
    assert np.isclose(sm.get_mean_depth(exes, whys, bkfEl=1, ignoreCeilings=True), 1-expectedResult)
    
def test_get_centroid():
    
    x1 = [0,0,1,1]
    y1 = [1,0,0,1]
    
    x2 = [0,2,4,6]
    y2 = [2,0,0,2]
    
    x3 = [0,0,1,4,5,4.5,6]
    y3 = [4,2,1,5,6,3.5,8]
    
    assert np.allclose(sm.get_centroid(x1,y1), [0.5, 0.5])
    assert np.allclose(sm.get_centroid(x2,y2), [3.0, 1.1666666666666667])
    assert np.allclose(sm.get_centroid(x3,y3), [2.4878787878787882, 4.3545454545454545])

def test_max_depth():
    
    y1 = [1,0,0,1]
    y2 = [3,1,1,2]
    
    assert sm.max_depth(y1,1) == 1
    assert sm.max_depth(y2,3) == 2
    assert sm.max_depth(y2,4) == 3
    
def test_max_width():
    
    x1 = [0,0,1,1]
    x2 = [0,2,4,6]
    x3 = [2,4,6,8]
    
    assert sm.max_width(x1) == 1
    assert sm.max_width(x2) == 6
    assert sm.max_width(x3) == 6
    
def test_length_of_overlap_1d():
    
    x1 = (0,1)
    x2 = (1,2)
    x3 = (0.5,1.5)
    x4 = (1.5,0.5)
    x5 = (0.25,0.65)
    x6 = (2,3)
    
    assert np.isclose(sm.length_of_overlap_1d(x1,x2), 0)
    assert np.isclose(sm.length_of_overlap_1d(x1,x3), 0.5)
    assert np.isclose(sm.length_of_overlap_1d(x1,x4), 0.5)
    assert np.isclose(sm.length_of_overlap_1d(x1,x5), 0.4)
    assert np.isclose(sm.length_of_overlap_1d(x1,x6), 0)
    
def test_length_of_overlap_2d():
    
    x1 = [[0,0],[2,1]]
    x2 = [[1,0.5],[3,1.5]]
    x3 = [[1.5,1],[3.5,2]]
    
    assert np.isclose(sm.length_of_overlap_2d(x1,x2), (1/2)*5**(1/2))
    assert np.isclose(sm.length_of_overlap_2d(x1,x3), 0)
    
def test_length_of_segment():
    
    p1 = (0,0)
    p2 = (1,0)
    p3 = (0,1)
    p4 = (1,1)
    p5 = (2,1)
    
    assert np.isclose(sm.length_of_segment((p1,p2)), 1)
    assert np.isclose(sm.length_of_segment((p1,p3)), 1)
    assert np.isclose(sm.length_of_segment((p1,p4)), 2**(1/2))
    assert np.isclose(sm.length_of_segment((p2,p5)), 2**(1/2))
    
def test_is_simple():
    
    x1 = [0,0,1,1]
    y1 = [1,0,0,1]
    
    x2 = [1,0,1,0]
    y2 = [1,0,0,1]
    
    assert sm.is_simple(x1, y1) == (True,-1,-1)
    assert sm.is_simple(x2, y2) == (False,1,3)
    
def test_project_point():
    
    v1 = (1,1)
    v2 = (2,2)
    v3 = (1,0)
    v4 = (1,2)
    
    assert np.allclose(sm.project_point(v1,v2), (1,1))
    assert np.allclose(sm.project_point(v1,v3), (1,0))
    assert np.allclose(sm.project_point(v1,v4), (0.6,1.2))
    assert np.allclose(sm.project_point(v3,v4), (0.2,0.4))

def test_centerline_series():
    
    x = [1,2,3,4,5]
    y = [2,3,5,5,6]
    
    assert np.allclose(sm.centerline_series(x,y), ([1,2,3.5,4,5],[2,3,4.5,5,6]))
    
def test_get_stationing():
    
    x = [1,2,3,4,5]
    y = [2,3,5,5,6]
    
    assert np.allclose(sm.get_stationing(x,y, project=True), [0,1.414213562373095,3.5355339059327364,4.242640687119285,5.656854249492379])
    assert np.allclose(sm.get_stationing(x,y, project=False), [0,1.4142135623730951,3.6502815398728847,4.650281539872885,6.06449510224598])
    
def test_monotonic_increasing():
    
    l1 = [1,2,3,3,4,5]
    l2 = [0,0,2,3,4,4,3,5]
    
    assert sm.monotonic_increasing(l1) == True
    assert sm.monotonic_increasing(l2) == False
    
def test_crawl_to_elevation():
    
    whys = [10,9,8,5,2,2,3,2,4,5,8,8.5,9,10,1,3,8]
    
    assert sm.crawl_to_elevation(whys, elevation=6, startInd=4) == [2,10]
    assert sm.crawl_to_elevation(whys, elevation=6, startInd=14) == [13,16]
    assert sm.crawl_to_elevation(whys, elevation=9, startInd=4) == [1,12]
    
def test_get_first():
    
    ser = pd.Series([1,4,5])
    assert sm.get_first(ser) == 1
    
def test_get_last():
    
    ser = pd.Series([1,4,5])
    assert sm.get_last(ser) == 5
    
def test_get_middle():
    
    ser = pd.Series([1,4,5])
    assert sm.get_middle(ser) == 3
    
def test_find_min_index():
    
    l1 = [1,3,2,4,-1,2.3,6]
    l2 = [6,1,3,-1,2,4,-1,2.3,6]
    
    assert sm.find_min_index(l1) == 4
    assert sm.find_min_index(l2) == 3
    
def test_find_max_index():
    
    l1 = [1,3,2,4,-1,2.3,6]
    l2 = [6,1,3,-1,2,4,-1,2.3,6]
    
    assert sm.find_max_index(l1) == 6
    assert sm.find_max_index(l2) == 0
    
def test_get_closest_index_by_value():
    
    l = [1,3,2,4,5,4,6]
    
    assert sm.get_closest_index_by_value(l, value=2.2) == 2
    
def test_get_nth_closest_index_by_value():
    
    l = [1,3,2,4,5,4,6]
    
    assert sm.get_nth_closest_index_by_value(l, value=2.2, n=2) == 1
    assert sm.get_nth_closest_index_by_value(l, value=2.2, n=3) == 0
    assert sm.get_nth_closest_index_by_value(l, value=2.2, n=4) == 3 or sm.get_nth_closest_index_by_value(l, value=2.2, n=4) == 5
    
def test_break_at_bankfull():
    
    exes = [0,1,2,3,4,5,6,7]
    whys = [5,4,3,2,1,1,2,3]
    
    assert np.allclose(sm.break_at_bankfull(exes,whys, bkfEl=3.5, startInd=4),
                       ([1.5,2,3,4,5,6,7,7],[3.5,3,2,1,1,2,3,3.5]))
    
    exes.reverse()
    whys.reverse()
    assert np.allclose(sm.break_at_bankfull(exes,whys, bkfEl=3.5, startInd=4),
                       ([7,7,6,5,4,3,2,1.5],[3.5,3,2,1,1,2,3,3.5]))
    
def test_make_countdict():
    
    l = [1,1,1,3,2,3,3,1,4,5,2,2]
    
    assert sm.make_countdict(l) == {1:4,3:3,2:3,4:1,5:1}
    
def test_strip_doubles():
    
    l = [1,1,1,3,2,3,3,1,4,5,2,2]
    
    assert sm.strip_doubles(l) == [1,3,2,3,1,4,5,2]
    
def test_make_monotonic():
    
    l = [1,1,1,3,2,3,3,1,4,5,2,2,0]
    
    assert sm.make_monotonic(l, increasing=True, removeDuplicates=False) == [1,1,1,3,3,3,4,5]
    assert sm.make_monotonic(l, increasing=True, removeDuplicates=True) == [1,3,4,5]
    assert sm.make_monotonic(l, increasing=False, removeDuplicates=False) == [1,1,1,1,0]
    assert sm.make_monotonic(l, increasing=False, removeDuplicates=True) == [1,0]
    
def test_diffreduce():
    
    l = [1,2,4,8]
    
    assert np.isclose(sm.diffreduce(l, delta=None), 1)
    assert np.isclose(sm.diffreduce(l, delta=2), 0.125)
    
    