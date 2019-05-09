"""
Tests for streammath.py using pytest
"""
import numpy as np
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
    seg2 = ((0,2),(2,0)) # barely touching - technically intersects
    seg3 = ((0,0.5),(2,0.25)) # obviously intersects
    seg4 = ((-1,0),(0,1)) # parallel
    seg5 = ((-1,0.25),(0,0.25)) # would intersect if extended further
    
    # assert sm.does_intersect(seg1, seg2) == True # test fails
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
    
    assert False
