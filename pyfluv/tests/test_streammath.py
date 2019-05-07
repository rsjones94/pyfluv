"""
Tests for streammath.py using pytest
"""

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
