"""
Tests for monitoringdata.py using pytest
"""

from .. import monitoringdata


def test_standard_survey():
    """
    Check to see if we get a Survey object back and if it has all the profiles
    and cross sections
    """
    survey = monitoringdata.standard_survey()
    testPros = ['proTrib2SUP', 'proTrib2', 'proR1', 'proTrib1', 'proR3', 'proR2']
    testCrosses = ['xsTrib2SUPpo', 'xsTrib2SUPri', 'xsTrib2po', 'xsTrib2ri',
                   'xsR1po', 'xsR1ri', 'xsR1SUP', 'xsTrib1po', 'xsTrib1ri',
                   'xsR3ri', 'xsR3po', 'xsR3sup', 'xsR2ri', 'xsR2po']

    getPros = list(survey.get_names()['Profiles'].keys())
    getCrosses = list(survey.get_names()['Cross Sections'].keys())

    assert getPros == testPros and getCrosses == testCrosses
    
def test_standard_pebbles():
    """
    Check to see if we get a list of GraindDistribution objects length 10
    """
    
    checkList = ['Reach 1 XS',
                 'Reach 2 XS',
                 'Reach 3 XS',
                 'Trib1 XS',
                 'Trib2 XS',
                 'R1 Profile',
                 'Reach 2 Profile',
                 'Reach 3 Profile',
                 'Trib1 Profile',
                 'Trib2 Profile']
    
    pebs = monitoringdata.standard_pebbles()
    pebNames = [l.name for l in pebs]
    
    assert pebNames == checkList
    
def test_eco71():
    """
    Test that we have a Reference object for ecoregion 71
    """
    curve = monitoringdata.eco71()
    assert curve.eco == 71
