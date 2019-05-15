"""
Unit tests for graindistributions.py with pytest
"""

# NOT DONE

import numpy as np
import pytest

from .. import monitoringdata
from .. import streamexceptions as se
from .. import streamprofiles as sp

@pytest.fixture()
def simple_df():
    
    exes = [float(l) for l in '0 1 2 3 4 5 6 7 8 9 10'.split()]
    whys = exes.copy()
    zees = [float(l) for l in '10 9 8 9 9 8 7 8 8 7 6'.split()]
    
    df = {'exes':exes, 'whys':whys, 'Thalweg':zees}
    return df

@pytest.fixture()
def simple_profile(simple_df):
    
    return sp.Profile(df=simple_df, name='Simple Profile', metric=False)

@pytest.fixture()
def featured_df():
    
    exes = [float(l) for l in '0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'.split()]
    whys = exes.copy()
    zees = [float(l) for l in '10 9 8 8 9 8 7 7 8 7 6 6 7 6 5 5 6 5 4 4 5'.split()]
    ri = ['Ri', 'Ri', None, None, 'Ri', 'Ri', None, None, 'Ri', 'Ri', None, None,
          'Ri', 'Ri', None, None, 'Ri', 'Ri', None, None, 'Ri']
    po = ['Po' if l == None else None for l in ri]
    
    df = {'exes':exes, 'whys':whys, 'Thalweg':zees, 'Riffle':ri, 'Pool':po}
    return df

@pytest.fixture()
def featured_survey(featured_df):
    
    return sp.Profile(df=featured_df, name='Simple Profile', metric=False)

@pytest.fixture(scope='module')
def default_survey():
    
    return monitoringdata.standard_survey().get_pro_objects()

##### end fixtures #####
    
def test__init__bad_columns():
    
    exes = [1,2]
    df = {'exes':exes, 'whys':exes.copy(), 'zees':exes.copy()}
    
    with pytest.raises(se.InputError) as e_info:
        e = sp.Profile(df=df, name=None, metric=False)
        
def test__init__good_columns():
    """
    Make sure the error check for in test__init__bad_columns() is NOT raised
    """
    
    exes = [1,2]
    df = {'exes':exes, 'whys':exes.copy(), 'Thalweg':exes.copy()}
    
    try:
        e = sp.Profile(df=df, name=None, metric=False)
    except se.InputError:
        pytest.fail('Unepected streamexceptions InputError')
        
def test__init__creates_station(simple_profile):
    
    assert 'Station' in simple_profile.filldf.columns
    
def test_generate_stationing(simple_profile):
    """
    __init__ should invoke self.generate_stationing(), but we'll call it again
    since we're just testing if generate stationing returns the correct result
    """
    
    simple_profile.generate_stationing()
    
    assert np.allclose(simple_profile.filldf['Station'], [0.0, 1.4142135623730951,
                       2.8284271247461903, 4.242640687119286, 5.656854249492381,
                       7.0710678118654755, 8.485281374238571, 9.899494936611667,
                       11.313708498984763, 12.727922061357859, 14.142135623730955])
        
def test__init__creates_correct_haveCols(featured_survey):
    
    assert all([l in featured_survey.haveCols for l in ['Riffle', 'Pool']])
    
def test__init__rejects_invalid_substrate(featured_df):
    
    gl = [None for l in featured_df['Riffle']]
    featured_df['Glide'] = 10
    featured_df['Pool'] = 10
    
    with pytest.raises(se.InputError) as e_info:
        e = sp.Profile(df=featured_df, name=None, metric=False)