"""
Tests for reference.py using pytest.
"""

import numpy as np
import pytest

from .. import monitoringdata


@pytest.fixture()
def default_ref():
    """
    This brings in a Reference object for us so we don't need to explicitly
    import the module
    """
    return monitoringdata.eco71()

def remove_nests(l):
    """
    Only flattens one level
    """
    
    flattened = []
    for x in l:
        try:
            for y in x:
                flattened.append(y)
        except TypeError:
            flattened.append(x)
            
    return flattened

##### end of fixtures and setup #####
    
def test_reference__init__(default_ref):
    
    assert default_ref.identify_draincol() == 'Drainage area'
    
def test_reference_fit(default_ref):
    
    expected = remove_nests((np.array([16.46339749,  0.78404166]), 0.9816838429999362))
    assert np.allclose(remove_nests(default_ref.fit('Bankfull area')), expected)