"""
Unit tests for graindistributions.py with pytest
"""

import numpy as np
import pytest

from .. import monitoringdata


@pytest.fixture()
def default_dist():

    return monitoringdata.standard_pebbles()[0]

##### end fixtures #####
    
def test_bedrock_coercion(default_dist):
    """
    Failing
    """
    
    expected = {0.0: 0, 0.062: 0, 0.125: 0, 0.25: 0, 0.5: 12.0, 1.0: 0,
                2.0: 0, 4.0: 4.0, 5.7: 8.0, 8.0: 3.0, 11.3: 19.0, 16.0: 12.0,
                22.6: 18.0, 32.0: 10.0, 45.0: 10.0, 64.0: 0, 90.0: 2.0,
                128.0: 0, 180.0: 0, 256.0: 0, 362.0: 0, 512.0: 0, 1024.0: 2}
    
    assert default_dist == expected
    
