"""
Unit tests for graindistributions.py with pytest
"""

import numpy as np
import pytest

from .. import graindistributions as gd

@pytest.fixture()
def default_distribution():
    
    return {0.0: 0, 0.062: 0, 0.125: 0, 0.25: 0, 0.5: 12.0, 1.0: 0,
            2.0: 0, 4.0: 4.0, 5.7: 8.0, 8.0: 3.0, 11.3: 19.0, 16.0: 12.0,
            22.6: 18.0, 32.0: 10.0, 45.0: 10.0, 64.0: 0, 90.0: 2.0,
            128.0: 0, 180.0: 0, 256.0: 0, 362.0: 0, 512.0: 0, 1024.0: 0,
            'Bedrock':2}

##### end fixtures #####

def test_bedrock_coerce_bedrock_dict(default_distribution):
    
    distribution = default_distribution
    expected = {0.0: 0, 0.062: 0, 0.125: 0, 0.25: 0, 0.5: 12.0, 1.0: 0,
                2.0: 0, 4.0: 4.0, 5.7: 8.0, 8.0: 3.0, 11.3: 19.0, 16.0: 12.0,
                22.6: 18.0, 32.0: 10.0, 45.0: 10.0, 64.0: 0, 90.0: 2.0,
                128.0: 0, 180.0: 0, 256.0: 0, 362.0: 0, 512.0: 0, 1024.0: 2}
    grains = gd.GrainDistribution(distr={}, name=None, metric=True)
    
    assert grains.coerce_bedrock(distribution=distribution) == expected
    
def test__init__coerces_list_to_dict():
    
    distribution = [11.3, 11.3, 2, 2, 11.3, 5, 7, 'Bedrock', 2, 'Bedrock']
    expected = {11.3:3, 2:3, 5:1, 7:1, 40.31496:2}
    grains = gd.GrainDistribution(distr=distribution, name=None, metric=False)
    
    assert grains.distr == expected
    
def test__init__with_list():
    
    distribution = [11.3, 11.3, 2, 2, 11.3, 5, 7, 2]
    grains = gd.GrainDistribution(distr=distribution, name=None, metric=False)
    expected = {11.3:3, 2:3, 5:1, 7:1}
    
    assert grains.distr == expected
    
def test__init__sorts_dict():
    """
    Isn't it neat that dicts are ordered now
    """
    
    distribution = [11.3, 11.3, 2, 2, 11.3, 5, 7, 2]
    grains = gd.GrainDistribution(distr=distribution, name=None, metric=False)
    keys = list(grains.distr.keys())
    keyPairs = zip(keys[:-1],keys[1:])
    
    assert all(pair[0] < pair[1] for pair in keyPairs)