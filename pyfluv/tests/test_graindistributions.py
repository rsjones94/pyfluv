"""
Unit tests for graindistributions.py with pytest
"""

# NOT DONE

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

def test_cumulative_sum(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    expected = [0, 0, 0, 0, 12.0, 12.0, 12.0, 16.0, 24.0, 27.0, 46.0, 58.0,
                76.0, 86.0, 96.0, 96.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0,
                100.0]

    assert np.allclose(grains.cumulative_sum(), expected)

def test_normalize_cumulative_sum(default_distribution):

    default_distribution = {key:val*4 for key,val in default_distribution.items()}
    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    expected = [0, 0, 0, 0, 12.0, 12.0, 12.0, 16.0, 24.0, 27.0, 46.0, 58.0,
                76.0, 86.0, 96.0, 96.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0,
                100.0]

    assert np.allclose(grains.normalize_cumulative_sum(), expected)

def test_dx_20(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    assert np.isclose(grains.dx(20), 4.8500000000000005)

def test_dx_50(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    assert np.isclose(grains.dx(50), 12.866666666666667)

def test_dx_75(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    assert np.isclose(grains.dx(75), 22.233333333333334)

def test_num_to_phi_metric():

    dist = []
    grains = gd.GrainDistribution(distr=dist, name=None, metric=True)
    assert np.isclose(grains.num_to_phi(2), -1.0)
    assert np.isclose(grains.num_to_phi(0.5), 1.0)

def test_num_to_phi_imperial():

    dist = []
    grains = gd.GrainDistribution(distr=dist, name=None, metric=False)
    assert np.isclose(grains.num_to_phi(2), -5.666756591884804)
    assert np.isclose(grains.num_to_phi(0.5), -3.6667565918848033)

def test_phi_to_num_metric():

    dist = []
    grains = gd.GrainDistribution(distr=dist, name=None, metric=True)
    assert np.isclose(grains.phi_to_num(2), 0.25)
    assert np.isclose(grains.num_to_phi(0.5), 1)

def test_phi_to_num_imperial():

    dist = []
    grains = gd.GrainDistribution(distr=dist, name=None, metric=False)
    assert np.isclose(grains.phi_to_num(2), 0.00984251968503937)
    assert np.isclose(grains.num_to_phi(0.5), -3.6667565918848033)

def test_phi_to_num_reconverts():

    dist = []
    grains = gd.GrainDistribution(distr=dist, name=None, metric=True)

    testX = np.arange(-5,5,1)
    for x in testX:
        assert np.isclose(grains.num_to_phi(grains.phi_to_num(float(x))), x)

def test_num_to_phi_reconverts():

    dist = []
    grains = gd.GrainDistribution(distr=dist, name=None, metric=True)

    testX = np.arange(0.1,5,0.1)
    for x in testX:
        assert np.isclose(grains.phi_to_num(grains.num_to_phi(x)), x)

def test_median(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    assert np.isclose(grains.median(), 12.866666666666667)

def test_mean(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    assert np.isclose(grains.mean(), 11.573383285807788)

def test_sorting(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    assert np.isclose(grains.sorting(), -1.7807474713885514)

def test_skewness(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    assert np.isclose(grains.skewness(), -0.32478588495139904)

def test_kurtosis(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    assert np.isclose(grains.kurtosis(), 1.5980624413178468)

def test_stddev(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    assert np.isclose(grains.stddev(), 13.06)

def test_make_countlist(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    expected = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                4.0, 4.0, 4.0, 4.0, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7,
                8.0, 8.0, 8.0, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3,
                11.3, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3, 11.3,
                11.3, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
                16.0, 16.0, 16.0, 22.6, 22.6, 22.6, 22.6, 22.6, 22.6, 22.6,
                22.6, 22.6, 22.6, 22.6, 22.6, 22.6, 22.6, 22.6, 22.6, 22.6,
                22.6, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0,
                32.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0,
                45.0, 90.0, 90.0, 1024.0, 1024.0]
    assert np.allclose(grains.make_countlist(), expected)

def test_bin_particles(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    expected = {'C': [0, 0], 'FM': [0.002, 0], 'MM': [0.0063, 0], 'CM': [0.02, 0],
                'FS': [0.063, 0], 'MS': [0.2, 12], 'CS': [0.63, 0], 'FG': [2, 12],
                'MG': [6.3, 34], 'CG': [20, 38], 'Cob': [63, 2], 'Bol': [200, 0],
                'LBol': [630, 2]}

    assert grains.bin_particles() == expected

def test_extract_binned_counts(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    expected = [0, 0, 0, 0, 0, 12, 0, 12, 34, 38, 2, 0, 2]

    assert np.allclose(grains.extract_binned_counts(), expected)

def test_extract_binned_cumsum(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    expected = [0, 0, 0, 0, 0, 12, 12, 24, 58, 96, 98, 98, 100]

    assert np.allclose(grains.extract_binned_cumsum(), expected)

def test_extract_unbinned_cumsum(default_distribution):

    grains = gd.GrainDistribution(distr=default_distribution, name=None, metric=True)
    expected = [0.0, 0.0, 0.0, 0.0, 12.0, 12.0, 12.0, 16.0, 24.0, 27.0, 46.0,
                58.0, 76.0, 86.0, 96.0, 96.0, 98.0, 98.0, 98.0, 98.0, 98.0,
                98.0, 100.0]

    assert np.allclose(grains.extract_unbinned_cumsum(), expected)
