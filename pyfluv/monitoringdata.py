"""
Contains methods to pull in packaged stream monitoring data.
"""

import os

import pandas as pd

from . import streamsurvey
from . import graindistributions
from . import reference

def piney_survey():
    """
    Returns a StreamSurvey object containing the geomorphic survey for the
    Year 5 (2018) summer monitoring at the West Piney River mitigation site.
    """
    selfloc = os.path.dirname(os.path.realpath(__file__))
    file = r'\data\wpr_myr5_survey_adjusted.csv'
    wpr = streamsurvey.StreamSurvey(selfloc+file,
                                    sep=',',
                                    metric=False,
                                    keywords=None,
                                    colRelations=None)
    return wpr

def piney_pebbles():
    """
    Returns a list of GrainDistribution objects containing the pebble survey for
    the Year 5 (2018) summer monitoring at the West Piney River mitigation site.

    Note that bedrock calls are converted to large boulders (>1024mm).
    """
    selfloc = os.path.dirname(os.path.realpath(__file__))
    file = r'\data\wpr_myr5_pebbles.csv'
    pebbles = pd.read_csv(selfloc+file, sep=',')
    sizes = list(pebbles['Minimum Size (mm)'])

    def try_float(value):
        try:
            return float(value)
        except ValueError:
            return value

    sizes = [try_float(val) for val in sizes]

    def dict_zip(el1, el2):
        return {l1:l2 for l1, l2 in zip(el1, el2)}

    pebbleCounts = []
    for colName, data in pebbles.iteritems():
        if colName != 'Minimum Size (mm)':
            data = dict_zip(sizes, list(data))
            gd = graindistributions.GrainDistribution(distr=data, name=colName, metric=True)
            pebbleCounts.append(gd)
    return pebbleCounts

def eco71():
    """
    Returns a Reference object containing reference reach data for ecoregion 71
    (the ecoregion that west piney is in). Data collected from
    https://www.tn.gov/content/dam/tn/environment/water/documents/wr_wq_regional-curves-ecoregion711.pdf
    """
    selfloc = os.path.dirname(os.path.realpath(__file__))
    file = r'\data\eco71.csv'
    eco = pd.read_csv(selfloc+file, sep=',', encoding="ISO-8859-1")
    ecoReference = reference.Reference(reaches=eco, eco=71)
    return ecoReference
