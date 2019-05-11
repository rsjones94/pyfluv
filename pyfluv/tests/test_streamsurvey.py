"""
Unit tests for streamsurvey.py using pytest
"""

import os

from copy import deepcopy
import pytest

from .. import streamsurvey as ss


@pytest.fixture(scope='module')
def default_survey():
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '..', 'data/myr5_survey_adjusted.csv')
    
    return ss.StreamSurvey(filename, sep=',', metric=False, keywords=None, colRelations=None)

@pytest.fixture()
def default_descs(default_survey):
    
    survey = deepcopy(default_survey)
    return [l.desc for l in survey.pack_shots()[0:10]]

@pytest.fixture()
def default_parseDict():
    
    return {'Profile':'pro',
            'Thalweg':'thw',
            'Riffle':'ri',
            'Run':'ru',
            'Pool':'po',
            'Glide':'gl',
            'Water Surface':'ws',
            'Bankfull':'bkf',
            'Top of Bank':'tob',
            'Cross Section':'xs',
            'Structure':'str',
            'breakChar':'-',
            'commentChar':'_'
            }

def test_StreamSurvey_get_profile_objects(default_survey):
    
    survey = deepcopy(default_survey)
    
    expectedNames = ['proTrib2SUP', 'proTrib2', 'proR1', 'proTrib1', 'proR3', 'proR2']
    names = [l.name for l in survey.get_profile_objects()]
    
    assert names == expectedNames
    
def test_StreamSurvey_get_cross_objects(default_survey):
    
    survey = deepcopy(default_survey)
    
    expectedNames = ['xsTrib2SUPpo', 'xsTrib2SUPri', 'xsTrib2po', 'xsTrib2ri',
                     'xsR1po', 'xsR1ri', 'xsR1SUP', 'xsTrib1po', 'xsTrib1ri',
                     'xsR3ri', 'xsR3po', 'xsR3sup', 'xsR2ri', 'xsR2po']
    names = [l.name for l in survey.get_cross_objects()]
    
    assert names == expectedNames
    
def test_StreamSurvey_get_names(default_survey):
    
    survey = deepcopy(default_survey)
    
    expectedDict = {'Profiles': {'proTrib2SUP': 121,
                    'proTrib2': 90,
                    'proR1': 133,
                    'proTrib1': 141,
                    'proR3': 157,
                    'proR2': 164},
                    'Cross Sections': {'xsTrib2SUPpo': 49,
                    'xsTrib2SUPri': 45,
                    'xsTrib2po': 35,
                    'xsTrib2ri': 28,
                    'xsR1po': 58,
                    'xsR1ri': 52,
                    'xsR1SUP': 38,
                    'xsTrib1po': 49,
                    'xsTrib1ri': 38,
                    'xsR3ri': 49,
                    'xsR3po': 73,
                    'xsR3sup': 72,
                    'xsR2ri': 32,
                    'xsR2po': 53}}
    
    assert survey.get_names() == expectedDict
    
def test_Parser_dict_split(default_parseDict, default_descs):

    parser = ss.Parser(parseDict=default_parseDict)
    
    expectedResult = [{'name': 'env1', 'descriptors': [], 'comment': None},
                      {'name': 'Trib2SUPbpin', 'descriptors': [], 'comment': None},
                      {'name': 'proTrib2SUP', 'descriptors': ['bri'], 'comment': 'testcomment'},
                      {'name': 'proTrib2SUP', 'descriptors': ['ws'], 'comment': None},
                      {'name': 'proTrib2SUP', 'descriptors': ['tob'], 'comment': None},
                      {'name': 'proTrib2SUP', 'descriptors': ['bkf'], 'comment': None},
                      {'name': 'proTrib2SUP', 'descriptors': ['ri'], 'comment': None},
                      {'name': 'proTrib2SUP', 'descriptors': ['ri'], 'comment': None},
                      {'name': 'proTrib2SUP', 'descriptors': ['ri'], 'comment': None},
                      {'name': 'proTrib2SUP', 'descriptors': ['ri'], 'comment': None}]
    
    assert [parser.dict_split(l) for l in default_descs] == expectedResult
    
def test_Parser_string_is_in():
    
    s1 = 'foo'
    s2 = 'bar'
    s3 = 'bah'
    s4 = 'rah'
    
    parser = ss.Parser(parseDict=None)
    
    assert parser.string_is_in(s1, s2) == False
    assert parser.string_is_in(s1, s2+s1+s2) == True
    assert parser.string_is_in(s1, s2+s3) == False
    assert parser.string_is_in(s2, s3[0:2]+s4) == True
    
def test_Parser_key_is_in(default_parseDict):
    
    parser = ss.Parser(parseDict=default_parseDict)
    
    assert parser.key_is_in('Riffle', 'ri') == True
    assert parser.key_is_in('Riffle', 'epo-bri') == True
    assert parser.key_is_in('Riffle', 'pori') == True
    assert parser.key_is_in('Riffle','po') == False
    assert parser.key_is_in('commentChar','_') == True
    assert parser.key_is_in('commentChar','pro1-bri_discard') == True
    assert parser.key_is_in('Cross Section','bri-xs') == True
    assert parser.key_is_in('Cross Section','bri') == False

def test_Parser_get_meaning(default_parseDict, default_descs):
    

    parser = ss.Parser(parseDict=default_parseDict)
    
    expectedResult = [{'type': None,
                       'morphs': [],
                       'name': 'env1',
                       'full': 'env1'},
                      {'type': None,
                       'morphs': [],
                       'name': 'Trib2SUPbpin',
                       'full': 'Trib2SUPbpin'},
                      {'type': 'Profile',
                       'morphs': ['Riffle'],
                       'name': 'proTrib2SUP',
                       'full': 'proTrib2SUP-bri_testcomment'},
                      {'type': 'Profile',
                       'morphs': ['Water Surface'],
                       'name': 'proTrib2SUP',
                       'full': 'proTrib2SUP-ws'},
                      {'type': 'Profile',
                       'morphs': ['Top of Bank'],
                       'name': 'proTrib2SUP',
                       'full': 'proTrib2SUP-tob'},
                      {'type': 'Profile',
                       'morphs': ['Bankfull'],
                       'name': 'proTrib2SUP',
                       'full': 'proTrib2SUP-bkf'},
                      {'type': 'Profile',
                       'morphs': ['Riffle'],
                       'name': 'proTrib2SUP',
                       'full': 'proTrib2SUP-ri'},
                      {'type': 'Profile',
                       'morphs': ['Riffle'],
                       'name': 'proTrib2SUP',
                       'full': 'proTrib2SUP-ri'},
                      {'type': 'Profile',
                       'morphs': ['Riffle'],
                       'name': 'proTrib2SUP',
                       'full': 'proTrib2SUP-ri'},
                      {'type': 'Profile',
                       'morphs': ['Riffle'],
                       'name': 'proTrib2SUP',
                       'full': 'proTrib2SUP-ri'}]
    
    assert [parser.get_meaning(l) for l in default_descs] == expectedResult
    
    