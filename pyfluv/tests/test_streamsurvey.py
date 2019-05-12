"""
Unit tests for streamsurvey.py using pytest. Coverage is incomplete.
"""

from copy import deepcopy
import os

import numpy as np
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
def default_cross_shots(default_survey):
    """
    List of Shot objects representing one cross section.
    """

    survey = deepcopy(default_survey)
    return survey.crossSections[4]

@pytest.fixture()
def default_pro_shots(default_survey):
    """
    List of Shot objects representing one cross section.
    """

    survey = deepcopy(default_survey)
    return survey.profiles[1]

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

@pytest.fixture()
def default_colRelations():

    return {'shotnum': 'Name',
            'whys': 'Northing',
            'exes': 'Easting',
            'zees': 'Elevation',
            'desc': 'Description'}

##### end of fixtures #####

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
    
def test_StreamSurvey_group_by_name(default_survey):

    survey = deepcopy(default_survey)
    survey.group_by_name()
    
    assert len(survey.crossSections) == 14
    assert len(survey.profiles) == 6

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

def test_PackGroupCross_name(default_parseDict, default_cross_shots):

    group = ss.PackGroupCross(packGroup=default_cross_shots, keywords=default_parseDict,
                              metric=False, stripName=False)

    assert group.name == 'xsR1po'

def test_PackGroupCross_stripName(default_parseDict, default_cross_shots):

    group = ss.PackGroupCross(packGroup=default_cross_shots, keywords=default_parseDict,
                              metric=False, stripName=True)

    assert group.name == 'R1po'

def test_PackGroupCross_pull_atts(default_parseDict, default_cross_shots):

    group = ss.PackGroupCross(packGroup=default_cross_shots, keywords=default_parseDict,
                              metric=False, stripName=False)

    expected = {'Water Surface': 622.4700706,
                'Bankfull': 625.1516846,
                'Top of Bank': None,
                'Thalweg': None}

    result = group.pull_atts()

    compare = zip(result.values(), expected.values())

    assert all([l[0] == l[1] or np.isclose(l[0],l[1]) for l in compare])
    # == must come first as compairng Nonetypes throws error using np.isclose

def test_PackGroupCross_pull_xs_survey_coords(default_parseDict, default_cross_shots):

    group = ss.PackGroupCross(packGroup=default_cross_shots, keywords=default_parseDict,
                              metric=False, stripName=False)

    expected = ([1128.427929,
                 1128.464363,
                 1128.177082,
                 1127.763433,
                 1127.083249,
                 1126.5944849999998,
                 1125.680002,
                 1124.7166710000001,
                 1124.218648,
                 1123.571651,
                 1122.882965,
                 1122.405419,
                 1121.610468,
                 1121.520078,
                 1122.026827,
                 1121.487216,
                 1121.15062,
                 1121.4320810000002,
                 1121.019289,
                 1120.871426,
                 1120.629267,
                 1120.507144,
                 1120.635377,
                 1119.674224,
                 1119.0655470000002,
                 1118.6129720000001,
                 1117.966646,
                 1117.805657,
                 1117.379492,
                 1116.732067,
                 1116.201157,
                 1115.4586199999999,
                 1115.116892,
                 1114.802704,
                 1114.606047,
                 1114.414427,
                 1113.759025,
                 1113.0707619999998,
                 1112.1178289999998,
                 1111.738698,
                 1110.791812,
                 1110.011598,
                 1109.414785,
                 1108.496069,
                 1107.470834,
                 1106.56145,
                 1105.6297359999999,
                 1104.773267,
                 1104.0945880000002,
                 1103.8441599999999,
                 1103.2329109999998,
                 1103.193012,
                 1102.107365,
                 1101.632891,
                 1101.0056710000001,
                 1099.990485,
                 1099.16572,
                 1099.307836],
                [999.806725,
                 999.7464779999999,
                 998.7691460000001,
                 996.9205539999999,
                 994.9607490000001,
                 993.0922609999999,
                 990.263224,
                 987.41335,
                 985.3733609999999,
                 982.9297369999999,
                 980.722015,
                 978.840617,
                 976.880109,
                 975.476143,
                 974.5133480000001,
                 973.2877880000001,
                 972.9161349999999,
                 972.592071,
                 971.1550779999999,
                 970.8673470000001,
                 970.8022,
                 970.246352,
                 970.422143,
                 968.391148,
                 966.37816,
                 964.274112,
                 962.283757,
                 961.289159,
                 960.14863,
                 958.077048,
                 956.1876609999999,
                 955.1788369999999,
                 953.9845970000001,
                 952.7257060000001,
                 951.9384130000001,
                 950.8384560000001,
                 949.424819,
                 947.2358300000001,
                 945.131117,
                 943.308011,
                 940.589066,
                 937.6599539999999,
                 935.5166529999999,
                 933.1964449999999,
                 930.1041779999999,
                 927.3163789999999,
                 924.4923539999999,
                 921.620735,
                 919.702059,
                 917.6943359999999,
                 915.7198199999999,
                 915.0424429999999,
                 912.4703099999999,
                 910.758876,
                 907.9369949999999,
                 905.2333039999999,
                 903.2402060000001,
                 903.187294],
                [628.6831116000001,
                 628.5441726,
                 628.5507476,
                 628.4901716,
                 628.4533276000001,
                 628.4870236,
                 628.4011146,
                 628.2018506000001,
                 628.2034566000001,
                 627.4622106,
                 626.8385636,
                 626.2195076,
                 625.4808446000001,
                 625.3335606,
                 625.1516846,
                 624.4758066,
                 624.0726986000001,
                 622.8788026000001,
                 622.7301186000001,
                 622.7894636,
                 622.3827236000001,
                 622.4110836,
                 620.7665806,
                 620.5011595999999,
                 620.0497056,
                 619.7858446,
                 619.8791006,
                 619.8806386,
                 619.9713996,
                 620.3782396,
                 620.7905106000001,
                 621.1736356,
                 621.6030766,
                 622.4700706,
                 623.5450986000001,
                 624.1354816,
                 624.5809896000001,
                 624.5871466,
                 624.8023346,
                 625.5793156000001,
                 625.9522736,
                 625.9480796,
                 625.8619146000001,
                 625.5222746000001,
                 625.4140646000001,
                 625.4788156000001,
                 624.9259406,
                 624.6433036000001,
                 624.1920606000001,
                 623.6351316,
                 623.0837846,
                 623.0643206,
                 625.1856236000001,
                 627.0366306,
                 627.2612536,
                 627.4128736,
                 627.2806186,
                 627.5000636],
                ['xsR1po-lbp',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po-bkf',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po-ws',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po',
                 'xsR1po-rbp'])

    result = group.pull_xs_survey_coords()

    compare = zip(expected, result)

    assert [l[0] == l[1] or np.allclose(l[0],l[1]) for l in compare]

def test_PackGroupCross_get_cross_morph(default_parseDict, default_cross_shots):

    group = ss.PackGroupCross(packGroup=default_cross_shots, keywords=default_parseDict,
                              metric=False, stripName=False)

    assert group.get_cross_morph() == 'Pool'

def test_PackGroupCross_create_cross_object(default_parseDict, default_cross_shots):

    group = ss.PackGroupCross(packGroup=default_cross_shots, keywords=default_parseDict,
                              metric=False, stripName=False)

    cross = group.create_cross_object(guessType=True, project=True)

    assert cross.name == 'xsR1po'

def test_PackGroupPro_name(default_parseDict, default_pro_shots, default_colRelations):

    group = ss.PackGroupPro(packGroup=default_pro_shots, keywords=default_parseDict,
                            colRelations=default_colRelations, metric=False, stripName=False)

    assert group.name == 'proTrib2'

def test_PackGroupPro_stripName(default_parseDict, default_pro_shots, default_colRelations):

    group = ss.PackGroupPro(packGroup=default_pro_shots, keywords=default_parseDict,
                            colRelations=default_colRelations, metric=False, stripName=True)

    assert group.name == 'Trib2'
    
def test_PackGroupPro_substrate_filter(default_parseDict, default_pro_shots, default_colRelations):

    group = ss.PackGroupPro(packGroup=default_pro_shots, keywords=default_parseDict,
                            colRelations=default_colRelations, metric=False, stripName=False)
    
    expectedShotNums = [224, 228, 229, 231, 232, 234, 235, 236, 237, 238, 243,
                        244, 248, 249, 250, 251, 255, 256, 258, 259, 260, 261,
                        262, 265, 266, 268, 271, 273, 274, 276, 277, 278, 280,
                        284, 285, 287, 324, 325, 326, 329, 330, 331, 332, 333,
                        334, 335, 336, 339, 340, 341, 342, 343, 344, 345, 349,
                        350, 351]
    
    result = group.substrate_filter()

    assert [l.shotnum for l in result] == expectedShotNums
    
def test_PackGroupPro_make_uCols(default_parseDict, default_pro_shots, default_colRelations):

    group = ss.PackGroupPro(packGroup=default_pro_shots, keywords=default_parseDict,
                            colRelations=default_colRelations, metric=False, stripName=False)
    
    assert group.make_uCols() == ['Thalweg', 'Riffle', 'Run', 'Pool', 'Glide',
                                  'Water Surface', 'Bankfull', 'Top of Bank',
                                  'Cross Section', 'Structure']
def test_PackGroupPro_make_sCols(default_parseDict, default_pro_shots, default_colRelations):

    group = ss.PackGroupPro(packGroup=default_pro_shots, keywords=default_parseDict,
                            colRelations=default_colRelations, metric=False, stripName=False)
    
    assert group.make_sCols() == ['shotnum', 'whys', 'exes', 'zees', 'desc']
    
def test_PackGroupPro_backstack_data(default_parseDict, default_pro_shots, default_colRelations):
    """
    Not testing every column, but enough to be pretty confident it's okay.
    """

    group = ss.PackGroupPro(packGroup=default_pro_shots, keywords=default_parseDict,
                            colRelations=default_colRelations, metric=False, stripName=False)
    
    expectedShotNums = [224, 228, 229, 231, 232, 234, 235, 236, 237, 238, 243,
                        244, 248, 249, 250, 251, 255, 256, 258, 259, 260, 261,
                        262, 265, 266, 268, 271, 273, 274, 276, 277, 278, 280,
                        284, 285, 287, 324, 325, 326, 329, 330, 331, 332, 333,
                        334, 335, 336, 339, 340, 341, 342, 343, 344, 345, 349,
                        350, 351]
    
    expectedDescs = [['proTrib2-bri', 'proTrib2-ws', 'proTrib2-bkf', 'proTrib2-tob'],
                     ['proTrib2-ri'], ['proTrib2-ri', 'proTrib2-ws'], ['proTrib2-ri-jam'],
                     ['proTrib2-eri', 'proTrib2-ws'], ['proTrib2-thw'], ['proTrib2-thw'],
                     ['proTrib2-thw'], ['proTrib2-thw'],
                     ['proTrib2-thw', 'proTrib2-ws', 'proTrib2-bkf', 'proTrib2-tob'],
                     ['proTrib2-thw'],
                     ['proTrib2-bri', 'proTrib2-ws', 'proTrib2-bkf', 'proTrib2-tob'],
                     ['proTrib2-ri'], ['proTrib2-ri'], ['proTrib2-ri'],
                     ['proTrib2-eri', 'proTrib2-ws', 'proTrib2-bkf', 'proTrib2-tob'],
                     ['proTrib2-thw'], ['proTrib2-bpo', 'proTrib2-ws'], ['proTrib2-po'],
                     ['proTrib2-po'], ['proTrib2-po'], ['proTrib2-po'], ['proTrib2-po'],
                     ['proTrib2-po'], ['proTrib2-epo', 'proTrib2-ws'],
                     ['proTrib2-thw', 'proTrib2-bkf', 'proTrib2-tob'],
                     ['proTrib2-thw', 'proTrib2-ws'], ['proTrib2-thw'],
                     ['proTrib2-thw', 'proTrib2-ws'], ['proTrib2-thw'], ['proTrib2-thw'],
                     ['proTrib2-thw', 'proTrib2-ws'],
                     ['proTrib2-bri', 'proTrib2-ws', 'proTrib2-bkf', 'proTrib2-tob'],
                     ['proTrib2-ri'], ['proTrib2-eri', 'proTrib2-ws'],
                     ['proTrib2-thw-xsp', 'proTrib2-thw-xsp'], ['proTrib2-thw'],
                     ['proTrib2-thw'], ['proTrib2-thw', 'proTrib2-ws', 'proTrib2-bkf'],
                     ['proTrib2-thw'], ['proTrib2-thw'], ['proTrib2-thw'], ['proTrib2-thw'],
                     ['proTrib2-thw'], ['proTrib2-thw'], ['proTrib2-thw'],
                     ['proTrib2-thw', 'proTrib2-ws', 'proTrib2-bkf'], ['proTrib2-thw'],
                     ['proTrib2-thw'], ['proTrib2-thw'], ['proTrib2-thw'], ['proTrib2-thw'],
                     ['proTrib2-thw'], ['proTrib2-thw', 'proTrib2-ws', 'proTrib2-bkf'],
                     ['proTrib2-thw'], ['proTrib2-thw-xsr', 'proTrib2-thw-xsr'],
                     ['proTrib2-thw', 'proTrib2-ws', 'proTrib2-bkf']]
    
    expectedZees = [631.5057635, 630.7317495, 631.1116945, 630.5337245000001, 630.2827424999999,
                    629.8488765, 630.1125275, 630.1415705000001, 630.0878225, 630.0295434999999,
                    629.6968065000001, 629.5955755, 629.3665485, 629.0099974999999, 628.9866415,
                    628.7991405, 628.7825035, 628.5776295, 628.5279535, 628.3069385, 627.8916445,
                    627.8906625, 627.5731405, 628.2948755, 628.5442445, 628.4785264999999,
                    627.7730285, 627.4289144999999, 627.3112725, 627.2066065, 627.4732535,
                    627.7877434999999, 627.5898865, 627.4997384999999, 627.3651365000001,
                    627.0503655, 626.9888235000001, 626.8710225, 627.5112995000001, 627.0991205,
                    626.9467775, 626.6418215, 626.8584745, 626.6145845, 626.8309085, 626.7694125,
                    626.6960285, 626.8416905, 626.9483644999999, 626.7019015, 626.3528325,
                    626.0296465, 626.4759625, 626.7990555, 626.6440635, 626.6360965, 626.8405865]
    
    stacked = group.backstack_data()
    
    assert all(stacked['shotnum'] == expectedShotNums)
    assert all(stacked['desc'] == expectedDescs)
    assert all(np.isclose(stacked['zees'], expectedZees))
    
def test_PackGroupPro_create_pro_object(default_parseDict, default_pro_shots, default_colRelations):

    group = ss.PackGroupPro(packGroup=default_pro_shots, keywords=default_parseDict,
                            colRelations=default_colRelations, metric=False, stripName=False)
    
    pro = group.create_pro_object(assignMethod='backstack')
    
    assert pro.name == 'proTrib2'