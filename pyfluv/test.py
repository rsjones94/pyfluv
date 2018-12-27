# junk code for quick tests
import matplotlib.pyplot as plt

import streammath as sm
import streamgeometry as strgeo
import graindistributions as grain


distr = {2:3,
         4:1,
         7:4,
         5:2,
         0.01:3,
         0.4:4
        }

work = grain.GrainDistribution(distr,name ='Work')

exes = [0,1,4,3,5,6,7,6,8]
whys = [0,0,2,5,4,6,7,10,11]
zees = [5,4,3,4,2,1,1,4,7]

myXS = strgeo.CrossSection(exes,whys,zees,name='TestChannel',thwStation=4)

a = myXS.bkf_binary_search('bkfW', 1)
print(a)

myXS.bkfEl = a
myXS.calculate_bankfull_statistics()

myXS.qplot(showBkf=True,showCutSection=True)
