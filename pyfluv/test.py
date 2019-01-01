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

myXS = strgeo.CrossSection(exes,whys,zees,name='TestChannel')

a = myXS.bkf_binary_search('bkfA', 10)
print(a)

myXS.bkfEl = a
myXS.calculate_bankfull_statistics()

myXS.qplot(showBkf=True,showCutSection=True)

myXS.manN = 0.03
myXS.waterSlope = 0.02
flowR = myXS.bkf_by_flow_release()

myXS.bkfEl = flowR
myXS.calculate_bankfull_statistics()
myXS.qplot(showBkf=True,showCutSection=True)

res = myXS._flow_release_array(absolute = True)
els = res[0]
dqdh = res[1]
plt.figure()
plt.plot(els,dqdh)