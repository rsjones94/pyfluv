# junk code for quick tests
import matplotlib.pyplot as plt
import numpy as np

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

exes = [0,1,2,3,4,4,4,4,5,9,10,14,15,15,15,15,16,17,18,19]
whys = exes
zees = [99,99,97,95,90,85,79,72,72,55,55,72,72,79,85,90,95,97,99,99]

myXS = strgeo.CrossSection(exes,whys,zees,name='TestChannel')

newEl = myXS.find_floodplain_elevation()
myXS.bkfEl = newEl
myXS.calculate_bankfull_statistics()
myXS.qplot(showBkf=True)

res = myXS.attribute_list('bkfA')
els = res[0]
atts = res[1]
dAtts = np.diff(atts)/0.1
ddAtts = np.diff(dAtts)/0.1

#plt.plot(atts,els)
plt.plot(dAtts,els[:-1])
plt.plot(ddAtts,els[1:-1])

plt.figure()
plt.plot(els,atts)
plt.plot(els[:-1],dAtts)
plt.plot(els[1:-1],ddAtts)

"""
res = myXS.attribute_list(attribute = 'bkfW')
els = res[0]
absAreas = res[1]

relAreas = [1]
for i in range(1,len(absAreas)):
    relAreas.append(absAreas[i]/absAreas[i-1])

dAbsAreas = np.diff(absAreas)/0.1 # units: sq ft / ft
dRelAreas = np.diff(relAreas)/0.1 # units: ft^-1

ddAbsAreas = np.diff(dAbsAreas) # units: sq ft / ft / ft

plt.figure()
plt.plot(els,absAreas)
plt.title('Absolute')
plt.figure()
plt.plot(els[:(len(els)-1)],dAbsAreas)
plt.title('First Derivative Absolute')
plt.figure()
plt.plot(els[:(len(els)-2)],ddAbsAreas)
plt.title('Second Derivative Absolute')

plt.figure()
plt.plot(els,relAreas)
plt.title('Relative Areas')
plt.figure()
plt.plot(els[:(len(els)-1)],dRelAreas)
plt.title('First Derivative Relative Areas')
plt.figure()
plt.plot(els[:(len(els)-2)],ddRelAreas)
plt.title('Second Derivative Relative Areas')  
"""
