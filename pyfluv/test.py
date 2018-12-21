# junk code for quick tests
import matplotlib.pyplot as plt

import streammath as sm
import streamgeometry as strgeo
"""
depth = 3

method = 'cut'
adjustY = True

if method == 'cut':
    findType = 'overhang'
    pareType = 'max'
elif method == 'fill':
    findType = 'undercut'
    pareType = 'min'
else:
    raise Exception('Invalid method. Method must be "cut" or "fill"')

line1 = (0,depth)

lineX = [0,1,3,2,5,7,10,7.5,10,10,12,15,14,16,14,13.5,12.5,17]
lineY = [1,3,1,4,3,5,5,4,2,-1,.5,0,2,4,3,4,4.5,5]
    
inters = sm.get_intersections(lineX,lineY,line1)

myPlot = plt.plot(lineX,lineY)
plt.scatter(inters[0],inters[1])

merged = sm.insert_points_in_series(lineX,lineY,inters)
plt.plot(merged[0],merged[1])

prepared = sm.prepare_cross_section(lineX,lineY,line1,thw=None) 
plt.plot(prepared[0],prepared[1], linewidth = 3)

print('Area = ' + str(round(sm.get_area(prepared[0],prepared[1]),2)))
print('Mean Depth = ', str(round(sm.get_mean_depth(prepared[0],prepared[1],depth),2)))

ov = sm.get_cuts(lineX,lineY,findType)
ovHangsX = [lineX[i] for i in ov]
ovHangsY = [lineY[i] for i in ov]
plt.scatter(ovHangsX,ovHangsY,s=100)

overhangSeqs = sm.find_contiguous_sequences(ov)
pareHangs = sm.pare_contiguous_sequences(overhangSeqs,lineY,pareType)

topHangsX = [lineX[i] for i in pareHangs]
topHangsY = [lineY[i] for i in pareHangs]
plt.scatter(topHangsX,topHangsY,s=200)

cut = sm.remove_overhangs(lineX,lineY,method,adjustY)
plt.plot(cut[0],cut[1], linewidth = 4)

cent = sm.get_centroid(prepared[0],prepared[1])
plt.scatter(cent[0],cent[1],s=250)

wetLength = sm.wetted_perimeter(prepared[0],prepared[1],lineX,lineY)
print("Hydraulic Radius: " + str(round(wetLength,2)))


sampX = [1,3,2,4,5,5,6,6,5.5,5,2,1]
sampY = [1,4,5,5,2,3,3,2,2,1,1.5,3]
plt.figure()
plt.plot(sampX,sampY)
print('Simple: ' + str(sm.is_simple(sampX,sampY)))


p2 = (3,3)
p1 = (4,1)

lox = (0,p2[0])
loy = (0,p2[1])

lax = (0,p1[0])
lay = (0,p1[1])

newproj = sm.project_point(p1,p2)
lpx = (newproj[0])
lpy = (newproj[1])

plx = (p1[0],newproj[0])
ply = (p1[1],newproj[1])

plt.figure()
plt.plot(lox,loy)
plt.plot(lax,lay)
plt.scatter(lpx,lpy)
plt.plot(plx,ply)
plt.axes().set_aspect('equal', 'datalim')
plt.show()


x1 = [3,4,6,7,11,13,12,14]
y1 = [2,5,6,7,10,11,5,8]

projected = sm.centerline_series(x1,y1)
projX = projected[0]
projY = projected[1]
plt.figure()
plt.plot(x1,y1)
plt.scatter(projX,projY)

for i in range(len(projX)):
    px = (x1[i],projX[i])
    py = (y1[i],projY[i])
    plt.plot(px,py)
"""
exes = [0,1,2,2,4,5,6,7,9,9,10,11,13,14]
whys = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
zees = [10,8.5,6,4,3,2.5,3,4,4,8,11,4.5,5,8]

el = 7
xs = strgeo.CrossSection(exes,whys,zees,thwStation = 10,name='MyXS',bkfEl = el,triggerRecalc = True)
xs.planplot(showProjections=True)
xs.qplot(showBkf=True,showCutSection=True)

"""
plt.figure()
plt.plot(exes,zees)
plt.scatter(exes,zees)
ov = sm.get_cuts(exes,zees,'overhang')
plt.scatter(exes[2],zees[2])

rem = sm.remove_overhangs(exes,zees,'fill',True)
plt.plot(rem[0],rem[1])
"""
