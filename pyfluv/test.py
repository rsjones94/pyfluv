# junk code for quick tests
import matplotlib.pyplot as plt

import streammath as sm
import streamgeometry as strgeo


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
