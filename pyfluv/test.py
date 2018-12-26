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

work = grain.GrainDistribution(distr,name ='Work',triggerRecalc=True)