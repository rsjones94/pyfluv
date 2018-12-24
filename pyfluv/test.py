# junk code for quick tests
import matplotlib.pyplot as plt

import streammath as sm
import streamgeometry as strgeo
import graindistributions as grain


distr = {2:20,
         4:30,
         7:10,
         5:3
        }

work = grain.GrainDistribution(distr,name ='Work',triggerRecalc=True)