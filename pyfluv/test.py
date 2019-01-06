# junk code for quick tests
import matplotlib.pyplot as plt
import numpy as np

import streammath as sm
import streamgeometry as strgeo
import graindistributions as grain

import westpineyriver as wpr


wpr.ccXSRiffle.sizeDist.semilogplot()
wpr.ccXSRiffle.sizeDist.cplot(normalize=False)
wpr.ccXSRiffle.sizeDist.bplot()