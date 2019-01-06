# pyfluv

A python package for the analysis of stream planform, profile and cross sections with a focus on restoration and mitigation.

This project is currently PRE-ALPHA. You *could* download it right now, but it wouldn't be a good use of your time. Come back later!

## Getting Started

Follow these instructions to get a copy of pyfluv on your machine.

### Prerequisites

Pyfluv is pip installable. If you choose not to pip install pyfluv, you'll need to make sure you have **numpy** on your machine,
and you'll probably pip install that. So why not just pip install pyfluv?

Pyfluv has only been tested for **Python 3.7**. There are no plans to test pyfluv with any other versions of Python,
but it *should* work with any **3.X** distribution.

### Installing

#### With pip

If you have pip on your machine, just

```
pip install pyfluv
```

from the terminal. Pip will install pyfluv's dependencies (as of version 0.10, just numpy and matplotlib) for you.

Once pyfluv is installed, it can be imported in Python with

```python
import pyfluv
```

## Getting Started

Because pyfluv is pre-alpha, it has limited functionality. At the time of writing is contains two core classes - CrossSection and GrainDistribution.

### CrossSection

The CrossSection class reads in raw survey data (eastings,northings,elevations). If a bankfull elevation is specified, then it can also calculate statistics such as cross section area and width. If a water slope and manning's N are specified, then flow statistics can be calculated as well. The CrossSection class also includes methods to calculate the floodplain elevation (find_floodplain_elevation()) and two methods to determine the optimal bankfull elevation based on a statistic dependent on the bankfull elevation, such as bankfull area or width (bkf_brute_search() and bkf_binary_search()).

The CrossSection class also includes basic plotting functions (qplot() and planplot())

### GrainDistribution

The GrainDistributionClass reads in a pebble count or sieve data as a dictionary that relates a grainsize to its prevalence. The class claculates statistics such as mean grain size, sorting, skewness and kurtosis. A method is included to calculate the grainsize such that X percent of particles in the sample are smaller than the calculated size (dx()).

The GrainDistributionClass also includes basic plotting functions (cplot(), semilogplot() and bplot()).

## Built With

* [Numpy](http://www.numpy.org/) - For manipulating arrays and linear algebra operations
* [Matplotlib](https://matplotlib.org/) - For quick plots

## Contributing

If you are a practitioner or academic involved in stream restoration or fluvial geomorphology and wish to contribute to or
comment on this project, contact me on [Github](https://github.com/rsjones94).

## Authors

* **Sky Jones** - *Development* - [Github](https://github.com/rsjones94)

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details.
