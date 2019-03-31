"""
Dictionaries of constants
"""
METRIC_CONSTANTS = {'gammaWater':9806.6500286389, # weight density of water, N/m^3
                    'rhoWater':1000, # mass density of water, kg/m^3
                    'manningsNumerator':1, # the numerator in Manning's equation
                    'metersToFeet':3.28084, # conversion factor from meters to feet
                    'feetToMeters':(1/3.28084), # conversion factor from meters to feet
                    'inchesToMil':25.4, # conversion factor from inches to millimeters
                    'milToInches':(1/25.4), # conversion factor from millimeters to inches
                    'g':9.80665, # little g, gravitational acceleration in m/s^2
                    'radToDegrees':57.2958, # radians to degrees
                    'degreesToRad':1/57.2958, # degrees to radians
                    'lengthUnit':'meters',
                    'timeUnit':'seconds',
                    'forceUnit':'newtons',
                    'massUnit':'kilograms',
                    'smallLengthUnit':'mm'
                    }

IMPERIAL_CONSTANTS = {'gammaWater':62.427961, # weight density of water, lbs/ft^3
                      'rhoWater':1.9403203368, # mass density of water, slugs/ft^3
                      'manningsNumerator':1.49, # the numerator in Manning's equation
                      'metersToFeet':3.28084, # conversion factor from meters to feet
                      'feetToMeters':(1/3.28084), # conversion factor from meters to feet
                      'inchesToMil':25.4, # conversion factor from inches to millimeters
                      'milToInches':(1/25.4), # conversion factor from millimeters to inches
                      'g':32.17405, # little g, gravitational acceleration in ft/s^2
                      'radToDegrees':57.2958, # radians to degrees
                      'degreesToRad':1/57.2958, # degrees to radians
                      'lengthUnit':'feet',
                      'timeUnit':'seconds',
                      'forceUnit':'pounds',
                      'massUnit':'slugs',
                      'smallLengthUnit':'inches'
                      }
