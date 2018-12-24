"""
dictionaries of constants
"""

METRIC_CONSTANTS = {'gammaWater':9806.6500286389, # weight density of water, N/m^3
                    'rhoWater':1000, # mass density of water, kg/m^3
                    'manningsNumerator':1, # the numerator in Manning's equation
                    'metersToFeet':3.28084, # conversion factor from meters to feet
                    'feetToMeters':(1/3.28084), # conversion factor from meters to feet
                    'inchesToMil':25.4, # conversion factor from inches to millimeters
                    'milToInches':(1/25.4) # conversion factor from millimeters to inches
                    }

IMPERIAL_CONSTANTS = {'gammaWater':62.427961, # weight density of water, lbs/ft^3
                      'rhoWater':1.9403203368, # mass density of water, slugs/ft^3
                      'manningsNumerator':1.49, # the numerator in Manning's equation
                      'metersToFeet':3.28084, # conversion factor from meters to feet
                      'feetToMeters':(1/3.28084), # conversion factor from meters to feet
                      'inchesToMil':25.4, # conversion factor from inches to millimeters
                      'milToInches':(1/25.4) # conversion factor from millimeters to inches
                      }