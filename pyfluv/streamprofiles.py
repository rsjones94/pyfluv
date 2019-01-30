"""
Contains the Profile class and helper classes.
"""

class Profile(object):
    """
    A longitudinal stream profile.
    
    Attributes:
        x
    """
    
    def __init__(self):
        """
        Args:
            x
        
        Raises:
            x
        """
        pass
    

class Morph(object):
    """
    A subsection of a longitudinal stream profile representing a distinct substrate morphology.
    
    Attributes:
        x
    """
    
    def __init__(self,df,name = None,morphType = None,metric = False):
        """
        Args:
            df: a dict or pandas dataframe with at least two columns/keys "Station", "Thalweg"
                and additional optional columns/keys. Standardized col/key names include 
                "Water Surface", "Bankfull", "Top of Bank"
        
        Raises:
            x
        """
        pass