"""
Custom exceptions.
"""

class InputError(Exception):
    """
    When improper input is passed to a method.
    """
    pass

class ShapeAgreementError(InputError):
    """
    When the length of two array-like objects do not agree but shape agreement
    is required for functionality. Inherits from InputError.
    """
    pass

class PhysicsLogicError(Exception):
    """
    When when a condition arises that would be physically impossible.
    """
    pass

class GeometryError(PhysicsLogicError):
    """
    When the geometry of a stream object is not simple or is otherwise invalid.
    Inherits from PhysicsLogicError.
    """
    pass

class NullVectorError(Exception):
    """
    Errors stemming from zero length vector operations.
    """
    pass

class NamingError(Exception):
    """
    Errors stemming from malformed stream survey descriptions.
    """
    pass
    
class MissingKeyError(Exception):
    """
    When a dictionary of keys is missing a mandatory key.
    """
    pass