# -*- coding: utf-8 -*-

"""
Helpers
=======

"""

# %% IMPORTS
# Package imports
import numpy as np

# All declaration
__all__ = ['is_buffer_obj']


# %% FUNCTION DEFINITIONS
# This function checks whether a provided object exposes its internal buffer
def is_buffer_obj(obj):
    """
    Checks if the provided `obj` exposes its internal buffer and can be used in
    uppercase communication methods.
    Currently, only NumPy arrays are seen as buffer objects.

    """

    # Check if provided obj is a NumPy array
    return(isinstance(obj, np.ndarray))
