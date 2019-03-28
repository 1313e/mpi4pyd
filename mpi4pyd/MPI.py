# -*- coding: utf-8 -*-

"""
MPI
===
Module that automatically picks the correct MPI module based on whether or not
the :mod:`mpi4py.MPI` module is available.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# MPI import
try:
    from mpi4py import MPI as _MPI
    from mpi4py.MPI import *
except ImportError:
    from mpi4pyd import dummyMPI as _MPI
    from mpi4pyd.dummyMPI import *

# All declaration
__all__ = []
if(_MPI.__name__ == 'mpi4py.MPI'):
    __all__.extend([prop for prop in dir(_MPI) if not prop.startswith('_')])
else:
    __all__.extend(_MPI.__all__)
