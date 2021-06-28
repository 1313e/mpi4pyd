# -*- coding: utf-8 -*-

"""
MPI
===
Module that automatically picks the correct MPI module based on whether or not
the :mod:`mpi4py.MPI` module is available.

"""


# %% IMPORTS
# MPI import
try:
    from mpi4py import MPI as _MPI
    from mpi4py.MPI import *
except ImportError:                         # pragma: no cover
    from mpi4pyd import dummyMPI as _MPI
    from mpi4pyd.dummyMPI import *
from . import _hybrid_comm
from ._hybrid_comm import *

# All declaration
__all__ = []
if(_MPI.__package__ == 'mpi4py'):
    __all__.extend([prop for prop in dir(_MPI) if not prop.startswith('_')])
else:                                       # pragma: no cover
    __all__.extend(_MPI.__all__)
__all__.extend(_hybrid_comm.__all__)

# Name and package declaration
__name__ = getattr(_MPI, '__name__', None)
__package__ = getattr(_MPI, '__package__', None)
