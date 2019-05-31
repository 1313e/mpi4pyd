# -*- coding: utf-8 -*-

"""
mpi4pyd
=======
MPI for Python Dummies.

"""


# %% IMPORTS
# mpi4pyd imports
from .__version__ import __version__
from . import dummyMPI
from . import MPI
from .MPI import get_HybridComm_obj
from . import utils
from .utils import *

# All declaration
__all__ = ['dummyMPI', 'MPI', 'utils', 'get_HybridComm_obj']
__all__.extend(utils.__all__)

# Author declaration
__author__ = "Ellert van der Velden (@1313e)"
