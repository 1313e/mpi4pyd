# -*- coding: utf-8 -*-

"""
mpi4pyd
=======
MPI for Python Dummies.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# mpi4pyd imports
from .__version__ import __version__
from ._buffer_comm import get_BufferComm_obj
from . import dummyMPI
from . import MPI
from . import utils
from .utils import *

# All declaration
__all__ = ['dummyMPI', 'MPI', 'utils', 'get_BufferComm_obj']
__all__.extend(utils.__all__)

# Author declaration
__author__ = "Ellert van der Velden (@1313e)"
