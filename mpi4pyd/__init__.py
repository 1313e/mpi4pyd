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
from .__version__ import version as __version__
from . import dummyMPI
from . import MPI
from . import utils
from .utils import *

# All declaration
__all__ = ['dummyMPI', 'MPI', 'utils']
__all__.extend(utils.__all__)

# Author declaration (optional)
__author__ = "Ellert van der Velden (@1313e)"
