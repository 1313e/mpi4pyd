# -*- coding: utf-8 -*-

# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# mpi4pyd imports
from mpi4pyd.utils import rprint


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for the rprint function
def test_rprint():
    # Check if rprint works correctly
    rprint('Testing')
