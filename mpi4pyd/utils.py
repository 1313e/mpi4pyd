# -*- coding: utf-8 -*-

"""
Utilities
=========
Provides several useful utility functions.

"""


# %% IMPORTS
# MPI import
from mpi4pyd import MPI

# All declaration
__all__ = ['rprint']

# Determine MPI size and ranks
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


# %% FUNCTION DEFINITIONS
# Redefine the print function to include the MPI rank if MPI is used
def rprint(*args, **kwargs):
    """
    Custom :func:`~print` function that prepends the rank of the MPI process
    that calls it to the message if the size of the world intra-communicator is
    more than 1.
    Takes the same input arguments as the normal :func:`~print` function.

    """

    # If MPI is used and size > 1, prepend rank to message
    if(MPI.__package__ == 'mpi4py' and size > 1):
        args = list(args)
        args.insert(0, "Rank %i:" % (rank))
    print(*args, **kwargs)
