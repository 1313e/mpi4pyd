# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from types import BuiltinMethodType, MethodType

# Package imports
import numpy as np
import pytest

# mpi4pyd imports
from mpi4pyd import MPI
from mpi4pyd.MPI import (COMM_WORLD as comm, HYBRID_COMM_WORLD as h_comm,
                         get_HybridComm_obj)


# Get size and rank
rank = comm.Get_rank()
size = comm.Get_size()

# Get method types
m_types = (BuiltinMethodType, MethodType)


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for get_HybridComm_obj() function
class Test_get_HybridComm_obj(object):
    # Test if default input arguments work
    def test_default(self):
        assert get_HybridComm_obj() is h_comm

    # Test if providing comm returns h_comm
    def test_comm(self):
        assert get_HybridComm_obj(comm) is h_comm

    # Test if providing h_comm returns itself
    def test_h_comm(self):
        assert get_HybridComm_obj(h_comm) is h_comm

    # Test if providing the wrong object raises an error
    def test_invalid_comm(self):
        with pytest.raises(TypeError):
            get_HybridComm_obj(0)


# Pytest for standard HybridComm obj
class Test_HybridComm_class(object):
    # Create fixture for making dummy NumPy arrays
    @pytest.fixture(scope='function')
    def array(self):
        np.random.seed(comm.Get_rank())
        return(np.random.rand(size, 10))

    # Test if h_comm has the same attrs as comm
    def test_has_attrs(self):
        instance_attrs = dir(comm)
        for attr in instance_attrs:
            assert hasattr(h_comm, attr)

    # Test if all non-overridden attrs in h_comm are the same as in comm
    def test_get_attrs(self):
        skip_attrs = ['info']
        attrs = [attr for attr in dir(comm) if
                 attr not in (*h_comm.overridden_attrs, *skip_attrs)]
        for attr in attrs:
            assert getattr(comm, attr) == getattr(h_comm, attr), attr

    # Test default broadcast
    def test_bcast(self, array):
        assert np.all(comm.bcast(array, 0) == h_comm.bcast(array, 0))

    # Test default gather
    def test_gather(self, array):
        g_array1 = comm.gather(array, 0)
        g_array2 = h_comm.gather(array, 0)
        assert type(g_array1) == type(g_array2)
        if not rank:
            for array1, array2 in zip(g_array1, g_array2):
                assert np.all(array1 == array2)

    # Test default scatter
    def test_scatter(self, array):
        assert np.all(comm.scatter(array, 0) == h_comm.scatter(array, 0))
