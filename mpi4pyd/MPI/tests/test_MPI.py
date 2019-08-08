# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from types import BuiltinMethodType, MethodType

# Package imports
import numpy as np
import pytest

# mpi4pyd imports
from mpi4pyd import MPI
from mpi4pyd.dummyMPI import COMM_WORLD as d_comm
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

    # Test if providing d_comm returns itself
    def test_d_comm(self):
        assert get_HybridComm_obj(d_comm) is d_comm

    # Test if providing a comm with size 1 returns d_comm
    @pytest.mark.skipif(size == 1, reason="Cannot be pytested in serial")
    def test_comm_size_unity(self):
        s_comm = comm.Split(comm.Get_rank(), 0)
        assert get_HybridComm_obj(s_comm) is d_comm
        s_comm.Free()

    # Test if providing the wrong object raises an error
    def test_invalid_comm(self):
        with pytest.raises(TypeError):
            get_HybridComm_obj(0)


# Pytest for standard HybridComm obj
@pytest.mark.skipif(size == 1, reason="Pointless to pytest in serial")
class Test_HybridComm_class(object):
    # Create fixture for making dummy NumPy arrays
    @pytest.fixture(scope='function')
    def array(self):
        np.random.seed(comm.Get_rank())
        return(np.random.rand(size, 10))

    # Create fixture for making dummy lists
    @pytest.fixture(scope='function')
    def lst(self, array):
        return(array.tolist())

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

    # Test the attribute setters
    def test_set_attrs(self):
        # Test if setting a comm attribute raises an error
        with pytest.raises(AttributeError):
            h_comm.rank = 1

        # Test if a new attribute can be created and read
        h_comm.pytest_attr = 'test'
        assert h_comm.pytest_attr == 'test'

        # Test if this attribute is not in comm
        assert not hasattr(comm, 'pytest_attr')

    # Test the attribute deleters
    def test_del_attrs(self):
        # Test if deleting a comm attribute raises an error
        with pytest.raises(AttributeError):
            del h_comm.rank

        # Test if deleting a new attribute can be done
        del h_comm.pytest_attr
        assert not hasattr(h_comm, 'pytest_attr')

    # Test default broadcast with an array
    def test_bcast_array(self, array):
        assert np.allclose(comm.bcast(array, 0), h_comm.bcast(array, 0))

    # Test default broadcast with a list
    def test_bcast_list(self, lst):
        assert np.allclose(comm.bcast(lst, 0), h_comm.bcast(lst, 0))

    # Test default gather with an array
    def test_gather_array(self, array):
        g_array1 = comm.gather(array, 0)
        g_array2 = h_comm.gather(array, 0)
        assert type(g_array1) == type(g_array2)
        if not rank:
            for array1, array2 in zip(g_array1, g_array2):
                assert np.allclose(array1, array2)

    # Test default gather with a list
    def test_gather_list(self, lst):
        g_lst1 = comm.gather(lst, 0)
        g_lst2 = h_comm.gather(lst, 0)
        assert type(g_lst1) == type(g_lst2)
        if not rank:
            for lst1, lst2 in zip(g_lst1, g_lst2):
                assert np.allclose(lst1, lst2)

    # Test default scatter with an array
    def test_scatter_array(self, array):
        assert np.allclose(comm.scatter(array, 0), h_comm.scatter(array, 0))

    # Test default scatter with a list
    def test_scatter_list(self, lst):
        assert np.allclose(comm.scatter(list(lst), 0),
                           h_comm.scatter(list(lst), 0))
