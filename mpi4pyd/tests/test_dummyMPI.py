# -*- coding: utf-8 -*-

# %% IMPORTS
# Package imports
import numpy as np
import pytest

# mpi4pyd imports
from mpi4pyd import MPI
from mpi4pyd.dummyMPI import (Comm, Intracomm, COMM_WORLD as comm, get_vendor,
                              SUM)


# Skip entire module if MPI is used
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="Cannot be pytested in MPI")


# %% CUSTOM CLASSES
class CommTest(Comm):
    pass


class CommTest2(Comm):
    def __init__(self):
        self._name = 'CommTest2'
        super().__init__()


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for custom Comm class
def test_CommTest():
    test_comm = CommTest()
    assert test_comm.name == 'dummyMPI_CommTest'


# Pytest for other custom Comm class
def test_CommTest2():
    test_comm = CommTest2()
    assert test_comm.name == 'dummyMPI_CommTest2'


# Pytest for custom Intracomm class
def test_Intracomm():
    with pytest.raises(TypeError):
        Intracomm(1)


# Pytest for COMM_WORLD instance
class Test_COMM_WORLD(object):
    array = np.array([1, 2, 3, 4, 5])
    buffer = np.empty_like(array)

    def test_props(self):
        assert comm.Get_name() == 'dummyMPI_COMM_WORLD'
        assert comm.Get_size() == 1
        assert comm.Get_rank() == 0

    def test_Allgather(self):
        comm.Allgather(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.allgather(self.array)[0] == self.array).all()
        comm.Allgatherv(self.array, self.buffer)
        assert (self.buffer == self.array).all()

    def test_Allreduce(self):
        comm.Allreduce(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.allreduce(self.array) == self.array).all()
        assert (comm.allreduce(1) == 1)
        assert comm.reduce([tuple(self.array.tolist())]) ==\
            tuple(self.array.tolist())

    def test_Barrier(self):
        comm.Barrier()
        comm.barrier()

    def test_Bcast(self):
        comm.Bcast(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.bcast(self.array) == self.array).all()

    def test_Gather(self):
        comm.Gather(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.gather(self.array)[0] == self.array).all()
        comm.Gatherv(self.array, self.buffer)
        assert (self.buffer == self.array).all()

    def test_Is_intra(self):
        assert comm.Is_intra()

    def test_Is_inter(self):
        assert not comm.Is_inter()

    def test_Reduce(self):
        comm.Reduce(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.reduce(self.array) == self.array).all()
        assert (comm.reduce(1) == 1)

    def test_Scatter(self):
        comm.Scatter([self.array], self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.scatter([self.array]) == self.array).all()
        comm.Scatterv([self.array], self.buffer)
        assert (self.buffer == self.array).all()

    def test_Sendrecv(self):
        assert (comm.Sendrecv(self.array) == self.array).all()
        assert (comm.sendrecv(self.array) == self.array).all()


# Pytest for get_vendor() function
def test_get_vendor():
    assert get_vendor()[0] == "dummyMPI"


# Pytest for SUM operator
def test_SUM():
    assert SUM() is None
