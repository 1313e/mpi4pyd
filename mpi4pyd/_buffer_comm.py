# -*- coding: utf-8 -*-

"""
BufferComm
==========

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Package imports
from e13tools import InputError
import numpy as np

# MPI imports
try:
    from mpi4py import MPI
except ImportError:
    from mpi4pyd import dummyMPI as MPI
finally:
    from mpi4pyd import dummyMPI

# All declaration
__all__ = ['BUFFER_COMM_WORLD', 'get_BufferComm_obj']


# Initialize buffer_comm_registry
buffer_comm_registry = {}


# %% FUNCTION DEFINITIONS
# Function factory that returns special BufferComm class instances
def get_BufferComm_obj(comm=None):
    """
    Function factory that returns an instance of the :class:`~BufferComm`
    class, defined as ``BufferComm(comm.__class__, object)``.

    This :class:`~BufferComm` class wraps the provided :obj:`MPI.Intracomm`
    instance `comm` and overrides all of its lowercase communication methods
    (e.g., :meth:`~MPI.Intracomm.bcast`, :meth:`~MPI.Intracomm.gather`,
    :meth:`~MPI.Intracomm.scatter`, :meth:`~MPI.Intracomm.recv` and
    :meth:`~MPI.Intracomm.send`) with improved versions. These improved
    communication methods automatically select the most optimal way of
    communicating their input arguments.

    Besides the new method functionalities, the returned instance behaves in
    the exact same way as the provided `comm` and can easily be used in any
    algorithm that expects an instance of the :class:`MPI.Intracomm` class.

    Optional
    --------
    comm : :obj:`~MPI.Intracomm` object or None. Default: None
        The MPI intra-communicator to use as the base for the
        :obj:`~BufferComm` instance.
        If *None*, use :obj:`MPI.COMM_WORLD` instead.

    Returns
    -------
    buffer_comm : :obj:`MPI.Comm` object
        The provided `comm` which has its lowercase communication methods
        overridden. If `comm` is *None* or :obj:`MPI.COMM_WORLD`,
        :obj:`mpi4pyd.MPI.BUFFER_COMM_WORLD` is returned instead.

    """

    # If comm is None, set it to MPI.COMM_WORLD
    if comm is None:
        comm = MPI.COMM_WORLD
    # Else, check if provided comm is an MPI intra-communicator
    elif not isinstance(comm, (MPI.Intracomm, dummyMPI.Intracomm)):
        raise TypeError("Input argument 'comm' must be an instance of "
                        "the MPI.Intracomm class!")

    # Check if provided comm already has a BufferComm instance
    if comm.name in buffer_comm_registry.keys():
        # If so, return that BufferComm instance instead
        return(buffer_comm_registry[comm.name])

    # Define list of attributes that are overridden
    # TODO: Find a more dynamic way of determining this
    override_props = ['_rank', '_size', '__init__', '__getattribute__',
                      '__dir__', 'bcast', 'gather']

    # Define BufferComm class
    class BufferComm(comm.__class__, object):
        """
        Custom :class:`~MPI.Intracomm` class.

        """

        def __init__(self):
            # Bind provided communicator
            self._rank = comm.Get_rank()
            self._size = comm.Get_size()

        # Override getattr property to use self._comm attributes if necessary
        def __getattribute__(self, name):
            if name not in override_props:
                return(getattr(comm, name))
            else:
                return(super().__getattribute__(name))

        # Override __dir__ attribute to use the one from self._comm
        def __dir__(self):
            return(dir(comm))

        # Specialized bcast function that automatically makes use of buffers
        def bcast(self, obj, root):
            """
            Special broadcast method that automatically uses the appropriate
            method (:meth:`~MPI.Intracomm.bcast` or
            :meth:`~MPI.Intracomm.Bcast`) depending on the type of the provided
            `obj`.

            Parameters
            ----------
            obj : :obj:`~numpy.ndarray` or object
                The object to broadcast to all MPI ranks.
                If :obj:`~numpy.ndarray`, use :meth:`~MPI.Intracomm.Bcast`.
                If not, use :meth:`~MPI.Intracomm.bcast` instead.
            root : int
                The MPI rank that broadcasts `obj`.

            Returns
            -------
            obj : object
                The broadcasted `obj`.

            """

            # Sender
            if(self._rank == root):
                # Check if provided object is a NumPy array
                if isinstance(obj, np.ndarray):
                    # If so, send shape and dtype of the NumPy array
                    comm.bcast(['NumPy ndarray', [obj.shape, obj.dtype]],
                               root=root)

                    # Then send the NumPy array as a buffer object
                    comm.Bcast(obj, root=root)

                # If not, send obj the normal way
                else:
                    # Try to send object
                    try:
                        comm.bcast([obj.__class__.__name__, obj], root=root)
                    # If this fails, raise error about byte size
                    except OverflowError:
                        raise InputError("Input argument `obj` has a byte size"
                                         " that cannot be stored in a 32-bit "
                                         "int (%i > %i)!"
                                         % (obj.__sizeof__(), 2**31-1))

            # Receivers wait for instructions
            else:
                # Receive object
                obj_type, obj = comm.bcast(obj, root=root)

                # If obj_type is NumPy ndarray, obj contains shape and dtype
                if(obj_type == 'NumPy ndarray'):
                    # Create empty NumPy array with given shape and dtype
                    obj = np.empty(*obj)

                    # Receive NumPy array
                    comm.Bcast(obj, root=root)

            # Return obj
            return(obj)

        # Specialized gather function that automatically makes use of buffers
        def gather(self, obj, root):
            """
            Special gather method that automatically uses the appropriate
            method (:meth:`~MPI.Intracomm.gather` or
            :meth:`~MPI.Intracomm.Gatherv`) depending on the type of the
            provided `obj`.

            Parameters
            ----------
            obj : :obj:`~numpy.ndarray` or object
                The object to gather from all MPI ranks.
                If :obj:`~numpy.ndarray`, use :meth:`~MPI.Intracomm.Gatherv`.
                If not, use :meth:`~MPI.Intracomm.gather` instead.
            root : int
                The MPI rank that gathers `obj`.

            Returns
            -------
            obj : list or None
                If MPI rank is `root`, returns a list of the gathered objects.
                Else, returns *None*.

            Warnings
            --------
            If some but not all MPI ranks use a NumPy array, this method will
            hang indefinitely.
            When gathering NumPy arrays, all arrays must have the same number
            of dimensions and the same shape, except for one axis.

            """

            # Check if provided object is a NumPy array
            if isinstance(obj, np.ndarray):
                # If so, gather the shapes of obj on the receiver
                shapes = comm.gather(obj.shape, root=root)

                # If obj has an empty dimension anywhere, replace it with dummy
                if not np.all(obj.shape):
                    obj = np.empty([1]*obj.ndim, dtype=obj.dtype)

                # Receiver sets up a buffer array and receives NumPy array
                if(self._rank == root):
                    # Obtain the required shape of the buffer array
                    buff_shape = (self._size, np.product(np.max(shapes,
                                                                axis=0)))

                    # Create buffer array
                    buff = np.empty(buff_shape, dtype=obj.dtype)

                    # Gather all NumPy arrays
                    comm.Gatherv(obj.ravel(), buff, root=root)

                    # Make an empty list holding individual arrays
                    arr_list = []

                    # Loop over buff and transform back to single arrays
                    for array, shape in zip(buff, shapes):
                        array_temp = array[:np.product(shape)]
                        arr_list.append(array_temp.reshape(shape))

                    # Replace buff by arr_list
                    buff = arr_list

                # Senders send the array
                else:
                    # Senders set up dummy buffer
                    buff = None

                    # Send array
                    comm.Gatherv(obj.ravel(), buff, root=root)

            # If not, gather obj the normal way
            else:
                # Try to send the obj
                try:
                    buff = comm.gather(obj, root=root)
                # If this fails, raise error about byte size
                except SystemError:
                    raise InputError("Input argument 'obj' is too large!")

            # Return buff
            return(buff)

    # Initialize BufferComm
    buffer_comm = BufferComm()

    # Register initialized BufferComm
    buffer_comm_registry[comm.name] = buffer_comm

    # Return buffer_comm
    return(buffer_comm)


# %% DEFAULT INSTANCES
BUFFER_COMM_WORLD = get_BufferComm_obj()
