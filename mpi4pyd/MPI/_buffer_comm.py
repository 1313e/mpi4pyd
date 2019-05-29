# -*- coding: utf-8 -*-

"""
BufferComm
==========

"""


# %% IMPORTS
# Built-in imports
from inspect import currentframe

# Package imports
from e13tools import InputError, ShapeError
import numpy as np

# mpi4pyd imports
from mpi4pyd import dummyMPI, MPI
from mpi4pyd.MPI._helpers import is_buffer_obj

# All declaration
__all__ = ['BUFFER_COMM_SELF', 'BUFFER_COMM_WORLD', 'get_BufferComm_obj']


# Initialize buffer_comm_registry
buffer_comm_registry = {}

# Make conversion dict from NumPy dtype to MPI Datatype
dtype_dict = {
    'int': MPI.LONG,
    'int32': MPI.INT,
    'int64': MPI.LONG,
    'float': MPI.DOUBLE,
    'float32': MPI.FLOAT,
    'float64': MPI.DOUBLE}


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

    Note
    ----
    Providing the same :obj:`~MPI.Intracomm` instance to this function twice,
    will not create two :obj:`~BufferComm` objects. Instead, the instance
    created the first time will be returned each consecutive time. All created
    :obj:`~BufferComm` objects are stored in the :obj:`~buffer_comm_registry`.

    """

    # If comm is None, set it to MPI.COMM_WORLD
    if comm is None:
        comm = MPI.COMM_WORLD
    # Else, check if provided comm is an MPI intra-communicator
    elif not isinstance(comm, (MPI.Intracomm, dummyMPI.Intracomm)):
        raise TypeError("Input argument 'comm' must be an instance of "
                        "the MPI.Intracomm class!")

    # Check if provided comm already has a BufferComm instance
    if hex(id(comm)) in buffer_comm_registry.keys():
        # If so, return that BufferComm instance instead
        return(buffer_comm_registry[hex(id(comm))])

    # Check if provided comm is not already a BufferComm instance
    if comm in buffer_comm_registry.values():
        # If so, return provided BufferComm instance instead
        return(comm)

    # Make tuple of overridden attributes
    overridden_attrs = ('__init__', 'bcast', 'gather', 'scatter')

    # %% BUFFERCOMM CLASS DEFINITION
    class BufferComm(comm.__class__, object):
        """
        Custom :class:`~MPI.Intracomm` class.

        """

        def __init__(self):
            # Bind provided communicator
            if not hasattr(self, '_rank'):
                self._rank = comm.Get_rank()
            if not hasattr(self, '_size'):
                self._size = comm.Get_size()

        # If requested attribute is not a method, use comm for getattr
        def __getattribute__(self, name):
            if name in dir(comm) and name not in overridden_attrs:
                return(getattr(comm, name))
            else:
                return(super().__getattribute__(name))

        # If requested attribute is not a method, use comm for setattr
        def __setattr__(self, name, value):
            if name in dir(comm) and name not in overridden_attrs:
                setattr(comm, name, value)
            else:
                super().__setattr__(name, value)

        # If requested attribute is not a method, use comm for delattr
        def __delattr__(self, name):
            if name in dir(comm) and name not in overridden_attrs:
                delattr(comm, name)
            else:
                super().__delattr__(name)

        # %% COMMUNICATION METHODS
        # Specialized bcast function that automatically makes use of buffers
        def bcast(self, obj, root=0):
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

            Optional
            --------
            root : int. Default: 0
                The MPI rank that broadcasts `obj`.

            Returns
            -------
            obj : object
                The broadcasted `obj`.

            """

            # Check if obj can be broadcasted as a buffer object
            use_buffer = self.__use_buffer_meth(obj, root)

            # If provided object uses a buffer
            if use_buffer:
                # Sender
                if(self._rank == root):
                    # If so, send shape and dtype of the NumPy array
                    comm.bcast([obj.shape, obj.dtype], root=root)

                    # Then send the NumPy array as a buffer object
                    comm.Bcast(obj, root=root)

                # Receivers receive NumPy array
                else:
                    # Create empty NumPy array with given shape and dtype
                    obj = np.empty(*comm.bcast(None, root=root))

                    # Receive NumPy array
                    comm.Bcast(obj, root=root)

            # If not, broadcast obj the normal way
            else:
                # Try to broadcast object
                try:
                    obj = comm.bcast(obj, root=root)
                # If this fails, raise error about byte size
                except OverflowError:
                    raise InputError("Input argument `obj` has a byte size "
                                     "that cannot be stored in a 32-bit int "
                                     "(%i > %i)!"
                                     % (obj.__sizeof__(), 2**31-1))

            # Return obj
            return(obj)

        # Specialized gather function that automatically makes use of buffers
        def gather(self, obj, root=0):
            """
            Special gather method that automatically uses the appropriate
            method (:meth:`~MPI.Intracomm.gather` or
            :meth:`~MPI.Intracomm.Gatherv`) depending on the lay-out of the
            provided `obj`.

            Parameters
            ----------
            obj : :obj:`~numpy.ndarray` or object
                The object to gather from all MPI ranks.
                If :obj:`~numpy.ndarray`, use :meth:`~MPI.Intracomm.Gatherv`.
                If not, use :meth:`~MPI.Intracomm.gather` instead.

            Optional
            --------
            root : int. Default: 0
                The MPI rank that gathers `obj`.

            Returns
            -------
            obj : list, object or None
                If MPI rank is `root`, returns a list of gathered objects or a
                single object, depending on `default`.
                Else, returns *None*.

            Warnings
            --------
            When gathering NumPy arrays, all arrays must have the same number
            of dimensions and the same shape, except for one axis.

            """

            # Check if obj can be gathered as a buffer object
            use_buffer = self.__use_buffer_meth(obj, root)

            # If all provided objects use buffers
            if use_buffer:
                # If so, gather the shapes of obj on the receiver
                shapes = np.array(comm.gather(obj.shape, root=root))

                # Receiver sets up a buffer array and receives NumPy array
                if(self._rank == root):
                    # Obtain counts and displacements
                    counts = np.product(shapes, axis=1)
                    disps = np.cumsum([0, *counts[:-1]])

                    # Get the maximum size in every axis
                    max_size = np.max(shapes, axis=0)

                    # Check which axis sizes differ
                    diff_size = ~np.all(np.equal(shapes, max_size), axis=0)

                    # If more than a single axis size differs, raise error
                    # TODO: Remove this limitation
                    if(sum(diff_size) > 1):
                        raise ShapeError("Input argument 'obj' differs in size"
                                         "in more than 1 axis!")

                    # Get the axis that differs and its cumulative size
                    diff_axis =\
                        np.nonzero(diff_size)[0][0] if sum(diff_size) else 0

                    # Get the buffer shape and size
                    buff_shape = max_size
                    buff_shape[diff_axis] = np.sum(shapes[:, diff_axis])
                    buff_size = np.product(buff_shape)

                    # Initialize empty buffer array
                    recv_obj = np.empty(buff_size, dtype=obj.dtype)

                    # Make buffer list
                    buff =\
                        [recv_obj, counts, disps, dtype_dict[obj.dtype.name]]

                    # Gather all NumPy arrays
                    comm.Gatherv([obj.ravel(), obj.size], buff, root=root)

                    # Reconstruct the original shapes of NumPy arrays
                    arr_list = np.split(recv_obj, disps[1:])
                    for i, shape in enumerate(shapes):
                        arr_list[i] = np.reshape(arr_list[i], shape)
                    recv_obj = arr_list

                # Senders send the array
                else:
                    # Send NumPy array
                    comm.Gatherv([obj, obj.size], None, root=root)
                    recv_obj = None

            # If not, gather obj the normal way
            else:
                # Try to gather the obj
                try:
                    recv_obj = comm.gather(obj, root=root)
                # If this fails, raise error about byte size
                except SystemError:
                    raise InputError("Input argument 'obj' is too large!")

            # Return recv_obj
            return(recv_obj)

        # Specialized scatter function that automatically makes use of buffers
        def scatter(self, obj, root=0):
            """
            Special scatter method that automatically uses the appropriate
            method (:meth:`~MPI.Intracomm.scatter` or
            :meth:`~MPI.Intracomm.Scatter`) depending on the type of the
            provided `obj`.

            Unlike :meth:`~MPI.Intracomm.scatter`, providing a buffer object
            with more than :attr:`~_size` items will not raise an error, but
            evenly distribute all the items instead.

            Parameters
            ----------
            obj : :obj:`~numpy.ndarray` or object
                The object to scatter to all MPI ranks.
                If :obj:`~numpy.ndarray`, use :meth:`~MPI.Intracomm.Scatterv`.
                If not, use :meth:`~MPI.Intracomm.scatter` instead.

            Optional
            --------
            root : int. Default: 0
                The MPI rank that scatters `obj`.

            Returns
            -------
            obj : :obj:`~numpy.ndarray` or object
                The object that has been scattered to this MPI rank.

            """

            # Check if obj can be scattered as buffer objects
            use_buffer = self.__use_buffer_meth(obj, root)

            # If provided object uses a buffer
            if use_buffer:
                # Sender prepares for scattering
                if(self._rank == root):
                    # Raise error if length of axis is not divisible by size
                    if len(obj) % self._size:
                        raise ShapeError("Input argument 'obj' cannot be "
                                         "divided evenly over the available "
                                         "number of MPI ranks!")

                    # Determine shape of scattered object
                    buff_shape = list(obj.shape)
                    buff_shape[0] //= self._size

                    # Initialize empty buffer array
                    recv_obj = np.empty(*comm.bcast([buff_shape, obj.dtype],
                                                    root=root))

                    # Scatter NumPy array
                    comm.Scatter(obj, recv_obj, root=root)

                # Receivers receive the array
                else:
                    # Initialize empty buffer array
                    recv_obj = np.empty(*comm.bcast(None, root=root))

                    # Receive scattered NumPy array
                    comm.Scatter(None, recv_obj, root=root)

                # Remove single dimensional entries from recv_obj
                recv_obj = recv_obj.squeeze()

            # If not, scatter obj the normal way
            else:
                # Try to scatter the obj
                try:
                    recv_obj = comm.scatter(obj, root=root)
                # If this fails, raise error about byte size
                except SystemError:
                    raise InputError("Input argument 'obj' is too large!")

            # Return recv_obj
            return(recv_obj)

        # %% UTILITY METHODS
        # This function checks if a buffer communication method can be used
        def __use_buffer_meth(self, obj, root):
            """
            Depending on which communication method calls this function,
            determines if the provided `obj` on all MPI ranks can be
            communicated using an uppercase communication method.

            This method must be called by all MPI ranks.
            This method must never be called directly.

            """

            # Determine the name of the frame calling this method
            meth_name = currentframe().f_back.f_code.co_name

            # Check who called this method and act accordingly
            if meth_name in ('bcast', 'scatter'):
                return(comm.bcast(is_buffer_obj(obj), root=root))
            elif meth_name in ('gather'):
                return(comm.allreduce(is_buffer_obj(obj), op=MPI.MIN))
            else:
                raise NotImplementedError

    # %% REMAINDER OF FUNCTION FACTORY
    # Initialize BufferComm
    buffer_comm = BufferComm()

    # Register initialized BufferComm
    buffer_comm_registry[hex(id(comm))] = buffer_comm

    # Return buffer_comm
    return(buffer_comm)


# %% DEFAULT INSTANCES
BUFFER_COMM_SELF = get_BufferComm_obj(MPI.COMM_SELF)
BUFFER_COMM_WORLD = get_BufferComm_obj(MPI.COMM_WORLD)
