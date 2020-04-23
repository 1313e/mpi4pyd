# -*- coding: utf-8 -*-

"""
HybridComm
==========

"""


# %% IMPORTS
# Built-in imports
from inspect import currentframe

# Package imports
import e13tools as e13
import numpy as np

# mpi4pyd imports
from mpi4pyd import dummyMPI, MPI
from mpi4pyd.MPI._helpers import is_buffer_obj

# All declaration
__all__ = ['HYBRID_COMM_SELF', 'HYBRID_COMM_WORLD', 'get_HybridComm_obj']


# Initialize hybrid_comm_registry
hybrid_comm_registry = {}

# Make conversion dict from NumPy dtype to MPI Datatype
dtype_dict = {
    'int': MPI.LONG,
    'int32': MPI.INT,
    'int64': MPI.LONG,
    'float': MPI.DOUBLE,
    'float32': MPI.FLOAT,
    'float64': MPI.DOUBLE}


# %% FUNCTION DEFINITIONS
# Function factory that returns special HybridComm class instances
def get_HybridComm_obj(comm=None):
    """
    Function factory that returns an instance of the :class:`~HybridComm`
    class, defined as ``HybridComm(comm.__class__, object)``.

    This :class:`~HybridComm` class wraps the provided :obj:`MPI.Intracomm`
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
        :obj:`~HybridComm` instance.
        If *None*, use :obj:`MPI.COMM_WORLD` instead.

    Returns
    -------
    hybrid_comm : :obj:`MPI.Comm` object
        The provided `comm` which has its lowercase communication methods
        overridden. If `comm` is *None* or :obj:`MPI.COMM_WORLD`,
        :obj:`mpi4pyd.MPI.HYBRID_COMM_WORLD` is returned instead.

    Note
    ----
    Providing the same :obj:`~MPI.Intracomm` instance to this function twice,
    will not create two :obj:`~HybridComm` objects. Instead, the instance
    created the first time will be returned each consecutive time. All created
    :obj:`~HybridComm` objects are stored in the :obj:`~hybrid_comm_registry`.

    If `comm` has a pool size of `1` (`comm.Get_size == 1`), this function will
    return :obj:`mpi4pyd.dummyMPI.COMM_WORLD` instead. This is because the
    dummy MPI intra-communicator is much more efficient than an associated real
    MPI intra-communicator (as the former uses no communications at all).

    """

    # If comm is None, set it to MPI.COMM_WORLD
    if comm is None:
        comm = MPI.COMM_WORLD
    # Else, check if provided comm is an MPI intra-communicator
    elif not isinstance(comm, (MPI.Intracomm, dummyMPI.Intracomm)):
        raise TypeError("Input argument 'comm' must be an instance of "
                        "the MPI.Intracomm class!")

    # Check if provided comm has a size of 1
    if(comm.Get_size() == 1):
        # If so, return dummyMPI.COMM_WORLD instead
        return(dummyMPI.COMM_WORLD)

    # Check if provided comm already has a HybridComm instance
    if hex(id(comm)) in hybrid_comm_registry.keys():
        # If so, return that HybridComm instance instead
        return(hybrid_comm_registry[hex(id(comm))])

    # Check if provided comm is not already a HybridComm instance
    if comm in hybrid_comm_registry.values():
        # If so, return provided HybridComm instance instead
        return(comm)

    # Make tuple of overridden attributes
    overridden_attrs = ('__init__', 'bcast', 'gather', 'recv', 'scatter',
                        'send')

    # %% HYBRIDCOMM CLASS DEFINITION
    class HybridComm(comm.__class__, object):
        """
        Custom :class:`~MPI.Intracomm` class.

        """

        def __init__(self):
            # Bind provided communicator
            if not hasattr(comm, '_rank'):
                self._rank = comm.Get_rank()
            if not hasattr(comm, '_size'):
                self._size = comm.Get_size()

        # If requested attribute is not a method, use comm for getattr
        def __getattribute__(self, name):
            if name not in overridden_attrs and name in comm.__dir__():
                return(getattr(comm, name))
            else:
                return(super().__getattribute__(name))

        # If requested attribute is not a method, use comm for setattr
        def __setattr__(self, name, value):
            if name not in overridden_attrs and name in comm.__dir__():
                setattr(comm, name, value)
            else:
                super().__setattr__(name, value)

        # If requested attribute is not a method, use comm for delattr
        def __delattr__(self, name):
            if name not in overridden_attrs and name in comm.__dir__():
                delattr(comm, name)
            else:
                super().__delattr__(name)

        # %% CLASS PROPERTIES
        @property
        def overridden_attrs(self):
            """
            list of str: List with all attribute names that have been
            overridden by this :obj:`~HybridComm` instance.

            """

            return(overridden_attrs)

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
            use_buffer = use_buffer_meth(obj, root)

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
                obj = comm.bcast(obj, root=root)

            # Return obj
            return(obj)

        # Specialized gather function that automatically makes use of buffers
        def gather(self, sendobj, root=0):
            """
            Special gather method that automatically uses the appropriate
            method (:meth:`~MPI.Intracomm.gather` or
            :meth:`~MPI.Intracomm.Gatherv`) depending on the lay-out of the
            provided `sendobj`.

            Parameters
            ----------
            sendobj : :obj:`~numpy.ndarray` or object
                The object to gather from all MPI ranks.
                If :obj:`~numpy.ndarray`, use :meth:`~MPI.Intracomm.Gatherv`.
                If not, use :meth:`~MPI.Intracomm.gather` instead.

            Optional
            --------
            root : int. Default: 0
                The MPI rank that gathers `sendobj`.

            Returns
            -------
            recvobj : list or None
                If MPI rank is `root`, returns a list of gathered objects.
                Else, returns *None*.

            """

            # Check if obj can be gathered as a buffer object
            use_buffer = use_buffer_meth(sendobj, root)

            # If all provided objects use buffers
            if use_buffer:
                # If so, gather the shapes of obj on the receiver
                shapes = np.array(comm.gather(sendobj.shape, root=root))

                # Set the key to use for this communication
                key = 147418621

                # Receiver sets up a buffer array and receives NumPy array
                if(self._rank == root):
                    # Initialize empty list of gathered objects
                    arr_list = [np.empty(shape, dtype=sendobj.dtype)
                                for shape in shapes]

                    # Gather all NumPy arrays from all ranks
                    for rank, arr in enumerate(arr_list):
                        # If this is the receivers rank, simply copy the data
                        if(rank == root):
                            arr[:] = sendobj
                        # Else, receive the object normally
                        else:
                            comm.Recv(arr, source=rank, tag=key+rank)

                    # Save arr_list as recvobj
                    recvobj = arr_list

                # Senders send the array
                else:
                    # Send NumPy array
                    comm.Send(sendobj, dest=root, tag=key+self._rank)
                    recvobj = None

                # MPI Barrier
                comm.Barrier()

            # If not, gather obj the normal way
            else:
                recvobj = comm.gather(sendobj, root=root)

            # Return recvobj
            return(recvobj)

        # Specialized recv function that automatically makes use of buffers
        def recv(self, buf=None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                 status=None):
            """
            Special receive method that automatically uses the appropriate
            method (:meth:`~MPI.Intracomm.recv` or :meth:`~MPI.Intracomm.Recv`)
            depending on the type of the object provided to :meth:`~send`.

            Optional
            --------
            buf : None. Default: None
                The `buf` argument that the :meth:`~MPI.Intracomm.recv` method
                takes. As the received object is always returned, this argument
                has no use, but is here to ensure that the method signature is
                the same.
            source : int. Default: :obj:`~mpi4py.MPI.ANY_SOURCE`
                The integer identifier of the MPI rank where the object will be
                sent from.
            tag : int. Default: :obj:`~mpi4py.MPI.ANY_TAG`
                The tag used for the send/receive communication between this
                rank and `recv`.
            status : :obj:`~mpi4py.MPI.Status` object or None. Default: None
                If not *None*, the status object to use for storing the status
                of this communication process.

            Returns
            -------
            recvobj : object
                The object that was received from `source`.

            """

            # Check if a buffer will be used
            use_buffer = use_buffer_meth(None, source, tag)

            # If to-be-received object uses a buffer, use Recv
            if use_buffer:
                # Create NumPy array with given shape and dtype
                recvobj = np.empty(*comm.recv(source=source, tag=tag))

                # Receive NumPy array
                comm.Recv(recvobj, source=source, tag=tag, status=status)

            # If not, receive obj the normal way
            else:
                recvobj = comm.recv(source=source, tag=tag, status=status)

            # Return recvobj
            return(recvobj)

        # Specialized scatter function that automatically makes use of buffers
        def scatter(self, sendobj, root=0):
            """
            Special scatter method that automatically uses the appropriate
            method (:meth:`~MPI.Intracomm.scatter` or
            :meth:`~MPI.Intracomm.Scatter`) depending on the type of the
            provided `sendobj`.

            Unlike :meth:`~MPI.Intracomm.scatter`, providing a buffer object
            with more than :attr:`~_size` items will not raise an error, but
            evenly distribute all the items instead.

            Parameters
            ----------
            sendobj : :obj:`~numpy.ndarray` or object
                The object to scatter to all MPI ranks.
                If :obj:`~numpy.ndarray`, use :meth:`~MPI.Intracomm.Scatterv`.
                If not, use :meth:`~MPI.Intracomm.scatter` instead.

            Optional
            --------
            root : int. Default: 0
                The MPI rank that scatters `sendobj`.

            Returns
            -------
            recvobj : :obj:`~numpy.ndarray` or object
                The object that has been scattered to this MPI rank.

            """

            # Check if obj can be scattered as buffer objects
            use_buffer = use_buffer_meth(sendobj, root)

            # If provided object uses a buffer
            if use_buffer:
                # Sender prepares for scattering
                if(self._rank == root):
                    # Raise error if length of axis is not divisible by size
                    if len(sendobj) % self._size:  # pragma: no cover
                        raise e13.ShapeError("Input argument 'sendobj' cannot "
                                             "be divided evenly over the "
                                             "available number of MPI ranks!")

                    # Determine shape of scattered object
                    buff_shape = list(sendobj.shape)
                    buff_shape[0] //= self._size

                    # Initialize empty buffer array
                    recvobj = np.empty(
                        *comm.bcast([buff_shape, sendobj.dtype], root=root))

                    # Scatter NumPy array
                    comm.Scatter(sendobj, recvobj, root=root)

                # Receivers receive the array
                else:
                    # Initialize empty buffer array
                    recvobj = np.empty(*comm.bcast(None, root=root))

                    # Receive scattered NumPy array
                    comm.Scatter(None, recvobj, root=root)

                # Remove single dimensional entries from recvobj
                recvobj = recvobj.squeeze()

            # If not, scatter obj the normal way
            else:
                recvobj = comm.scatter(sendobj, root=root)

            # Return recvobj
            return(recvobj)

        # Specialized send function that automatically makes use of buffers
        def send(self, obj, dest, tag=0):
            """
            Special send method that automatically uses the appropriate
            method (:meth:`~MPI.Intracomm.send` or :meth:`~MPI.Intracomm.Send`)
            depending on the type of the provided `obj`.

            Parameters
            ----------
            obj : :obj:`~numpy.ndarray` or object
                The object to send to the MPI rank `dest`.
                If :obj:`~numpy.ndarray`, use :meth:`~MPI.Intracomm.Send`.
                If not, use :meth:`~MPI.Intracomm.send` instead.
            dest : int
                The integer identifier of the MPI rank where `obj` must be sent
                to.

            Optional
            --------
            tag : int. Default: 0
                The tag used for the send/receive communication between this
                rank and `dest`.

            """

            # Check if obj can be sent as a buffer object
            use_buffer = use_buffer_meth(obj, dest, tag)

            # If provided object uses a buffer, use Send
            if use_buffer:
                # Send the shape and dtype of obj to receiver
                comm.send([obj.shape, obj.dtype], dest=dest, tag=tag)

                # Then send the NumPy array as a buffer object
                comm.Send(obj, dest=dest, tag=tag)

            # If not, send obj the normal way
            else:
                comm.send(obj, dest=dest, tag=tag)

    # %% UTILITY FUNCTIONS
    # This function checks if a buffer communication method can be used
    def use_buffer_meth(obj, src_dest, tag=0):
        """
        Depending on which communication method calls this function,
        determines if the provided `obj` on all MPI ranks can be
        communicated using an uppercase communication method.

        This function must be called by all MPI ranks that are communicating.
        This function must never be called directly.

        """

        # Determine the name of the frame calling this method
        meth_name = currentframe().f_back.f_code.co_name

        # Check who called this method and act accordingly
        # SEND/RECV
        if meth_name in ('recv', 'send'):
            # SEND
            if(meth_name == 'send'):
                # Determine if this object is a buffer object
                buff_flag = is_buffer_obj(obj)

                # Send this to the receiver
                comm.send(buff_flag, dest=src_dest, tag=tag)

                # Return buff_flag
                return(buff_flag)

            # RECV
            else:
                # Receive and return buff_flag
                return(comm.recv(obj, source=src_dest, tag=tag))

        # BCAST/SCATTER
        elif meth_name in ('bcast', 'scatter'):
            return(comm.bcast(is_buffer_obj(obj), root=src_dest))

        # GATHER
        elif(meth_name == 'gather'):
            return(comm.allreduce(is_buffer_obj(obj), op=MPI.MIN))

        # NOT IMPLEMENTED
        else:  # pragma: no cover
            raise NotImplementedError

    # %% REMAINDER OF FUNCTION FACTORY
    # Initialize HybridComm
    hybrid_comm = HybridComm()

    # Register initialized HybridComm
    hybrid_comm_registry[hex(id(comm))] = hybrid_comm

    # Return hybrid_comm
    return(hybrid_comm)


# %% DEFAULT INSTANCES
HYBRID_COMM_SELF = get_HybridComm_obj(MPI.COMM_SELF)
HYBRID_COMM_WORLD = get_HybridComm_obj(MPI.COMM_WORLD)
