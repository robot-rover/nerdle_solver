import ctypes
import win32api
import win32con
import os
import numpy as np

NUM_SLOTS = 8

dll_name = 'cuda_lib_sh.dll'
dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dll_name)
dll_handle = win32api.LoadLibraryEx(dll_path, 0, win32con.LOAD_WITH_ALTERED_SEARCH_PATH)
dll_obj = ctypes.CDLL(dll_name)

SLOTS_PTR = ctypes.POINTER(ctypes.c_uint8)
CLUES_PTR = ctypes.POINTER(ctypes.c_uint16)
ENTROPY_PTR = ctypes.POINTER(ctypes.c_double)

class CudaLib:
    def __init__(self, lib):
        self.helloworld = lib.helloworld
        self.helloworld.argtypes = []
        self.helloworld.restype = None

        self.create_context = lib.create_context
        self.create_context.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.create_context.restype = ctypes.c_void_p

        self.free_context = lib.free_context
        self.free_context.argtypes = [ctypes.c_void_p]
        self.free_context.restype = None

        self.generate_clueg = lib.generate_clueg
        self.generate_clueg.argtypes = [ctypes.c_void_p,
            SLOTS_PTR, ctypes.c_uint32,
            SLOTS_PTR, ctypes.c_uint32,
            CLUES_PTR]
        self.generate_clueg.restype = ctypes.c_int

        self.generate_entropies = lib.generate_entropies
        self.generate_entropies.argtypes = [ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ENTROPY_PTR,
            ctypes.c_bool
        ]

cuda_lib = CudaLib(dll_obj)

def _numpy_to_ptr(array, name, ndims, dtype, ptr_type, req_dims=None):
    assert len(array.shape) == ndims, f"{name} must be {ndims}D"
    assert array.dtype == dtype, f"{name} dtype must be {dtype}"
    if req_dims is not None:
        assert len(req_dims) == len(array.shape), \
            f"{name} req dim len ({len(req_dims)}) doesn't match array dims ({len(array.shape)})"
        for dim_id, (req_dim, act_dim) in enumerate(zip(req_dims, array.shape)):
            if req_dim is None:
                continue
            op, req_dim = req_dim
            if op == '>':
                assert act_dim > req_dim, f'{name} dim mismatch on number {dim_id}, ({act_dim} <= {req_dim})'
            if op == '<':
                assert act_dim < req_dim, f'{name} dim mismatch on number {dim_id}, ({act_dim} >= {req_dim})'
            if op == '=':
                assert act_dim == req_dim, f'{name} dim mismatch on number {dim_id}, ({act_dim} != {req_dim})'
            if op == '>=':
                assert act_dim >= req_dim, f'{name} dim mismatch on number {dim_id}, ({act_dim} < {req_dim})'
            if op == '<=':
                assert act_dim <= req_dim, f'{name} dim mismatch on number {dim_id}, ({act_dim} > {req_dim})'
    assert 'data' in array.__array_interface__, f"{name}: no data available"
    assert array.__array_interface__['strides'] == None, f"{name}: strides not supported"
    ptr = ctypes.cast(array.__array_interface__['data'][0], ptr_type)
    return ptr

class PythonClueContext:
    def __init__(self, num_guesses, num_secrets):
        self.num_secrets = num_secrets
        self.num_guesses = num_guesses
        self.ctx_handle = ctypes.c_void_p()

    def __enter__(self):
        self.ctx_handle = cuda_lib.create_context(self.num_guesses, self.num_secrets)
        return self

    def generate_clue(self, guesses, secrets, clues):
        assert self.ctx_handle != ctypes.c_void_p(), "Context not initialized"

        guess_ptr = _numpy_to_ptr(guesses, "guesses", 2, np.uint8, SLOTS_PTR, (None, ('=', NUM_SLOTS)))
        secret_ptr = _numpy_to_ptr(secrets, "secrets", 2, np.uint8, SLOTS_PTR, (None, ('=', NUM_SLOTS)))

        clue_ptr = CLUES_PTR()
        if clues is not None:
            clue_ptr = _numpy_to_ptr(clues, "clues", 2, np.uint16, CLUES_PTR,
                (('>=', guesses.shape[0]), ('>=', secrets.shape[0])))

        retval = cuda_lib.generate_clueg(
            self.ctx_handle,
            guess_ptr,
            guesses.shape[0],
            secret_ptr,
            secrets.shape[0],
            clue_ptr)

        if retval < 0:
            raise RuntimeError(f"CUDA lib returned nonzero: {retval}")

    def generate_entropies(self, guesses, secrets, entropies, use_sort_alg=False):
        assert self.ctx_handle != ctypes.c_void_p(), "Context not initialized"

        entropy_ptr = _numpy_to_ptr(entropies, 'entropies', 1, np.double, ENTROPY_PTR, (('=',guesses.shape[0]),))

        self.generate_clue(guesses, secrets, None)
        retval = cuda_lib.generate_entropies(self.ctx_handle, guesses.shape[0], secrets.shape[0], entropy_ptr, use_sort_alg)
        if retval < 0:
            raise RuntimeError(f"CUDA lib returned nonzero: {retval}")

        if np.isnan(entropies).any():
            raise RuntimeError("Overflow on GPU Kernel Counts")

    def __exit__(self, exc_type, exc_value, traceback):
        cuda_lib.free_context(self.ctx_handle)
        self.ctx_handle = ctypes.c_void_p()

    @staticmethod
    def test_binding():
        cuda_lib.helloworld()
