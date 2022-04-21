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
            SLOTS_PTR]
        self.generate_clueg.restype = ctypes.c_int

cuda_lib = CudaLib(dll_obj)

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

        assert len(secrets.shape) == 2, "secrets must be 2D"
        assert len(guesses.shape) == 2, "guesses must be 2D"
        assert len(clues.shape) == 3, "clues must be 2D"

        assert secrets.dtype == np.uint8, "array dtype must be uint8"
        assert guesses.dtype == np.uint8, "array dtype must be uint8"
        assert secrets.dtype == np.uint8, "array dtype must be uint8"

        assert secrets.shape[1] == NUM_SLOTS, f"last dim must be NUM_SLOTS ({NUM_SLOTS})"
        assert guesses.shape[1] == NUM_SLOTS, f"last dim must be NUM_SLOTS ({NUM_SLOTS})"
        assert clues.shape[2] == NUM_SLOTS, f"last dim must be NUM_SLOTS ({NUM_SLOTS})"
        assert clues.shape[0] >= guesses.shape[0] and clues.shape[1] >= secrets.shape[0], \
            "clues is not large enough"

        assert secrets.__array_interface__['strides'] == None, "strides not supported"
        assert guesses.__array_interface__['strides'] == None, "strides not supported"
        assert clues.__array_interface__['strides'] == None, "strides not supported"

        assert 'data' in secrets.__array_interface__, "no data available"
        assert 'data' in guesses.__array_interface__, "no data available"
        assert 'data' in clues.__array_interface__, "no data available"

        secret_ptr = ctypes.cast(secrets.__array_interface__['data'][0], SLOTS_PTR)
        guess_ptr = ctypes.cast(guesses.__array_interface__['data'][0], SLOTS_PTR)
        clue_ptr = ctypes.cast(clues.__array_interface__['data'][0], SLOTS_PTR)

        retval = cuda_lib.generate_clueg(
            self.ctx_handle,
            guess_ptr,
            guesses.shape[0],
            secret_ptr,
            secrets.shape[0],
            clue_ptr)

        if retval < 0:
            raise RuntimeError(f"CUDA lib returned nonzero: {retval}")

    def __exit__(self, exc_type, exc_value, traceback):
        cuda_lib.free_context(self.ctx_handle)
        self.ctx_handle = ctypes.c_void_p()

    @staticmethod
    def test_binding():
        cuda_lib.helloworld()
