from distutils.core import setup, Extension
import os
import numpy as np

if 'CUDA_PATH' in os.environ:
   CUDA_PATH = os.environ['CUDA_PATH']
   print(f'Found CUDA: {CUDA_PATH}')
else:
   raise RuntimeError("Could not find CUDA_PATH in environment variables.")

if not os.path.isdir(CUDA_PATH):
   print("CUDA_PATH {} not found. Please update the CUDA_PATh variable and rerun".format(CUDA_PATH))
   exit(0)

setup(name = 'nerdle_cuda', version = '1.0', packages=['nerdle_cuda'], ext_modules=[
    Extension('nerdle_cuda_ext',
    ['python_lib.c'],
    include_dirs=[np.get_include()],
    libraries=["cuda_lib_rt", "cudart"],
    library_dirs = [".", os.path.join(CUDA_PATH, "lib","x64")])])