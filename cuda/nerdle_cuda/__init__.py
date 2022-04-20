import os
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
from nerdle_cuda_ext import helloworld, PythonClueContext
