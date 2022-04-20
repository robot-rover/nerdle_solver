import unittest
import nerdle_cuda
import numpy as np

class TestCudaExtImport(unittest.TestCase):
    def test(self):
        test = nerdle_cuda.PythonClueContext(1,1)
