import unittest
import nerdle_cuda
import numpy as np

class TestCudaExtImport(unittest.TestCase):
    def test(self):
        nerdle_cuda.helloworld()

    def test_link(self):
        a = np.random.randn(1000)
        b = np.random.randn(1000)
        c = a + b
        d = nerdle_cuda.vector_add(a, b)
        np.testing.assert_array_equal(c, d)

    def test_cuda(self):
        a = np.random.randn(1000)
        b = np.random.randn(1000)
        c = a + b
        d = nerdle_cuda.vector_add_cuda(a, b)
        np.testing.assert_array_equal(c, d)
