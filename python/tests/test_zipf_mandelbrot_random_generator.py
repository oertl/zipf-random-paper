#############################
# Copyright 2017 Otmar Ertl #
#############################

import unittest
import zipf_mandelbrot_random_generator
import numpy
import scipy.stats

class TestZipfMandelbrotGenerator(unittest.TestCase):

    def run_test(self, n, q, v):
        N = 1000000

        expectedRaw = numpy.array([pow(v+k,-q) for k in range(0, n)])
        expected = expectedRaw/(numpy.sum(expectedRaw)) * N
        gen = zipf_mandelbrot_random_generator.ZipfMandelbrotRandomGenerator(n, q, v)
        observed = numpy.zeros(n)
        for i in range(0,N):
            observed[next(gen)] += 1
        _, p = scipy.stats.chisquare(observed, expected)
        self.assertGreater(p, 0.001)

    def test1(self):
        self.run_test(n = 10, q = 1, v = 1)

    def test2(self):
        self.run_test(n = 11, q = 0.5, v = 0.5)

    def test3(self):
        self.run_test(n = 7, q = 1.5, v = 1.5)

    def test4(self):
        self.run_test(n = 8, q = 0, v = 0.5)

    def test5(self):
        self.run_test(n = 8, q = numpy.nextafter(1,1), v = 0.5)

    def test6(self):
        self.run_test(n = 8, q = numpy.nextafter(1,-1), v = 0.5)

    def test7(self):
        self.run_test(n = 8, q = 0.5, v = 1000)

    def test8(self):
        self.run_test(n = 10, q = 1, v = 0.1)

    def test9(self):
        self.run_test(n = 10, q = 2, v = 0.1)

    def test10(self):
        self.run_test(n = 10, q = 0.5, v = 0.1)

    def test11(self):
        self.run_test(n = 10, q = 0.5, v = 0.01)

    def test12(self):
        self.run_test(n = 10, q = 0.1, v = 0.01)

    def test13(self):
        self.run_test(n = 10, q = 0.01, v = 0.01)

    def test14(self):
        self.run_test(n = 10, q = 2, v = 1e5)

    def test15(self):
        self.run_test(n = 10, q = 2, v = 1e10)

    def test16(self):
        self.run_test(n = 10, q = 2, v = 1e15)

    def test17(self):
        self.run_test(n = 10, q = 2, v = 1e20)

    def test18(self):
        self.run_test(n = 10, q = 2, v = 1e-20)

    def test19(self):
        self.run_test(n = 10, q = 0, v = numpy.nextafter(0,1))

    def test20(self):
        self.run_test(n = 10, q = 0.1, v = numpy.nextafter(0,1))

if __name__ == '__main__':
    unittest.main()
