#############################
# Copyright 2017 Otmar Ertl #
#############################

import unittest
import zipf_mandelbrot_random_generator
import numpy
import scipy.stats

class TestZipfMandelbrotGenerator(unittest.TestCase):

    def test1(self):
        N = 1000000
        n = 10
        q = 1
        v = 1

        expectedRaw = numpy.array([pow(v+k,-q) for k in range(0, n)])
        expectedRawSum = expectedRaw.sum
        expected = expectedRaw/(numpy.sum(expectedRaw)) * N
        gen = zipf_mandelbrot_random_generator.ZipfMandelbrotRandomGenerator(n, q, v)
        observed = numpy.zeros(n)
        for i in range(0,N):
            observed[next(gen)] += 1
        _, p = scipy.stats.chisquare(observed, expected)
        self.assertGreater(p, 0.001)

    def test2(self):
        N = 1000000
        n = 11
        q = 0.5
        v = 0.5

        expectedRaw = numpy.array([pow(v+k,-q) for k in range(0, n)])
        expectedRawSum = expectedRaw.sum
        expected = expectedRaw/(numpy.sum(expectedRaw)) * N
        gen = zipf_mandelbrot_random_generator.ZipfMandelbrotRandomGenerator(n, q, v)
        observed = numpy.zeros(n)
        for i in range(0,N):
            observed[next(gen)] += 1
        _, p = scipy.stats.chisquare(observed, expected)
        self.assertGreater(p, 0.001)

    def test3(self):
        N = 1000000
        n = 7
        q = 1.5
        v = 1.5

        expectedRaw = numpy.array([pow(v+k,-q) for k in range(0, n)])
        expectedRawSum = expectedRaw.sum
        expected = expectedRaw/(numpy.sum(expectedRaw)) * N
        gen = zipf_mandelbrot_random_generator.ZipfMandelbrotRandomGenerator(n, q, v)
        observed = numpy.zeros(n)
        for i in range(0,N):
            observed[next(gen)] += 1
        _, p = scipy.stats.chisquare(observed, expected)
        self.assertGreater(p, 0.001)

    def test4(self):
        N = 1000000
        n = 8
        q = 0
        v = 0.5

        expectedRaw = numpy.array([pow(v+k,-q) for k in range(0, n)])
        expectedRawSum = expectedRaw.sum
        expected = expectedRaw/(numpy.sum(expectedRaw)) * N
        gen = zipf_mandelbrot_random_generator.ZipfMandelbrotRandomGenerator(n, q, v)
        observed = numpy.zeros(n)
        for i in range(0,N):
            observed[next(gen)] += 1
        _, p = scipy.stats.chisquare(observed, expected)
        self.assertGreater(p, 0.001)



if __name__ == '__main__':
    unittest.main()
