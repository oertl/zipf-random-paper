#############################
# Copyright 2017 Otmar Ertl #
#############################

from math import log, exp, expm1, log1p
from random import Random

# generator for random numbers with probability mass function
# p_k = 1 / (v + k)^q for 0 <= k < n
# q >= 0 and v > 0
def ZipfMandelbrotRandomGenerator(n, q = 1., v = 1., seed = None):

    r = 1. - q     # <= 1
    a = v + 0.5    # >= 0.5
    b = 1. / a     # in (0, 2]
    c = pow(a, r)  # >= 0
    d = 1. / c     # >= 0

    # helper function that calculates log(1 + x) / x
    def helper1(x):
        return log1p(x) / x if x != 0. else 1.

    # helper function to calculate (exp(x) - 1) / x
    def helper2(x):
        return expm1(x) / x if x != 0. else 1.

    # H(x) := ( (v + 0.5 + x)^(1-q) - (v + 0.5)^(1-q) ) / (1 - q), if q != 1
    # H(x) := log(1 + x / (v + 0.5) ), if q == 1
    def H(x):
        w = log1p(x * b)
        return helper2( r * w ) * w * c

    # h(x) := H'(x - 0.5) = 1 / (v+x)^q
    def h(x):
        return exp(-q * log(v+x))

    def Hmh(x):
        return H(x) - h(x)

    # H_inv(H(x)) := x
    def H_inv(x): # x >= 0
        dx = d * x # >= 0
        t = dx * r
        if (t <= -1):
            return float('inf')
        return a * expm1(helper1(t) * dx) # >= 0

    h0 = h(0)
    s = H_inv(max(Hmh(1.), 0.))
    Hn = H(n - 1) + h0
    rnd = Random(seed)

    while True:
        u = rnd.random() * Hn - h0
        if u < 0:
            yield 0
        else:
            x = H_inv(u)   # >= 0
            if x < n - 1:
                k = int(x) + 1 # >= 1
                if (k - 1 <= x - s or u >= Hmh(k)):
                    yield k
