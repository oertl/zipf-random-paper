#############################
# Copyright 2017 Otmar Ertl #
#############################

from math import log, exp, expm1, log1p
from random import Random

# generator for random numbers with probability mass function
# p_k = 1 / (v + k)^q for 0 <= k < n
# q >= 0 and v > 0
def ZipfMandelbrotRandomGenerator(n, q, v = 1, seed = None):

    # helper function that calculates log(1 + x) / x
    def helper1(x):
        if x == -1.:
            return float('inf')
        elif x == 0.:
            return 1.
        else:
            return log1p(x)/x

    # helper function to calculate (exp(x) - 1) / x
    def helper2(x):
        return expm1(x) / x if x != 0. else 1.

    r = 1. - q
    a = v + 0.5
    b = 1. / a
    c = pow(a,  r)
    d = 1. / c

    # H(x) := ( (v + 0.5 + x)^(1-q) - (v + 0.5)^(1-q) ) / (1 - q), if q != 1
    # H(x) := log(1 + x / (v + 0.5) ), if q == 1
    def H(x):
        logVpX = log1p(x * b)
        return helper2( r * logVpX ) * logVpX * c

    # h(x) := H'(x) = 1 / (v+x)^q
    def h(x):
        logVpX = log(v+x)
        return exp(-q * logVpX)

    def Hmh(x):
        return H(x) - h(x)

    # H_inv(H(x)) := x
    def H_inv(x):
        t = x * r * d
        if (t < -1):
            # t could be smaller than -1 in some rare cases due to numerical errors
            print("t = " + str(t))
            t = -1
            assert False
        return a * expm1(helper1(t) * d * x) # >= 0

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
            k = int(x) + 1 # >= 1

            # k could be larger than n-1 due to numerical inaccuracies
            if k >= n:
                k = n - 1
                assert False

            if (k - 1 <= x - s or u >= Hmh(k)):

                yield k
