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
        return expm1(x) / x if abs(x) > 0 else 1.

    # H(x) := ((v+x)^(1-q) - 1)/(1 - q), if q != 1
    # H(x) := log(v+x), if q == 1
    def H(x):
        logVpX = log(v+x)
        return helper2( (1. - q) * logVpX ) * logVpX

    # h(x) := H'(x) = 1/(v+x)^q
    def h(x):
        logVpX = log(v+x)
        return exp(-q * logVpX)

    # H_inv(H(x)) := x
    def H_inv(x):
        t = x * (1. - q)
        if (t < -1):
            # limit value to the range [-1, +inf)
            # t could be smaller than -1 in some rare cases due to numerical errors
            t = -1
        return exp(helper1(t) * x) - v

    Hx0 = H(0.5) - h(0)
    s = 1 - H_inv(H(1.5) - h(1))
    Hn = H(n - 0.5)
    rnd = Random(seed)

    while True:
        u = Hn + rnd.random() * (Hx0 - Hn)
        # u is uniformly distributed in (Hx0, Hn]

        x = H_inv(u)
        k = int(x + 0.5)

        # limit k to the range [0, n - 1], since
        # k could be outside due to numerical inaccuracies
        if k < 0:
            k = 0
        elif k > n - 1:
            k = n - 1

        # Here, the distribution of k is given by:
        #
        #   P(k = 0) = C * (H(0.5) - Hx0) = C * h(0) = C / v^q
        #   P(k = m) = C * (H(m + 1/2) - H(m - 1/2)) for 1 <= m < n
        #
        #   where C := 1 / (Hn - Hx0)

        if (k - x <= s or u >= H(k + 0.5) - h(k)):

            # Case k = 0:
            #
            #   The right inequality is always true, because replacing k by 0 gives
            #   u >= H(0.5) - h(0) = Hx0 and u is taken from
            #   (Hx0, Hn].
            #
            #   Therefore, the acceptance rate for k = 0 is P(accepted | k = 0) = 1
            #   and the probability that 1 is returned as random value is
            #   P(k = 0 and accepted) = P(accepted | k = 0) * P(k = 0) = C / v^q
            #
            # Case k >= 1:
            #
            #   The left inequality (k - x <= s) is just a short cut
            #   to avoid the more expensive evaluation of the right inequality
            #   (u >= H(k + 0.5) - h(k)) in many cases.
            #
            #   Hence, the right inequality determines the acceptance rate:
            #   H(m+1/2) - h(m) >= H(m-1/2), because h is convex
            #   P(accepted | k = m) = h(m) / (H(m+1/2) - H(m-1/2))
            #   The probability that m is returned is given by
            #   P(k = m and accepted) = P(accepted | k = m) * P(k = m) = C * h(m) = C / (v + m)^q.
            #
            # In both cases the probabilities are proportional to the probability mass function
            # of the Zipf distribution.

            yield k
