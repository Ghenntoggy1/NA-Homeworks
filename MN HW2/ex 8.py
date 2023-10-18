# Problem 6a
import numpy as np


def lambda_n(xc2, xc1, xc0):
    return np.double((xc2 - xc1) / (xc1 - xc0))


xlist = [np.double(x) for x in [1, 0.36788, 0.69220, 0.50047, 0.60624, 0.54540, 0.57961]]
err = [np.double(x) for x in [-6.3212e-1, 3.2432e-1, -1.9173e-1, 1.0577e-1, -6.0848e-2, 3.4217e-2]]

for x in range(2, 7):
    lmbd = abs(lambda_n(xlist[x], xlist[x-1], xlist[x-2]))
    if x < 6:
        print("Rate of convergence, computed by lambda_n =", lmbd)
        print("Rounded version of Rate of convergence =", round(lmbd, 2))
    else:
        print("========================================")
        print("Rate of convergence, computed by lambda_n =", lmbd)
        print("Rounded version of Rate of convergence =", round(lmbd, 2))
        # frac = Fraction(round(lmbd, 2))
        frac = round(1/ (1-lmbd))
        print("As we see, by Aitken Error Estimation Formula, lambda_n, that can be used as Rate of Convergence"
              "when n tends to infinity, is fluctuating between 0.51 and 0.59, therefore it represents a Linear"
              "Convergence.")
        print("Fixed point iteration Method, with this rate of convergence represents Linear Convergence with "
              "rate approximately 0.56, that is slower than Bisection Method, but not too much.")
        print("In order to accelerate the convergence of this method I would propose to use Aitken Algorithm, that will"
              "accelerate the convergence of the function")
        print("========================================")
