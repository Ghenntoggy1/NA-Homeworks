# Problem 6a
import numpy as np
from fractions import Fraction


def lambda_n(xc2, xc1, xc0):
    return np.double((xc2 - xc1) / (xc1 - xc0))


xlist = [np.double(x) for x in [2, 2.1248, 2.2148, 2.2805, 2.3289, 2.3647,
                                2.3913, 2.4111, 2.4260, 2.4370, 2.4453]]
err = [np.double(x) for x in [1, 0.124834, 0.089944, 0.065698, 0.048386, 0.035827,
                              0.026624, 0.019835, 0.014803, 0.011062, 0.0082745]]

for x in range(2, 11):
    lmbd = lambda_n(xlist[x], xlist[x-1], xlist[x-2])
    if x < 10:
        print("Rate of convergence, computed by lambda_n =", lmbd)
        print("Rounded version of Rate of convergence =", round(lmbd, 2))
    else:
        print("========================================")
        print("Rate of convergence, computed by lambda_n =", lmbd)
        print("Rounded version of Rate of convergence =", round(lmbd, 2))
        frac = Fraction(round(lmbd, 2))
        print("Newton's Method with this rate of convergence represents Linear Convergence with rate 0.75, \n"
              "that is slower than Bisection Method")
        print("Lambda_n in form of fraction =", frac)
        frac_str = str(frac)
        print("Therefore, Multiplicity of root of f(x) =", frac_str[2])
        print("That's why Newton's Method in this case has a Linear Convergence (root has a multiplicity higher than 1)")
        print("Knowing function f(x), I would solve instead of f(x) = 0, its derivative, \n"
              "that in our case will be of order 34 (m = 4 => compute f'''(x) = 0, i.e. derivative of order 4-1,\n"
              "which is 3) or would consider the fixed point iteration method")
        print("========================================")
