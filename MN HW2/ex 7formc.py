# Problem 7 form c
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math

x = sp.symbols('x')
func = x - (x+sp.log(x))/2
f_func = sp.lambdify(x, func, modules=['numpy'])

xpoints = np.linspace(0, 6, num=200)
ypoints = f_func(xpoints)
plt.plot(xpoints, ypoints, label=f'{sp.exp(x-sp.pi) + sp.cos(x) - x + sp.pi}')
plt.legend(loc='upper center', shadow=True)
plt.grid('both')
plt.show()

# Problem 4b


def Newton_method(f, df, epsilon, x0, list):
    xn = x0
    for j in range(1, 1001):
        fxn = f(xn)
        dfxn = df(xn)
        xn1 = xn - fxn / dfxn
        list.append(abs(xn1-xn))
        if abs(xn1 - xn) < epsilon:
            print("====================================================")
            print("Iteration: ", j, "\t Found Root =", xn1)
            print("Error lower than tolerance =", abs(np.pi - xn1))
            print("Order of convergence =", math.log(list[j - 1] / list[j - 2]) / math.log(list[j - 2] / list[j - 3]))
            print("====================================================")
            return xn1, j
        else:
            print("Iteration: ", j, "\t Approximation =", xn1)
            print("Estimation of error =", abs(np.pi - xn1))
            if j > 2:
                print("Order of convergence =", math.log(list[j-1] / list[j - 2]) / math.log(list[j - 2] / list[j - 3]))
        xn = xn1


def deriv(f):
    return sp.diff(f, x)


err = []
deq = sp.diff(func, x)
deq_func = sp.lambdify(x, deq, modules=['numpy'])
xi0 = 0.5
tolerance = 0.000001
a, iter1 = Newton_method(f_func, deq_func, tolerance, xi0, err)

m = 1
fmul = f_func
fpot = func
if fmul(np.pi) == 0:
    while sp.lambdify(x, deriv(fpot), modules=['numpy'])(np.pi) < tolerance:
        fpot = deriv(fpot)
        m += 1
print("Multiplicity =", m)