# Problem 4c
import numpy as np
import sympy
import math


def Newton_method2(f, df, epsilon, x0, lst):
    xn = x0
    for j in range(1, 1001):
        fxn = f(xn)
        dfxn = df(xn)
        xn1 = xn - 2*(fxn / dfxn)  # I took multiplicity 2 that I found in previous part of this exercise
        lst.append(abs(xn1-xn))
        if abs(xn1 - xn) < epsilon:
            print("====================================================")
            print("Iteration: ", j, "\t Found Root =", xn1)
            print("Error lower than tolerance =", abs(np.pi - xn1))
            if j > 2 and lst[j-2] != 0 and lst[j-3] != 0 and lst[j-4] != 0:
                print("Order of convergence =", math.log(lst[j - 2] / lst[j - 3]) / math.log(lst[j - 3] / lst[j - 4]))
            print("====================================================")
            return xn1, j
        else:
            print("Iteration: ", j, "\t Approximation =", xn1)
            print("Estimation of error =", abs(np.pi - xn1))
            if j > 2 and lst[j-1] != 0 and lst[j-2] != 0 and lst[j-3] != 0:
                print("Order of convergence =", math.log(lst[j-1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))
        xn = xn1


x = sympy.symbols('x')
func = sympy.exp(x-sympy.pi) + sympy.cos(x) - x + sympy.pi
f_func = sympy.lambdify(x, func, modules=['numpy'])

err2 = []
deq = sympy.diff(func, x)
deq2 = sympy.diff(deq, x)
deq_func = sympy.lambdify(x, deq, modules=['numpy'])
deq_func2 = sympy.lambdify(x, deq2, 'numpy')
xi0 = 0
tolerance = 0.000001
a2, iter12 = Newton_method2(f_func, deq_func, tolerance, xi0, err2)

# As we see, Order of convergence became quadratic after implementing Newton's Method on second derivative of our
# function. Also, root became more accurate, and error became much smaller.
