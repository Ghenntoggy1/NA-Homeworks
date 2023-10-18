# Problem 7formab
import numpy as np
import sympy
import math


def fixed_iterative_method(f, epsilon, x0, lst):
    x1 = x0
    error = 1
    j = 0
    xlist = []
    while error > epsilon:
        j += 1
        x2 = f(x1)
        error = abs(x2 - x1)
        lst.append(abs(x2-x1))
        xlist.append(x2)
        if abs(x2 - x1) < epsilon:
            print("====================================================")
            print("Iteration: ", j, "\t Found Root =", x2)
            print("Error lower than tolerance =", abs(x1 - x2))
            if j > 2 and lst[j-1] != 0 and lst[j-2] != 0 and lst[j-3] != 0:
                print("Order of convergence =", math.log(lst[j - 1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))
            print("====================================================")
            return x2, j
        else:
            print("Iteration: ", j, "\t Approximation =", x2)
            print("Estimation of error =", abs(x1 - x2))
            if j > 2 and lst[j-1] != 0 and lst[j-2] != 0 and lst[j-3] != 0:
                print("Order of convergence =", math.log(lst[j - 1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))
        x1 = x2


x = sympy.symbols('x')
func = x - (x+sympy.log(x))/2
err2 = []
f_func = sympy.lambdify(x, func, modules=['numpy'])
j = 0
xi0 = 15
tolerance = 0.000001
a2 = fixed_iterative_method(f_func, tolerance, xi0, err2)