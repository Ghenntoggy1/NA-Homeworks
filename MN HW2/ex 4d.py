# Problem 4d
import numpy as np
import sympy
import math


def fixed_iterative_method(f, epsilon, x0, lst):
    x1 = x0
    error = 1
    j = 0
    while error > epsilon:
        j += 1
        x_next = f(x1)
        error = abs(x_next - x1)
        lst.append(abs(x_next-x1))
        if abs(x_next - x1) < epsilon:
            print("====================================================")
            print("Iteration: ", j, "\t Found Root =", x_next)
            print("Error lower than tolerance =", abs(np.pi - x_next))
            if j > 2 and lst[j-1] != 0 and lst[j-2] != 0 and lst[j-3] != 0:
                print("Order of convergence =", math.log(lst[j - 1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))
            print("====================================================")
            return x_next, j
        else:
            print("Iteration: ", j, "\t Approximation =", x_next)
            print("Estimation of error =", abs(np.pi - x_next))
            if j > 2 and lst[j-1] != 0 and lst[j-2] != 0 and lst[j-3] != 0:
                print("Order of convergence =", math.log(lst[j - 1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))
        x1 = x_next


x = sympy.symbols('x')
func = sympy.exp(x-sympy.pi) + sympy.cos(x) + sympy.pi
err2 = []
f_func = sympy.lambdify(x, func, modules=['numpy'])
xi0 = 3
tolerance = 0.000001
a2 = fixed_iterative_method(f_func, tolerance, xi0, err2)

# Converges very slowly, on iteration 990, with Linear Convergence
