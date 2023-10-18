# Problem 5a
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
func = sympy.cos(x) - 1 + x
err2 = []
f_func = sympy.lambdify(x, func, modules=['numpy'])
j = 0
xi0 = 0.1
tolerance = 0.000001
a2 = fixed_iterative_method(f_func, tolerance, xi0, err2)

# Converges VERY slow, found root on iteration 1391, that grants an error smaller than tolerance

# Problem 5b

def bisection_method(f, a, b, epsilon, ji, lst):
    c = (a+b)/2
    if np.abs(f(c)) < epsilon:
        ji += 1
        error = np.abs(f(c)) - 0
        lst.append(error)
        print("====================================================")
        print("Iteration:", ji)
        print("Root =", c)
        print("Error lower than tolerance =", np.abs(f(c)) - 0)
        if ji > 2 and lst[j - 1] != 0 and lst[j - 2] != 0 and lst[j - 3] != 0:
            print("Order of convergence =", math.log(lst[j - 1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))
        print("====================================================")
        return c, ji
    elif np.sign(f(a)) == np.sign(f(c)):
        ji += 1
        error = np.abs(f(c) - 0)
        lst.append(error)
        print("Iteration:", ji)
        print("Root approximation =", c)
        print("Error lower than tolerance =", np.abs(f(c)) - 0)
        if ji > 2:
            print("Order of convergence =", math.log(lst[j - 1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))
        return bisection_method(f, c, b, epsilon, ji, lst)
    elif np.sign(f(b)) == np.sign(f(c)):
        ji += 1
        error = np.abs(f(c)) - 0
        lst.append(error)
        print("Iteration:", ji)
        print("Root approximation=", c)
        print("Error estimation =", np.abs(f(c) - 0))
        if ji > 2:
            print("Order of convergence =", math.log(lst[j - 1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))
        return bisection_method(f, a, c, epsilon, ji, lst)

err1 = []
bisection_method(f_func, 0, 10, 0.000001, j, err1)

# Bisection Method is a lot faster that method we used initially (fixed point iteration method)
