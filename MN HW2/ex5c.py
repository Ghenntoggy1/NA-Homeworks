# Problem 5c
import sympy
import math


def lambda_n(xc2, xc1, xc0):
    return (xc2 - xc1) / (xc1 - xc0)


def gfunc(f, val):
    return f(val)


def aitken_algoritm(f, x0, epsilon, lst):
    xlist = [x0]
    xind = 0
    j = 0
    error = 1
    while error > epsilon:
        j += 1
        xlist.append(gfunc(f, xlist[xind]))
        xlist.append(gfunc(f, xlist[xind + 1]))
        xind += 2
        lmbd = lambda_n(xlist[xind], xlist[xind-1], xlist[xind-2])
        xlist.append(xlist[xind] + (lmbd/(1-lmbd))*(xlist[xind] - xlist[xind - 1]))
        lmbd = lambda_n(xlist[xind + 1], xlist[xind], xlist[xind - 1])
        error = math.fabs((lmbd/(1-lmbd))*(xlist[xind] - xlist[xind - 1]))
        xind += 1
        lst.append(error)
        if error < epsilon:
            print("====================================================")
            print("Iteration:", j)
            print("Root =", xlist[-1])
            print("Error lower than epsilon =", error)
            if j > 2:
                print("Order of convergence =", math.log(lst[j - 1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))
            print("====================================================")
            return xlist[-1], j
        else:
            print("Iteration:", j)
            print("Root =", xlist[xind])
            print("Error lower than tolerance =", error)
            if j > 2:
                print("Order of convergence =", math.log(lst[j - 1] / lst[j - 2]) / math.log(lst[j - 2] / lst[j - 3]))


x = sympy.symbols('x')
func = sympy.cos(x) - 1 + x
err2 = []
f_func = sympy.lambdify(x, func, modules=['numpy'])
xi0 = 0.1
tolerance = 0.000001
aitken_algoritm(f_func, xi0, tolerance, err2)

# Using Aitken Algorithm, we can achieve our root very fast.