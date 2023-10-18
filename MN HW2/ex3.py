# Problem 2.3
import sympy as sym
import math


def Newton_method(f, df, epsilon, x0, list, fl):
    xn = x0
    for j in range(1, 1001):
        fxn = f(xn)
        dfxn = df(xn)
        xn1 = xn - fxn / dfxn
        if not fl:
            list.append(abs(xn1-xn))
        print("Iteration:", j, "\tApproximation =", xn1)
        print("Estimation of error =", abs(xn1 - xn))
        if abs(xn1 - xn) < epsilon:
            return xn1, j
        xn = xn1


R = sym.symbols('R')

eq1 = 1.129241*10**(-3) + (2.341077*10**(-4))*sym.log(R) + (8.775468*10**(-8))*(sym.log(R))**3 - (1/(19.01 + 273.15))
eq2 = 1.129241*10**(-3) + (2.341077*10**(-4))*sym.log(R) + (8.775468*10**(-8))*(sym.log(R))**3 - (1/(18.99 + 273.15))

deq1 = sym.diff(eq1, R)
deq2 = sym.diff(eq2, R)

eq1_func = sym.lambdify(R, eq1, 'numpy')
eq2_func = sym.lambdify(R, eq2, 'numpy')
deq1_func = sym.lambdify(R, deq1, 'numpy')
deq2_func = sym.lambdify(R, deq2, 'numpy')

tolerance = 0.000001
R0 = 15000

err = []
flag = False
a, iter1 = Newton_method(eq1_func, deq1_func, tolerance, R0, err, flag)
print("==============================================")
flag = True
b, iter2 = Newton_method(eq2_func, deq2_func, tolerance, R0, err, flag)
print("==============================================")
print("First Equation Root R =", a, "on iteration:", iter1)
print("Second Equation Root R =", b, "on iteration:", iter2)
lst = [round(a, 5), round(b, 5)]
print("Range for resistance values:", lst)

ofc = 0
c = 0
# print(err)
for i in range(1, len(err)-1):
    if err[i-1] != float(0) and err[i] != float(0) and err[i+1] != float(0):
        ofc += (math.log(err[i + 1] / err[i]) / math.log(err[i] / err[i - 1]))
        c += 1
print("========================================")
print("Order of convergence =", ofc / c)
print("Order of convergence specific x3 x2 x1 =", (math.log(err[3] / err[2]) / math.log(err[2] / err[1])))

# Order of converge 2 - quadratic convergence, which corresponds to Newton's Method
