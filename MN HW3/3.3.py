# Problem 3.3
import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as scp


def Divided_difference(xp, yp):
    n = len(xp)
    difference = []
    coef = [[0 for _ in range(n)]
            for _ in range(n)]

    for i in range(n):
        coef[i][0] = yp[i]

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (xp[i + j] - xp[i])
    for i in range(n):
        if i != 0:
            difference.append(coef[0][i])
    return difference, coef


def Newton_Divided_difference(xp, yp, coef, xsym):
    n = len(xp)
    p = yp[0]
    fact = xsym - xp[0]
    for k in range(0, n-1):
        if k > 0:
            fact = fact * (xsym - xp[k])
        p += coef[k] * fact
    return p


x = sp.symbols('x')
func = sp.sqrt(x + 1)

xpoints_orig = np.linspace(-1, 1, num=500)
xpoints = np.linspace(-1, 1, num=8)  # Interpolant Polynomial of degree n on evenly spaced points => n+1 nodes

plt.plot(xpoints_orig, sp.lambdify(x, func)(xpoints_orig), label='f(x)')


def vals(n):
    fi = [math.pi/(2*n), -math.pi/(2*n)]
    fiel = math.pi/(2*n)
    cosinus = [math.cos(fi[0]), math.cos(fi[1])]
    it = 2
    while True:
        fiel += math.pi / n
        fi.append(fiel)
        fi.append(-fiel)
        cosinus.append(math.cos(fi[it]))
        it += 1
        cosinus.append(math.cos(fi[it]))
        it += 1
        if len([*set(cosinus)]) >= n:
            break
    return [*set(cosinus)]


print("Roots of Chebyshev Polynomial =", vals(8))  # roots of Chebyshev Polynomial

ypoints_chebyshev = []
for i in vals(8):
    ypoints_chebyshev.append(math.sqrt(i + 1))

dif_list_ch, coeff_ch = Divided_difference(vals(8), ypoints_chebyshev)
polyn_form_ch = Newton_Divided_difference(vals(8), ypoints_chebyshev, dif_list_ch, x)

print("T7(x) = ")
sp.pprint(polyn_form_ch)
print("Divided difference coefficients for Chebyshev Polynomial (first column - y values):")
for i in coeff_ch:
    print(i)

xpoints_ch = list(vals(8))
plt.plot(xpoints_orig, sp.lambdify(x, polyn_form_ch)(xpoints_orig), label="m7(x)", linewidth=1)

ypoints = []
for i in xpoints:
    ypoints.append(math.sqrt(i + 1))

dif_list, coeff = Divided_difference(xpoints, ypoints)
polyn_form = Newton_Divided_difference(xpoints, ypoints, dif_list, x)

print("P7(x) = ")
sp.pprint(polyn_form)
print("Divided difference coefficients (first column - y values):")
for i in coeff:
    print(i)

plt.plot(xpoints_orig, sp.lambdify(x, polyn_form)(xpoints_orig), label="P7(x)", linewidth=1)
plt.plot(xpoints, ypoints, 'ro')
plt.legend(loc='upper left')
plt.title("Original function, minimax polynomial approximation and polynomial interpolant of f(x)")
plt.grid("both")
plt.show()
