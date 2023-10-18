# Problem 3.1 a
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


def Newton_Divided_difference(yp, xp, coef, xsym):
    n = len(xp)
    p = yp[0]
    fact = xsym - xp[0]
    for k in range(0, n-1):
        if k > 0:
            fact = fact * (xsym - xp[k])
        p += coef[k] * fact
    return p


xpoints = np.array([2.1, 4.6, 5.25, 7.82, 9.2, 10.6])
ypoints = np.array([7.3, 7.0, 6.0, 5.1, 3.5, 5.0])
xpoints_final = np.linspace(2, 11, num=200)
x = sp.symbols('x')
dif_list, coef_1 = Divided_difference(xpoints, ypoints)
polyn_form = Newton_Divided_difference(ypoints, xpoints, dif_list, x)
print("Divided Differences (first column - y values):")
for i in coef_1:
    print(i)
print("P5(x) = ")
sp.pprint(polyn_form)

# Problem 3.1 b

cubic_spline_natural = scp.interpolate.CubicSpline(xpoints, ypoints, bc_type='natural')
cubic_spline_clamped = scp.interpolate.CubicSpline(xpoints, ypoints, bc_type='clamped')
cubic_spline_not_a_knot = scp.interpolate.CubicSpline(xpoints, ypoints, bc_type='not-a-knot')

plt.plot(xpoints, ypoints, 'ro')
plt.title("Polynomial and Cubic Spline Interpolation of given set of Data Points")
plt.plot(xpoints_final, sp.lambdify(x, polyn_form)(xpoints_final), label=f"Interpolating Polynomial P5(x)")
plt.plot(xpoints_final, cubic_spline_natural(xpoints_final), label="Cubic spline on Natural Boundary Conditions")
plt.plot(xpoints_final, cubic_spline_clamped(xpoints_final), label="Cubic spline on Clamped Boundary Conditions")
plt.plot(xpoints_final, cubic_spline_not_a_knot(xpoints_final), label="Cubic spline on Not-a-Knot Boundary Conditions")
plt.legend(loc='upper center', shadow=True)
plt.grid('both')
plt.show()
