# Problem 3.2 a
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
    return difference


def Newton_Divided_difference(xp, yp, coef, xsym):
    n = len(xp)
    p = yp[0]
    fact = xsym - xp[0]
    for k in range(0, n-1):
        if k > 0:
            fact = fact * (xsym - xp[k])
        p += coef[k] * fact
    return p


npoints = np.array([1, 2, 3, 4, 5])
gamma_n_points = np.array([1, 1, 2, 6, 24])

gamma_func_orig = scp.special.gamma
xpoints_func = np.linspace(1, 5, num=200)
x = sp.symbols('x')
dif_list = Divided_difference(npoints, gamma_n_points)
polyn_form = Newton_Divided_difference(npoints, gamma_n_points, dif_list, x)

print("P4(x) = ")
sp.pprint(polyn_form)

# Problem 3.2 b, c, d
cubic_spline_natural = scp.interpolate.CubicSpline(npoints, gamma_n_points, bc_type='natural')

plt.plot(npoints, gamma_n_points, 'ro')
plt.plot(xpoints_func, sp.lambdify(x, polyn_form)(xpoints_func), label="P4(x)", linewidth=1)
plt.plot(xpoints_func, gamma_func_orig(xpoints_func), label="Original Gamma Function", linewidth=1)
plt.plot(xpoints_func, cubic_spline_natural(xpoints_func), linewidth=1, label="S(x)")
gamma_n_points = np.array([np.log(i) for i in gamma_n_points])

dif_list = Divided_difference(npoints, gamma_n_points)
polyn_form2 = Newton_Divided_difference(npoints, gamma_n_points, dif_list, x)

print("q4(x) = ")
polyn_form_func2 = sp.exp(polyn_form2)
print(polyn_form_func2)
plt.plot(xpoints_func, sp.lambdify(x, polyn_form_func2)(xpoints_func), linewidth=1, label="q(x)")

plt.legend(loc='upper center', shadow=False)
plt.grid('both')
plt.title("Polynomial and Cubic Spline Interpolation of given set of Data Points")
plt.show()

# Problem 3.2 e
max1 = max(abs(gamma_func_orig(xpoints_func) - sp.lambdify(x, polyn_form)(xpoints_func)))
print(f"Maximum difference between Gamma Func and P4(x) on [1, 5] is = {max1}")
max3 = max(abs(gamma_func_orig(xpoints_func) - cubic_spline_natural(xpoints_func)))
print(f"Maximum difference between Gamma Func and S(x) on [1, 5] is = {max3}")
max2 = max(abs(gamma_func_orig(xpoints_func) - sp.lambdify(x, polyn_form_func2)(xpoints_func)))
print(f"Maximum difference between Gamma Func and q(x) on [1, 5] is = {max2}")

print(f"More Accurate estimation on [1, 5] has maximum difference of {min(max1, max2, max3)} and corresponds to "
      f"{'Natural Cubic Spline function' if min(max1, max2, max3) == max3 else 'P4(x)' if min(max1, max2, max3) == max1 else 'q(x)'}")