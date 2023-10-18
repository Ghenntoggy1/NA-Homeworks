# Problem 2.1
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.symbols('x')
# Getting an interval between -3 and 3 split in 500 points:
xpoints = np.linspace(-3, 3, num=500)
func = 2/sp.sqrt(sp.pi)

# Getting first error between polynomial with 1 term and 2 terms:
d1 = x
d2 = (x - sp.Pow(x, 3)/3)
erftayinit = func * d1
erftaylast = func * d2
orig = sp.erf
erftayinit_func = sp.lambdify(x, erftayinit, modules=['numpy'])
erftaylast_func = sp.lambdify(x, erftaylast, modules=['numpy'])

a = abs(orig(3) - erftaylast_func(3))
init = 5
term = 2
denom = 2
d3 = d2
# Finding n, in other words, number of terms in Taylor Polynomial, which is 14:
while a > 0.000001:
    d3 = d3 + (sp.Pow(x, init)) / (init * sp.factorial(denom))
    erftayinit = func * d3
    init += 2
    print(erftayinit)
    d3 = d3 - sp.Pow(x, init)/(init * sp.factorial(denom + 1))
    erftaylast = func * d3
    init += 2
    print(erftaylast)
    denom += 2
    term += 1
    term += 1
    erftayinit_func = sp.lambdify(x, erftayinit, modules=['numpy'])
    erftaylast_func = sp.lambdify(x, erftaylast, modules=['numpy'])
    a = abs(orig(3) - erftaylast_func(3))

print("Error function with 32 terms")
print(erftaylast)
plt.axis([-3, 3, -1, 1])
plt.title("Approximation of error function using Taylor series")
plt.plot(xpoints, erftaylast_func(xpoints), label=f"Polynomial with: {term} terms")
plt.legend(loc="upper right")

plt.grid('both')
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
