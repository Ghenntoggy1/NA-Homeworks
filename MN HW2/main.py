import numpy as np
from scipy.special import erf
from scipy.interpolate import approximate_taylor_polynomial
import math
import matplotlib.pyplot as plt

x = np.linspace(-3.0, 3.0, num=100)
plt.plot(x, erf(x), label='original error')
for degree in np.arange(1, 20, step=1):
    erf_taylor = approximate_taylor_polynomial(erf, 0, degree, 1, order=degree + 1)
    t = approximate_taylor_polynomial(erf, 0, degree - 1, 1, order=degree)
    plt.plot(x, erf_taylor(x), label=f"degree{degree}")

    if abs(erf_taylor(1) - t(1)) < 10**(-6):
        print(erf_taylor(1) - t(1))
        print(erf_taylor)
        break
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0, shadow=True)

plt.tight_layout()
plt.xlim([-3, 3])
plt.show()