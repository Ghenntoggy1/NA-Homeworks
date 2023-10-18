# Problem 2.2
import math
import numpy


def fibonacci(n, list):
    return numpy.double(list[n - 1]) + numpy.double(list[n - 2])


def r_term(n, list):
    return numpy.double(list[n + 1]) / numpy.double(list[n])


list_fib = [numpy.double(1), numpy.double(1)]
for i in range(2, 52):
    list_fib.append(numpy.double(fibonacci(i, list_fib)))

list_r_terms = []
for i in range(0, 51):
    list_r_terms.append(numpy.double(r_term(i, list_fib)))

golden_ratio = (1 + math.sqrt(5)) / 2

ofc = 0
err = []
for i in range(0, 50):
    print(f"R{i + 1} =", list_r_terms[i])
    print("Error =", numpy.double(abs(golden_ratio - list_r_terms[i])))
    err.append(numpy.double(abs(golden_ratio - list_r_terms[i])))

# Found a practical method in order to estimate order of convergence on internet.

c = 0
for i in range(1, len(err)):
    if err[i - 1] != numpy.double(0) and err[i] != numpy.double(0) and err[i + 1] != numpy.double(0):
        ofc += ((err[i + 1] / err[i]) / (err[i] / err[i - 1]))
        c += 1
print("========================================")
print("Order of convergence =", ofc / c)
print("========================================")
print('Order of Convergence (specific values R25 R24 R23) =', ((abs((list_r_terms[25] - golden_ratio) /
                                                                            (list_r_terms[24] - golden_ratio)))) /
                                                              ((abs((list_r_terms[24] - golden_ratio) /
                                                                            (list_r_terms[23] - golden_ratio)))))
# This rate of convergence corresponds to Linear Convergence. I used the method mentioned above, and did calculations
# for every 3 R ratios. After that, I got rid of cases where error is 0, in order not to get division by 0 and ln(0)
# and got the mean value of the rate, which is 1.0227256976477916. I also checked the rate for specific R terms, i.e.
# R25 R24 R23, which gave me order of convergence 1.0000067387452052, which is also Linear Convergence.
