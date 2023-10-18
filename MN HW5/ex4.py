# Problem 5.4:
import sympy as sp
import numpy as np
from prettytable import PrettyTable


# a) Direct Method (Gaussian Elimination):
def matrix_representation(system, syms):
    a, b = sp.linear_eq_to_matrix(system, syms)
    return np.asarray(a.col_insert(len(syms), b), dtype=np.float32)


def upper_triangular(M):
    M = np.concatenate((M[np.any(M != 0, axis=1)], M[np.all(M == 0, axis=1)]), axis=0)
    for i in range(0, M.shape[0]):
        j = 1
        pivot = M[i][i]
        while pivot == 0 and i + j < M.shape[0]:
            M[[i, i + j]] = M[[i + j, i]]
            j += 1
            pivot = M[i][i]

        if pivot == 0:
            return M

        row = M[i]
        M[i] = row / pivot
        for j in range(i + 1, M.shape[0]):
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * M[j][i]
    return M


def backsubstitution(M, syms):
    for i, row in reversed(list(enumerate(M))):
        eqn = -M[i][-1]
        for j in range(len(syms)):
            eqn += syms[j] * row[j]
        syms[i] = sp.solve(eqn, syms[i])[0]
    return syms


def validate_solution(system, solutions, tolerance=1e-6):
    for eqn in system:
        assert eqn.subs(solutions) < tolerance


def linalg_solve(system, syms):
    M, c = sp.linear_eq_to_matrix(system, syms)
    M, c = np.asarray(M, dtype=np.float32), np.asarray(c, dtype=np.float32)
    return np.linalg.solve(M, c)


if __name__ == '__main__':
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10')
    symbolic_vars = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

    equations = [x2 + 5 * x3 - 7 * x4 + 23 * x5 - x6 + 7 * x7 + 8 * x8 + x9 - 5 * x10 - 10,
                 17 * x1 - 24 * x3 - 75 * x4 + 100 * x5 - 18 * x6 + 10 * x7 - 8 * x8 + 9 * x9 - 50 * x10 + 40,
                 3 * x1 - 2 * x2 + 15 * x3 - 78 * x5 - 90 * x6 - 70 * x7 + 18 * x8 - 75 * x9 + x10 + 17,
                 5 * x1 + 5 * x2 - 10 * x3 - 72 * x5 - x6 + 80 * x7 - 3 * x8 + 10 * x9 - 18 * x10 - 43,
                 100 * x1 - 4 * x2 - 75 * x3 - 8 * x4 + 83 * x6 - 10 * x7 - 75 * x8 + 3 * x9 - 8 * x10 + 53,
                 70 * x1 + 85 * x2 - 4 * x3 - 9 * x4 + 2 * x5 + 3 * x7 - 17 * x8 - x9 - 21 * x10 - 12,
                 x1 + 15 * x2 + 100 * x3 - 4 * x4 - 23 * x5 + 13 * x6 + 7 * x8 - 3 * x9 + 17 * x10 + 60,
                 16 * x1 + 2 * x2 - 7 * x3 + 89 * x4 - 17 * x5 + 11 * x6 - 73 * x7 - 8 * x9 - 23 * x10 - 100,
                 51 * x1 + 47 * x2 - 3 * x3 + 5 * x4 - 10 * x5 + 18 * x6 - 99 * x7 - 18 * x8 + 12 * x10 - 0,
                 x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 - 100
                 ]
    print("Equations:")
    [print(eqn) for eqn in equations]
    augmented_matrix = matrix_representation(system=equations, syms=symbolic_vars)
    print('\naugmented matrix:\n', augmented_matrix)

    upper_triangular_matrix = upper_triangular(augmented_matrix)
    print('\nupper triangular matrix:\n')

    p = PrettyTable()
    for row in upper_triangular_matrix:
        p.add_row(row)
    print(p.get_string(header=False))

    backsub_matrix = upper_triangular_matrix[np.any(upper_triangular_matrix != 0, axis=1)]

    numeric_solution = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    if backsub_matrix.shape[0] != len(symbolic_vars):
        print('dependent system. infinite number of solutions')
    elif not np.any(backsub_matrix[-1][:len(symbolic_vars)]):
        print('inconsistent system. no solution..')
    else:
        numeric_solution = backsubstitution(backsub_matrix, symbolic_vars)

    a = 0
    inx = 0
    for i in numeric_solution:
        inx += 1
        if i == numeric_solution[-1]:
            print(f"x{inx} = {i}")
            continue
        else:
            print(f"x{inx} = {i}")
            a += i

    print("Verification:")
    print("x2 + 5x3 - 7x4 + 23x5 - x6 + 7x7 + 8x8 + x9 - 5x10 =", 0 * numeric_solution[0] + numeric_solution[1] +
          5 * numeric_solution[2] - 7 * numeric_solution[3] + 23 * numeric_solution[4] -
          numeric_solution[5] + 7 * numeric_solution[6] + 8 * numeric_solution[7] +
          numeric_solution[8] - 5 * numeric_solution[9])
    print("x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 =", a)

# b) Iterative Method (Gauss - Seidel Method):
print("===================================================================\nGauss-Jordan Method:")
def seidel(a, x, b):
    # Finding length of a(3)
    n = len(a)
    # for loop for 3 times as to calculate x, y , z
    for j in range(0, n):
        # temp variable d to store b[j]
        d = b[j]

        # to calculate respective xi, yi, zi
        for i in range(0, n):
            if j != i:
                d -= a[j][i] * x[i]
        # updating the value of our solution
        x[j] = d / a[j][j]
    # returning our updated solution
    return x


n = 10

# initial solution depending on n(here n=3)
x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
a1 = [
    [100, -4, -75, -8, 0, 83, -10, -75, 3, -8],
    [0, 1, 5, -7, 23, -1, 7, 8, 1, -5],
    [1, 15, 100, -4, -23, 13, 0, 7, -3, 17],
    [17, 0, -24, -75, 100, -18, 10, -8, 9, -50],
    [70, 85, -4, -9, 2, 0, 3, -17, -1, -21],
    [5, 5, -10, 0, -72, -1, 80, -3, 10, -18],
    [3, -2, 15, 0, -78, -90, -70, 18, -75, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [16, 2, -7, 89, -17, 11, -73, 0, -8, -23],
    [51, 47, -3, 5, -10, 18, -99, -18, 0, 12]
]
b1 = [10, -40, -17, 43, -53, 12, -60, 100, 0, 100]
print(x)

# loop run for m times depending on m the error value
for i in range(0, 25):
    x = seidel(a1, x, b1)
    # print each time the updated solution
    print(x)

a = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9]
print("Verification:")
print("x2 + 5x3 - 7x4 + 23x5 - x6 + 7x7 + 8x8 + x9 - 5x10 =", 0 * x[0] + x[1] +
      5 * x[2] - 7 * x[3] + 23 * x[4] -
      x[5] + 7 * x[6] + 8 * x[7] +
      x[8] - 5 * x[9])
print("x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 =", a)