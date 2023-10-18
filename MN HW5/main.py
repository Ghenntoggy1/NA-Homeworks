import sympy as sp
import numpy as np


# form augmented matrix
def matrix_representation(system, syms):
    # extract equation coefficients and constant
    a, b = sp.linear_eq_to_matrix(system, syms)

    # insert right hand size values into coefficients matrix
    return np.asarray(a.col_insert(len(syms), b), dtype=np.float32)


# write rows in row echelon form
def upper_triangular(M):
    # move all zeros to buttom of matrix
    M = np.concatenate((M[np.any(M != 0, axis=1)], M[np.all(M == 0, axis=1)]), axis=0)

    # iterate over matrix rows
    for i in range(0, M.shape[0]):

        # initialize row-swap iterator
        j = 1

        # select pivot value
        pivot = M[i][i]

        # find next non-zero leading coefficient
        while pivot == 0 and i + j < M.shape[0]:
            # perform row swap operation
            M[[i, i + j]] = M[[i + j, i]]

            # incrememnt row-swap iterator
            j += 1

            # get new pivot
            pivot = M[i][i]

        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # extract row
        row = M[i]

        # get 1 along the diagonal
        M[i] = row / pivot

        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * M[j][i]

    # return upper triangular matrix
    return M


def backsubstitution(M, syms):
    # symbolic variable index
    for i, row in reversed(list(enumerate(M))):
        # create symbolic equation
        eqn = -M[i][-1]
        for j in range(len(syms)):
            eqn += syms[j] * row[j]

        # solve symbolic expression and store variable
        syms[i] = sp.solve(eqn, syms[i])[0]

    # return list of evaluated variables
    return syms


def validate_solution(system, solutions, tolerance=1e-6):
    # iterate over each equation
    for eqn in system:
        # assert equation is solved
        assert eqn.subs(solutions) < tolerance


# solve system using numpy built in functions
def linalg_solve(system, syms):
    # convert list of equations to matrix form
    M, c = sp.linear_eq_to_matrix(system, syms)

    # form augmented matrix - convert sympy matrices to numpy arrays and concatenate
    M, c = np.asarray(M, dtype=np.float32), np.asarray(c, dtype=np.float32)

    # solve system of equations
    return np.linalg.solve(M, c)


if __name__ == '__main__':

    # symbolic variables
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10')
    symbolic_vars = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

    # define system of equations
    equations = [x2 + 5*x3 - 7*x4 + 23*x5 - x6 + 7*x7 + 8*x8 + x9 - 5*x10 - 10,
                 17*x1 - 24*x3 - 75*x4 + 100*x5 - 18*x6 + 10*x7 - 8*x8 + 9*x9 - 50*x10 + 40,
                 3*x1 - 2*x2 + 15*x3 - 78*x5 - 90*x6 - 70*x7 + 18*x8 - 75*x9 + x10 + 17,
                 5*x1 + 5*x2 - 10*x3 - 72*x5 - x6 + 80*x7 - 3*x8 + 10*x9 - 18*x10 - 43,
                 100*x1 - 4*x2 - 75*x3 - 8*x4 + 83*x6 - 10*x7 - 75*x8 + 3*x9 - 8*x10 + 53,
                 70*x1 + 85*x2 - 4*x3 - 9*x4 + 2*x5 + 3*x7 - 17*x8 - x9 - 21*x10 - 12,
                 x1 + 15*x2 + 100*x3 - 4*x4 - 23*x5 + 13*x6 + 7*x8 - 3*x9 + 17*x10 + 60,
                 16*x1 + 2*x2 - 7*x3 + 89*x4 - 17*x5 + 11*x6 - 73*x7 - 8*x9 - 23*x10 - 100,
                 51*x1 + 47*x2 - 3*x3 + 5*x4 - 10*x5 + 18*x6 - 99*x7 - 18*x8 + 12*x10 - 0,
                 x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 - 100
                 ]

    # display equations
    [print(eqn) for eqn in equations]

    # obtain augmented matrix representation
    augmented_matrix = matrix_representation(system=equations, syms=symbolic_vars)
    print('\naugmented matrix:\n', augmented_matrix)

    # generate upper triangular matrix form
    upper_triangular_matrix = upper_triangular(augmented_matrix)
    print('\nupper triangular matrix:\n', upper_triangular_matrix)

    # remove zero rows
    backsub_matrix = upper_triangular_matrix[np.any(upper_triangular_matrix != 0, axis=1)]

    # initialise numerical solution
    numeric_solution = np.array([0., 0., 0.])

    # assert that number of rows in matrix equals number of unknown variables
    if backsub_matrix.shape[0] != len(symbolic_vars):
        print('dependent system. infinite number of solutions')
    elif not np.any(backsub_matrix[-1][:len(symbolic_vars)]):
        print('inconsistent system. no solution..')
    else:
        # backsubstitute to solve for variables
        numeric_solution = backsubstitution(backsub_matrix, symbolic_vars)
        print(f'\nsolutions:\n{numeric_solution}')

    a = 0
    for i in numeric_solution:
        if i == numeric_solution[-1]:
            continue
        else:
            a += i
    print(a)

#
                 # x2 + 5*x3 - 7*x4 + 23*x5 - x6 + 7*x7 + 8*x8 + x9 - 5*x10 - 10,
                 # 17*x1 - 24*x3 - 75*x4 + 100*x5 - 18*x6 + 10*x7 - 8*x8 + 9*x9 - 50*x10 + 40,
                 # 3*x1 - 2*x2 + 15*x3 - 78*x5 - 90*x6 - 70*x7 + 18*x8 - 75*x9 + x10 + 17,
                 # 5*x1 + 5*x2 - 10*x3 - 72*x5 - x6 + 80*x7 - 3*x8 + 10*x9 - 18*x10 - 43,
                 # 100*x1 - 4*x2 - 75*x3 - 8*x4 + 83*x6 - 10*x7 - 75*x8 + 3*x9 - 8*x10 + 53,
                 # 70*x1 + 85*x2 - 4*x3 - 9*x4 + 2*x5 + 3*x7 - 17*x8 - x9 - 21*x10 - 12,
                 # x1 + 15*x2 + 100*x3 - 4*x4 - 23*x5 + 13*x6 + 7*x8 - 3*x9 + 17*x10 + 60,
                 # 16*x1 + 2*x2 - 7*x3 + 89*x4 - 17*x5 + 11*x6 - 73*x7 - 8*x9 - 23*x10 - 100,
                 # 51*x1 + 47*x2 - 3*x3 + 5*x4 - 10*x5 + 18*x6 - 99*x7 - 18*x8 + 12*x10 - 0,
                 # x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 - 100

# Python program to find a solution to a set of linear equations using the Gaussian Elimination method

# Creating a function to print the augmented matrix with the given set of linear equations
def print_aug(mat):
    no = len(mat)
    for i in range(0, no):
        l = ""
        for k in range(0, n + 1):
            l += str(mat[i][k]) + "\t"
            if j == no - 1:
                l += "| "
        print(l)
    print("")


# Creating a function to perform gaussian elimination on the given matrix mat
def gauss_elem(mat):
    num = len(mat)

    for i in range(0, num):
        # Searching the maximum value of a particular column
        max_el = abs(mat[i][i])
        # Row having the element of maximum value
        max_row = i
        for k in range(i + 1, num):
            if abs(mat[k][i]) > max_el:
                max_el = abs(mat[k][i])
                max_row = k

                # Swapping the maximum row with the current row
        for k in range(i, n + 1):
            temp = mat[max_row][k]
            mat[max_row][k] = mat[i][k]
            mat[i][k] = temp

            # Chaning the value of the rows below the current row to 0
        for k in range(i + 1, n):
            curr = -mat[k][i] / mat[i][i]
            for j in range(i, n + 1):
                if i == j:
                    mat[k][j] = 0
                else:
                    mat[k][j] += curr * mat[i][j]

                    # Solving the equation Ax = b for the created upper triangular matrix mat
    l = [0 for i in range(n)]
    for j in range(n - 1, -1, -1):
        l[j] = mat[j][n] / mat[j][j]
        for k in range(j - 1, -1, -1):
            mat[k][n] -= mat[k][j] * l[j]
    return l


if __name__ == "__main__":
    from fractions import Fraction
    import math
    n = int(input())

    A_mat = [[0 for j in range(n + 1)] for i in range(n)]

    # Reading the input coefficients of the linear equations
    for j in range(0, n):
        l = map(Fraction, input().split(" "))
        for i, elem in enumerate(l):
            A_mat[j][i] = elem

    l = input().split(" ")
    print(l)
    last = list(map(Fraction, l))
    for j in range(0, n):
        A_mat[j][n] = last[j]

        # Printing the augmented matrix from the input data
    print_aug(A_mat)

    # Calculating the solution of the matrix
    x = gauss_elem(A_mat)

    # Printing the result
    l = 0
    for j in range(0, n-1):
        print(float(x[j]))
        l += float(round(x[j], 3))
    print(l)

# Input:
# 10
# 0 1 5 -7 23 -1 7 8 1 -5
# 17 0 -24 -75 100 -18 10 -8 9 -50
# 3 -2 15 0 -78 -90 -70 18 -75 1
# 5 5 -10 0 -72 -1 80 -3 10 -18
# 100 -4 -75 -8 0 83 -10 -75 3 -8
# 70 85 -4 -9 2 0 3 -17 -1 -21
# 1 15 100 -4 -23 13 0 7 -3 17
# 16 2 -7 89 -17 11 -73 0 -8 -23
# 51 47 -3 5 -10 18 -99 -18 0 12
# 1 1 1 1 1 1 1 1 1 0
# 10 -40 -17 43 -53 12 -60 100 0 100
