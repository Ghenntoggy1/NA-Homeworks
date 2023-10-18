# Problem 5.5
from prettytable import PrettyTable

# a) Gauss-Jordan Method:
def GaussJordanMethod(a, n):
    flag = 0
    for i in range(n):
        if a[i][i] == 0:
            c = 1
            while (i + c) < n and a[i + c][i] == 0:
                c += 1
            if (i + c) == n:
                flag = 1
                break
            j = i
            for k in range(1 + n):
                temp = a[j][k]
                a[j][k] = a[j + c][k]
                a[j + c][k] = temp
        for j in range(n):
            if i != j:
                p = a[j][i] / a[i][i]
                for k in range(n + 1):
                    a[j][k] = a[j][k] - (a[i][k]) * p
    return flag


def PrintResult(a, n, flag):
    print("Result: ")
    if (flag == 2):
        print("Infinite Solutions Exists<br>")
    elif (flag == 3):
        print("No Solution Exists<br>")
    else:
        for i in range(n):
            sol = a[i][n] / a[i][i]
            sols.append(sol)
            print(sol, end=" ")


def CheckConsistency(a, n, flag):
    # flag == 2 for infinite solution
    # flag == 3 for No solution
    flag = 3
    for i in range(n):
        sum = 0
        for j in range(n):
            sum = sum + a[i][j]
        if sum == a[i][j]:
            flag = 2

    return flag


a = [[8, 3, 2, 20.00], [16, 6, 4.001, 40.02], [4, 1.501, 1, 10.01]]
n = 3
flag = 0
sols = []
flag = GaussJordanMethod(a, n)

if flag == 1:
    flag = CheckConsistency(a, n, flag)

print("Final Augmented Matrix:")
p = PrettyTable()
for row in a:
    p.add_row(row)
print(p.get_string(header=False))

PrintResult(a, n, flag)
print("\nVerification:")
print("8x + 3y + 2z =", 8*sols[0] + 3*sols[1] + 2*sols[2])
print("16x + 6y + 4.001z =", 16*sols[0] + 6*sols[1] + 4.001*sols[2])
print("4x + 1.501y + z =", 4*sols[0] + 1.501*sols[1] + 1*sols[2])
print("=====================================================")
print("Error 1 equation =", 20.00 - (8*sols[0] + 3*sols[1] + 2*sols[2]))
print("Error 2 equation =", 40.02 - (16*sols[0] + 6*sols[1] + 4.001*sols[2]))
print("Error 3 equation =", 10.01 - (4*sols[0] + 1.501*sols[1] + 1*sols[2]))

# b) Gauss-Seidel Method:
def f1(y, z):
    return (20.00 - 3*y - 2*z)/8

def f2(x, z):
    return (40.02 - 16*x - 4.001*z)/6

def f3(x, y):
    return (10.01 - 4*x - 1.501*y)/1


x0 = 1
y0 = 1
z0 = 1
count = 1

e = float(input('Enter tolerable error: '))
print('\nIteration\tx\ty\tz\n')

condition = True
e1 = 1
e2 = 1
e3 = 1
while e1 > e and e2 > e and e3 > e:
    x1 = f1(y0, z0)
    y1 = f2(x1, z0)
    z1 = f3(x1, y1)
    print('Iteration = %d\tx = %0.4f\ty = %0.4f\tz = %0.4f\n' % (count, x1, y1, z1))
    e1 = abs(x0 - x1)
    e2 = abs(y0 - y1)
    e3 = abs(z0 - z1)
    count += 1
    x0 = x1
    y0 = y1
    z0 = z1

print('Solution: \nx = %0.3f, \ny = %0.3f, \nz = %0.3f' % (x1, y1, z1))
print("\nVerification:")
print("8x + 3y + 2z =", 8*x1 + 3*y1 + 2*z1)
print("16x + 6y + 4.001z =", 16*x1 + 6*y1 + 4.001*z1)
print("4x + 1.501y + z =", 4*x1 + 1.501*y1 + 1*z1)
print("=====================================================")
print("Error 1 equation =", 20.00 - (8*x1 + 3*y1 + 2*z1))
print("Error 2 equation =", 40.02 - (16*x1 + 6*y1 + 4.001*z1))
print("Error 3 equation =", 10.01 - (4*x1 + 1.501*y1 + 1*z1))

