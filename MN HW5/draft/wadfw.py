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


n = 3

# initial solution depending on n(here n=3)
x = [1, 1, 1]
a1 = [
    [8, 3, 2],
    [16, 6, 4.001],
    [4, 1.501, 1]
]
b1 = [20.00, 40.02, 10.01]
print(x)

# loop run for m times depending on m the error value
for i in range(0, 100000):
    x = seidel(a1, x, b1)
    # print each time the updated solution
    print(x)
x1 = x[0]
y1 = x[1]
z1 = x[2]
print("\nVerification:")
print("8x + 3y + 2z =", 8*x1 + 3*y1 + 2*z1)
print("16x + 6y + 4.001z =", 16*x1 + 6*y1 + 4.001*z1)
print("4x + 1.501y + z =", 4*x1 + 1.501*y1 + 1*z1)
print("=====================================================")
print("Error 1 equation =", 20.00 - (8*x1 + 3*y1 + 2*z1))
print("Error 2 equation =", 40.02 - (16*x1 + 6*y1 + 4.001*z1))
print("Error 3 equation =", 10.01 - (4*x1 + 1.501*y1 + 1*z1))