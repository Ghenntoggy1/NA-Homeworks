def seidel(a, x, b):
    n = len(a)
    for j in range(0, n):
        d = b[j]
        for i in range(0, n):
            if (j != i):
                d -= a[j][i] * x[i]
        x[j] = d / a[j][j]
    return x


n = 3
a = []
b = []

x = [1, 1, 1]
a1 = [[8, 3, 2], [16, 6, 4.001], [4, 1.501, 1]]
b1 = [20.00, 40.02, 10.01]
print(x)

# loop run for m times depending on m the error value
for i in range(0, 100):
    x = seidel(a1, x, b1)
    # print each time the updated solution
    print(x)
print("\nVerification:")
print(8*x[0] + 3*x[1] + 2*x[2])
print(16*x[0] + 6*x[1] + 4.001*x[2])
print(4*x[0] + 1.501*x[1] + 1*x[2])