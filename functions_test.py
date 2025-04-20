import numpy as np

def ackley(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

def bukin_function(x):
    x1 = x[0]  
    x2 = x[1]  
    return 100 * np.sqrt(abs(x2 - 0.01 * x1**2)) + 0.01 * abs(x1 + 10)

def cross_in_tray(x):
    x1 = x[0]  
    x2 = x[1]  
    term1 = np.sin(x1) * np.sin(x2)
    term2 = np.exp(abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    return -0.0001 * (abs(term1 * term2) + 1) ** 0.1

def drop_wave(x):
    x1 = x[0]  
    x2 = x[1]  
    numerator = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    denominator = 0.5 * (x1**2 + x2**2) + 1
    return -numerator / denominator

def griewank(x):
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1

def langermann(x):
    m = 5 
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])  
    C = np.array([1, 2, 5, 2, 3])
    x, y = x[:2]
    total = 0
    for i in range(m):
        xi, yi = A[i]
        ci = C[i]
        term1 = np.exp(-((x - xi)**2 + (y - yi)**2) / np.pi)
        term2 = np.cos(np.pi * ((x - xi)**2 + (y - yi)**2))
        total += ci * term1 * term2
    return -total  

def levy(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def rastrigin_function(x): 
    A = 10
    n = len(x)
    return A * n + sum((xi**2 - A * np.cos(2 * np.pi * xi) for xi in x))

def chaffer_n2(x):
    x1 = x[0]  
    x2 = x[1]  
    numerator = np.sin(x1**2 - x2**2)**2 - 0.5
    denominator = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + numerator / denominator

def schaffer_n4(x):
    x, y = x[:2]
    return 0.5 + (np.sin(x**2 + y**2)**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2

def shubert(x):
    x, y = x[:2]
    sum_x = np.sum([i * np.cos((i + 1) * x + i) for i in range(1, 6)])
    sum_y = np.sum([i * np.cos((i + 1) * y + i) for i in range(1, 6)])
    return sum_x * sum_y

def bohachevsky(x):
    x, y = x[:2]
    return (x**2 + y**2 - 0.3 * np.cos(3 * np.pi * x) - 0.4 * np.cos(4 * np.pi * y) + 0.7)

def perm0_function(x):
    n = len(x)
    total = 0
    for i in range(1, n + 1):
        inner_sum = sum((x[j - 1]**i / j**i - 1) for j in range(1, n + 1))
        total += inner_sum**2
    return total

def sphere(x):
    return np.sum(x**2)

def sum_of_different_powers(x):
    return np.sum(np.abs(x) ** np.arange(1, len(x) + 1))

def sum_squares(x):
    return np.sum((np.arange(1, len(x) + 1)) * (x ** 2))

def booth(x):
    x, y = x[:2]
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def matyas(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def zakharov(x):
    term1 = np.sum(x ** 2)
    term2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
    term3 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
    return term1 + term2 + term3

def three_hump_camel(x):
    x, y = x[:2]
    return 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y**2
