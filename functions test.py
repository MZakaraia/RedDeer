from red_deer_optimization import *


def ackley(x):
  
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e
fn = lambda x: ackley(x)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=30,
                          bounds=(-32.768, 32.768),  # Bounds for both dimensions
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
#####################################################1
def bukin_function(x):
    # Ensure x has at least 30 dimensions
    
    
    x1 = x[0]  # First dimension
    x2 = x[1]  # Second dimension
    
    # Calculate the Bukin Function No. 6
    return 100 * np.sqrt(abs(x2 - 0.01 * x1**2)) + 0.01 * abs(x1 + 10)
fn = lambda xy: bukin_function(xy)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-15, 3),  # Bounds for both dimensions
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")

#################################################################2
def cross_in_tray(x):
   
    # Ensure x has at least 30 dimensions
   
    x1 = x[0]  # First dimension
    x2 = x[1]  # Second dimension
    
    term1 = np.sin(x1) * np.sin(x2)
    term2 = np.exp(abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    
    return -0.0001 * (abs(term1 * term2) + 1) ** 0.1
fn = lambda xy:cross_in_tray(xy)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-10, 10),  # Bounds for both dimensions
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")

##########################################################3
def drop_wave(x):
    
    
    x1 = x[0]  # First dimension
    x2 = x[1]  # Second dimension
    
    numerator = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    denominator = 0.5 * (x1**2 + x2**2) + 1
    
    return -numerator / denominator
fn = lambda xy:drop_wave(xy)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-5.12, 5.12),  # Bounds for both dimensions
                          population_size=50,
                          num_males=15,
                          roosting_males=0.4,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")

########################################################4
def griewank(x):
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1

fn = lambda x: griewank(x)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=30,
                          bounds=(-600, 600),  # Bounds for both dimensions
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
##########################################################5
def langermann(xy):
    m = 5 
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])  
    C = np.array([1, 2, 5, 2, 3])
    x, y = xy
    total = 0
    for i in range(m):
        xi, yi = A[i]
        ci = C[i]
        term1 = np.exp(-((x - xi)**2 + (y - yi)**2) / np.pi)
        term2 = np.cos(np.pi * ((x - xi)**2 + (y - yi)**2))
        total += ci * term1 * term2
    return -total  

fn = lambda xy:langermann(xy)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(0, 10),  # Bounds for both dimensions
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")

##########################################################6
def levy(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3
fn = lambda x:levy(x)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                        dimensions=30,
                        bounds=  (-10, 10),
                        population_size=50,
                        num_males=15,
                        roosting_males=0.5,
                        max_iter=1000)
best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
best_solution, best_fitness = rdo.optimize()


######################################################7
def rastrigin_function(x): 
    A = 10
    n = len(x)
    return A * n + sum((xi**2 - A * np.cos(2 * np.pi * xi) for xi in x))

fn = lambda x: rastrigin_function(x)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                        dimensions=30,
                        bounds=  (-5.12, 5.12),
                        population_size=50,
                        num_males=15,
                        roosting_males=0.5,
                        max_iter=1000)
best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
best_solution, best_fitness = rdo.optimize()
####################################################8

def chaffer_n2(x):
   
    
    x1 = x[0]  # First dimension
    x2 = x[1]  # Second dimension
    
    numerator = np.sin(x1**2 - x2**2)**2 - 0.5
    denominator = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + numerator / denominator
fn = lambda xy:chaffer_n2(xy)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-100, 100),  # Bounds for both dimensions
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")



################################################9
def schaffer_n4(xy):
    """Schaffer N. 4 function definition."""
    x, y = xy
    return 0.5 + (np.sin(x**2 + y**2)**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2

fn = lambda xy: schaffer_n4(xy)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-100, 100),  # Bounds for both dimensions
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")


####################################################10

def shubert(xy):
    x, y = xy
    sum_x = np.sum([i * np.cos((i + 1) * x + i) for i in range(1, 6)])
    sum_y = np.sum([i * np.cos((i + 1) * y + i) for i in range(1, 6)])
    return sum_x * sum_y

fn = lambda xy: shubert(xy)  # Corrected to match the function signature

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-5.12, 5.12),  # Changed to a list of tuples
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")  # Added best_fitness for completeness
print(f"Best Fitness: {best_fitness}")
#######################################################11
def Bohachevsky(xy):
    """Bohachevsky function definition."""
    x, y = xy  # Unpack the xy tuple
    return (x**2 + y**2 
            - 0.3 * np.cos(3 * np.pi * x) 
            - 0.4 * np.cos(4 * np.pi * y) + 0.7)

fn = lambda xy: Bohachevsky(xy)  # Corrected to match the function signature

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-100, 100),  # Changed to a list of tuples
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")  # Added best_fitness for completeness
print(f"Best Fitness: {best_fitness}")
######################################################12
 def perm0_function(x):
    n = len(x)
    total = 0
    for i in range(1, n + 1):
        inner_sum = sum((x[j - 1]**i / j**i - 1) for j in range(1, n + 1))
        total += inner_sum**2
    return total
fn = lambda x: perm0_function(x)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                        dimensions=5,
                        bounds=  (-5, 5),
                        population_size=50,
                        num_males=15,
                        roosting_males=0.5,
                        max_iter=1000)
best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
############################################################13

def sphere(x):
    return np.sum(x**2)
fn = lambda x:sphere(x)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                        dimensions=30,
                        bounds=  [-5.12, 5.12],
                        population_size=50,
                        num_males=15,
                        roosting_males=0.5,
                        max_iter=1000)
best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")

#####################################################14

def sum_of_different_powers(x):
     return np.sum(np.abs(x) ** np.arange(1, len(x) + 1))
     fn = lambda x:sphere(x)

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                        dimensions=30,
                        bounds=  [-1, 1],
                        population_size=50,
                        num_males=15,
                        roosting_males=0.5,
                        max_iter=1000)
best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
#bounds = [(-1, 1)] 
###############################################15
def sum_squares(x):
    return np.sum((np.arange(1, len(x) + 1)) * (x ** 2))

fn = lambda x:sum_squares(x)  # Removed the unnecessary colon

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=30,
                          bounds=(-10, 10),  # Changed to a list of tuples
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
    ##############################################16
def booth(xy):
    x, y = xy
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

fn = lambda x:booth(x)  # Removed the unnecessary colon

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-10, 10),  # Changed to a list of tuples
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
    #########################################17
def matyas(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

fn = lambda x: matyas(x)  # Removed the unnecessary colon

# Initialize optimizer
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-10, 10),  # Changed to a list of tuples
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
    ###############################################18
def zakharov(x):
    term1 = np.sum(x ** 2)
    term2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
    term3 = np.sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
    return term1 + term2 + term3

fn = lambda x: zakharov(x)

# Initialize optimizer
# Ensure bounds are specified correctly for 2 dimensions
rdo = RedDeerOptimization(fn, 
                          dimensions=30,
                          bounds=(-5, 10),  # Changed to a list of tuples
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")

################################################################19
def three_hump_camel(xy):
    x, y = xy
    return 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y**2

fn = lambda xy: three_hump_camel(xy)

# Initialize optimizer
# Ensure bounds are specified correctly for 2 dimensions
rdo = RedDeerOptimization(fn, 
                          dimensions=2,
                          bounds=(-5, 5),  # Changed to a list of tuples
                          population_size=50,
                          num_males=15,
                          roosting_males=0.5,
                          max_iter=1000)

best_solution, best_fitness = rdo.optimize()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
#######################################################20









   
