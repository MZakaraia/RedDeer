import numpy as np

class RedDeerOptimization:
    def __init__(self, objective_func, dimensions=2, bounds=(-5, 5),
                 population_size=50, num_males=10, roosting_males=0.5,
                 max_iter=100, verbose=True):
        self.objective_func = objective_func
        self.dimensions = dimensions
        self.bounds = bounds
        self.pop_size = population_size
        self.num_males = num_males
        self.roosting_males = int(num_males * roosting_males)
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Validate inputs
        assert self.num_males < self.pop_size, "Number of males must be less than population size"
        assert self.roosting_males < self.num_males, "Roosting males must be less than total males"

    def optimize(self):
        # Initialize population
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], 
                                          (self.pop_size, self.dimensions))
        self.fitness = np.array([self.objective_func(ind) for ind in self.population])
        
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
        
        for iteration in range(self.max_iter):
            # Sort population by fitness
            sorted_idx = np.argsort(self.fitness)
            self.population = self.population[sorted_idx]
            self.fitness = self.fitness[sorted_idx]
            
            # Divide into males and females
            males = self.population[:self.num_males]
            females = self.population[self.num_males:]
            
            # Divide males into commanders and stags
            commanders = males[:self.roosting_males]
            stags = males[self.roosting_males:]
            
            # Male roaring process
            for i in range(len(commanders)):
                alpha = np.random.rand()
                new_pos = commanders[i] + alpha * (np.random.rand(self.dimensions) - 0.5)
                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                new_fitness = self.objective_func(new_pos)
                
                if new_fitness < self.fitness[i]:
                    commanders[i] = new_pos
                    self.fitness[i] = new_fitness
            


            # Form harems
            harem_sizes = self._calculate_harem_sizes(len(commanders), len(females))
            harems = []
            start = 0
            for size in harem_sizes:
                harems.append(females[start:start+size])
                start += size
            
            # Convex Combinition Solution
            RndCommander = commanders[np.random.choice(len(commanders), 1, replace=False)[0]]
            RndStag = stags[np.random.choice(len(stags), 1, replace=False)[0]]
            female1, female2 = females[np.random.choice(len(harems), 2, replace=False)]
            RndSol = np.random.uniform(self.bounds[0], self.bounds[1],self.dimensions)
            ConvexCombinationSolution = self.ConvexCombination(RndCommander, RndStag, female1, female2, RndSol)
            ConvexObjValue = self.objective_func(ConvexCombinationSolution)
            if ConvexObjValue < self.best_fitness:
                self.best_fitness = ConvexObjValue
                self.best_solution = ConvexCombinationSolution


            
            # Mating process
            offspring = []
            for commander, harem in zip(commanders, harems):
                # Mate with harem
                for female in harem:
                    child = self._crossover(commander, female)
                    child = self._mutate(child)
                    offspring.append(child)
                
                # Mate with random stag
                if stags.size > 0:
                    stag = stags[np.random.randint(0, len(stags))]
                    child = self._crossover(commander, stag)
                    child = self._mutate(child)
                    offspring.append(child)
            
            # Create new generation
            offspring = np.array(offspring)
            offspring_fitness = np.array([self.objective_func(ind) for ind in offspring])
            
            combined_pop = np.vstack((self.population, offspring))
            combined_fitness = np.hstack((self.fitness, offspring_fitness))
            
            # Select fittest individuals
            sorted_idx = np.argsort(combined_fitness)[:self.pop_size]
            self.population = combined_pop[sorted_idx]
            self.fitness = combined_fitness[sorted_idx]
            
            # Update best solution
            if self.fitness[0] < self.best_fitness:
                self.best_solution = self.population[0]
                self.best_fitness = self.fitness[0]
            
            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter-1):
                print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_fitness:.6f}")
                
        return self.best_solution, self.best_fitness

    def _calculate_harem_sizes(self, num_commanders, num_females):
        proportions = np.random.dirichlet(np.ones(num_commanders))
        sizes = (proportions * num_females).astype(int)
        
        # Distribute remaining females
        remainder = num_females - sum(sizes)
        for i in range(remainder):
            sizes[i % num_commanders] += 1
            
        return sizes

    def _crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dimensions)
        return parent1 * alpha + parent2 * (1 - alpha)
    # Convex Combination Function
    def ConvexCombination(self, *arg):        
        Alpha = np.array([1/self.objective_func(i) for i in arg])
        Alpha = Alpha/Alpha.sum()
        return np.array([i * j for i in arg for j in Alpha]).sum(axis=1)

    def _mutate(self, individual):
        mutation_prob = 0.1
        mask = np.random.rand(self.dimensions) < mutation_prob
        noise = np.random.normal(0, 0.1, self.dimensions)
        new_ind = individual + mask * noise
        return np.clip(new_ind, self.bounds[0], self.bounds[1])
    

# Example usage
if __name__ == "__main__":
    # Define objective function (Sphere function)
    def sphere(x):
        return np.sum(x**2)
    
    def ConvexCombination(*arg):        
        Alpha = np.array([self.objective_func(i) for i in arg])
        Alpha = Alpha/Alpha.sum()
        return np.array([i * j for i in arg for j in Alpha]).sum(axis=1)
    
    # Initialize optimizer
    rdo = RedDeerOptimization(sphere, 
                            dimensions=30,
                            bounds=(-5.12, 5.12),
                            population_size=50,
                            num_males=15,
                            roosting_males=0.4,
                            max_iter=100)
    
    # Run optimization
    best_solution, best_fitness = rdo.optimize()
    
    print("\nOptimization Results:")
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")