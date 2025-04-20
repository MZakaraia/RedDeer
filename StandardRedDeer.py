import numpy as np

class StandardRedDeerOptimization:
    def __init__(self, objective_func, dimensions=2, bounds=(-5, 5),
                 population_size=50, num_males=10, 
                 max_iter=100, verbose=True):
        self.objective_func = objective_func
        self.dimensions = dimensions
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.num_males = num_males
        self.max_iter = max_iter
        self.verbose = verbose
        self.convergenceCurve = []

        # Validate inputs
        assert self.num_males < self.pop_size, "Number of males must be less than population size"

    def optimize(self):
        # Initialize population
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], 
                                          (self.pop_size, self.dimensions))
        self.fitness = np.apply_along_axis(self.objective_func, 1, self.population)

        # Track best solution
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]

        for iteration in range(self.max_iter):
            # Sort population based on fitness
            sorted_idx = np.argsort(self.fitness)
            self.population, self.fitness = self.population[sorted_idx], self.fitness[sorted_idx]

            # Divide population into males and females
            males, females = np.split(self.population, [self.num_males])

            # Male roaring process
            noise = (np.random.rand(self.num_males, self.dimensions) - 0.5)
            new_positions = np.clip(males + noise, self.bounds[0], self.bounds[1])
            new_fitness = np.apply_along_axis(self.objective_func, 1, new_positions)

            # Update males with better positions
            improvement_mask = new_fitness < self.fitness[:self.num_males]
            males[improvement_mask] = new_positions[improvement_mask]
            self.fitness[:self.num_males][improvement_mask] = new_fitness[improvement_mask]

            # Form harems
            harems = np.array_split(females, self._calculate_harem_sizes(len(males), len(females)))

            # Mating Process
            offspring = []
            for male, harem in zip(males, harems):
                for female in harem:
                    child = self._mutate(self._crossover(male, female))
                    offspring.append(child)

            # Convert offspring to array and evaluate fitness
            if offspring:
                offspring = np.array(offspring)
                offspring_fitness = np.apply_along_axis(self.objective_func, 1, offspring)

                # Merge and select next generation
                combined_pop = np.vstack((self.population, offspring))
                combined_fitness = np.hstack((self.fitness, offspring_fitness))
                best_indices = np.argsort(combined_fitness)[:self.pop_size]
                self.population, self.fitness = combined_pop[best_indices], combined_fitness[best_indices]

                # Update global best
                if self.fitness[0] < self.best_fitness:
                    self.best_solution, self.best_fitness = self.population[0], self.fitness[0]
            
            self.convergenceCurve.append(self.best_fitness)

        return self.best_solution, self.best_fitness, np.array(self.convergenceCurve)

    def _calculate_harem_sizes(self, num_males, num_females):
        proportions = np.random.dirichlet(np.ones(num_males))
        sizes = (proportions * num_females).astype(int)
        sizes[:num_females - sizes.sum()] += 1  # Distribute remaining females
        return sizes

    def _crossover(self, parent1, parent2):
        return parent1 * np.random.rand(self.dimensions) + parent2 * (1 - np.random.rand(self.dimensions))

    def _mutate(self, individual):
        mask = np.random.rand(self.dimensions) < 0.1
        noise = np.random.normal(0, 0.1, self.dimensions)
        return np.clip(individual + mask * noise, self.bounds[0], self.bounds[1])

# Example usage
# if __name__ == "__main__":
#     def sphere(x):
#         return np.sum(x**2)

#     rdo = StandardRedDeerOptimization(sphere, dimensions=30, bounds=(-5.12, 5.12), 
#                                     population_size=50, num_males=15, max_iter=100)
#     best_solution, best_fitness, _ = rdo.optimize()

#     print("\nOptimization Results:")
#     print(f"Best Solution: {best_solution}")
#     print(f"Best Fitness: {best_fitness:.6f}")