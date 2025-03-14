import numpy as np

class RedDeerOptimization:
    def __init__(self, objective_func, dimensions=2, bounds=(-5, 5),
                 population_size=50, num_males=10, roosting_males=0.5,
                 max_iter=100, verbose=True):
        self.objective_func = objective_func
        self.dimensions = dimensions
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.num_males = num_males
        self.roosting_males = max(1, int(num_males * roosting_males))  # Ensure at least 1
        self.max_iter = max_iter
        self.verbose = verbose
        self.convexCounter = 0

        # Validate inputs
        assert self.num_males < self.pop_size, "Number of males must be less than population size"
        assert self.roosting_males <= self.num_males, "Roosting males must be <= total males"

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
            learning_rate = 1 - (iteration / self.max_iter)

            # Sort population based on fitness
            sorted_idx = np.argsort(self.fitness)
            self.population, self.fitness = self.population[sorted_idx], self.fitness[sorted_idx]

            # Divide population into males and females
            males, females = np.split(self.population, [self.num_males])

            # Further divide males into commanders and stags
            commanders, stags = np.split(males, [self.roosting_males])

            # Male roaring process (improved vectorized update)
            noise = (1 - learning_rate) * (np.random.rand(self.roosting_males, self.dimensions) - 0.5)
            new_positions = np.clip(commanders + noise, self.bounds[0], self.bounds[1])
            new_fitness = np.apply_along_axis(self.objective_func, 1, new_positions)

            # Update commanders with better positions
            improvement_mask = new_fitness < self.fitness[:self.roosting_males]
            commanders[improvement_mask] = new_positions[improvement_mask]
            self.fitness[:self.roosting_males][improvement_mask] = new_fitness[improvement_mask]

            # Form harems
            harems = np.array_split(females, self._calculate_harem_sizes(len(commanders), len(females)))

            # Convex Combination Solution
            if len(females) >= 2:
                convex_solution = self.ConvexCombination(
                    commanders[0], stags[0] if stags.size > 0 else commanders[-1], females[0], females[1], self.best_solution
                )
                convex_solution = np.clip(convex_solution + learning_rate * (self.best_solution - convex_solution), self.bounds[0], self.bounds[1])
                convex_fitness = self.objective_func(convex_solution)

                if convex_fitness < self.best_fitness:
                    self.best_solution, self.best_fitness = convex_solution, convex_fitness
                    self.convexCounter += 1

            # Mating Process
            offspring = []
            for commander, harem in zip(commanders, harems):
                for female in harem:
                    child = self._mutate(self._crossover(commander, female))
                    offspring.append(child)

                if stags.size > 0:
                    stag = stags[np.random.randint(len(stags))]
                    child = self._mutate(self._crossover(commander, stag))
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

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness

    def _calculate_harem_sizes(self, num_commanders, num_females):
        proportions = np.random.dirichlet(np.ones(num_commanders))
        sizes = (proportions * num_females).astype(int)
        sizes[:num_females - sizes.sum()] += 1  # Distribute remaining females
        return sizes

    def _crossover(self, parent1, parent2):
        return parent1 * np.random.rand(self.dimensions) + parent2 * (1 - np.random.rand(self.dimensions))

    def ConvexCombination(self, *args):        
        alpha = np.array([1 / (self.objective_func(i) + 1e-6) for i in args])  # Avoid division by zero
        return np.array(args).T @ (alpha / alpha.sum())

    def _mutate(self, individual):
        mask = np.random.rand(self.dimensions) < 0.1
        noise = np.random.normal(0, 0.1, self.dimensions)
        return np.clip(individual + mask * noise, self.bounds[0], self.bounds[1])


# Example usage
if __name__ == "__main__":
    def sphere(x):
        return np.sum(x**2)

    rdo = RedDeerOptimization(sphere, dimensions=30, bounds=(-5.12, 5.12), population_size=50, num_males=15, roosting_males=0.4, max_iter=100)
    best_solution, best_fitness = rdo.optimize()

    print("\nOptimization Results:")
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness:.6f}")
