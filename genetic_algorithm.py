import numpy as np

class GeneticAlgorithm:
    def __init__(self, objective_func, population_size=50, dimensions=3, 
                 mutation_rate=0.1, crossover_rate=0.9,
                 bounds=(-5.12, 5.12), max_generations=100, elite_size=1):
        self.pop_size = population_size
        self.dims = dimensions
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.bounds = bounds
        self.max_gens = max_generations
        self.elite_size = elite_size
        self.fn = objective_func
        
    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], 
                                 (self.pop_size, self.dims))
    
    def evaluate_fitness(self, population):
        return np.array([1 / (1 + self.fn(ind)) for ind in population])
    
    def tournament_selection(self, population, fitness, k=3):
        """Tournament Selection: Picks the best out of k randomly chosen individuals."""
        selected = np.random.choice(np.arange(self.pop_size), size=k, replace=False)
        best_idx = selected[np.argmax(fitness[selected])]
        return population[best_idx]
    
    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dims)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual, generation):
        """Adaptive mutation: reduces mutation rate over generations."""
        adaptive_rate = self.mutation_rate * (1 - generation / self.max_gens)
        for i in range(self.dims):
            if np.random.rand() < adaptive_rate:
                individual[i] += np.random.normal(0, 0.3 * (1 - generation / self.max_gens))
                individual[i] = np.clip(individual[i], *self.bounds)
        return individual
    
    def run(self):
        population = self.initialize_population()
        best_fitness = []
        
        for gen in range(self.max_gens):
            fitness = self.evaluate_fitness(population)
            sorted_indices = np.argsort(-fitness)  # Sort in descending order
            best_individuals = population[sorted_indices[:self.elite_size]]  # Preserve elites
            
            best_fitness.append(self.fn(population[sorted_indices[0]]))
            
            new_population = list(best_individuals)  # Start with elites
            
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection(population, fitness)
                p2 = self.tournament_selection(population, fitness)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1, gen)
                c2 = self.mutate(c2, gen)
                new_population.extend([c1, c2])
            
            population = np.array(new_population[:self.pop_size])  # Ensure correct population size
        
        self.bestFitness = best_fitness[-1]
        return population[sorted_indices[0]], self.bestFitness, best_fitness
