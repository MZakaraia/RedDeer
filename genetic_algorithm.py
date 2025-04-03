import numpy as np


class GeneticAlgorithm:
    def __init__(self, objective_func, population_size=50, dimensions=3, 
                 mutation_rate=0.1, crossover_rate=0.9,
                 bounds=(-5.12, 5.12), max_generations=100):
        self.pop_size = population_size
        self.dims = dimensions
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.bounds = bounds
        self.max_gens = max_generations
        self.fn = objective_func
        
    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], 
                               (self.pop_size, self.dims))
    
    def evaluate_fitness(self, population):
        return np.array([1/(1 + self.fn(ind)) for ind in population])
    
    def select_parents(self, population, fitness):
        indices = np.random.choice(np.arange(self.pop_size), size=2, 
                                 p=fitness/fitness.sum())
        return population[indices[0]], population[indices[1]]
    
    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dims)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        for i in range(self.dims):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.5)
                individual[i] = np.clip(individual[i], *self.bounds)
        return individual
    
    def run(self):
        population = self.initialize_population()
        best_fitness = []
        
        for gen in range(self.max_gens):
            fitness = self.evaluate_fitness(population)
            best_idx = np.argmin([self.fn(ind) for ind in population])
            best_fitness.append(self.fn(population[best_idx]))
            
            new_population = []
            for _ in range(self.pop_size//2):
                p1, p2 = self.select_parents(population, fitness)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_population.extend([c1, c2])
                
            population = np.array(new_population)
        
        self.bestFitness = best_fitness[-1]
            
        return population[best_idx], self.bestFitness , best_fitness

