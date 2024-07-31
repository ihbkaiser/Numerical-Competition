import numpy as np
class Individual:
    def __init__(self, MAX_NVARS, bounds):
        self.MAX_NVARS = MAX_NVARS
        self.genes = np.zeros(self.MAX_NVARS)
        self.fitness = np.inf
        self.bounds = bounds

    def init(self):
        self.genes = np.random.uniform(self.bounds[0], self.bounds[1], self.MAX_NVARS)

    def fcost(self):
        return self.fitness
    def cal_fitness(self, prob):
        self.fitness = prob.fitness_of_ind(self.genes)