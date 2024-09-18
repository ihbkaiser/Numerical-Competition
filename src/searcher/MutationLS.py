import random
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils.Parameter import *
from utils.Individual import Individual
class MutationLS:
    def __init__(self, prob):
        self.prob = prob
        self.effective = float('inf')
    def search(self, ind, maxeval):
        # Mutation only use 1 function evaluation
        p = ind.genes
        bounds = ind.bounds
        p = (p - bounds[0]) / (bounds[1] - bounds[0])
        mp = float(1. / p.shape[0])
        u = np.random.uniform(size=[p.shape[0]])
        r = np.random.uniform(size=[p.shape[0]])
        tmp = np.copy(p)
        for i in range(p.shape[0]):
            if r[i] < mp:
                if u[i] < 0.5:
                    delta = (2 * u[i]) ** (1 / (1 + Parameter.mum)) - 1
                    tmp[i] = p[i] + delta * p[i]
                else:
                    delta = 1 - (2 * (1 - u[i])) ** (1 / (1 + Parameter.mum))
                    tmp[i] = p[i] + delta * (1 - p[i])
        tmp = np.clip(tmp, 0, 1)
        tmp = tmp * (bounds[1] - bounds[0]) + bounds[0]
        new_ind = Individual(len(ind.genes), bounds)
        new_ind.genes = tmp
        new_ind.cal_fitness(self.prob)
        if new_ind.fitness < ind.fitness:
            self.effective = (ind.fitness - new_ind.fitness) / abs(ind.fitness)
            return new_ind
        else:
            self.effective = 0
            return ind

