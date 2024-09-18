import random
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils.Parameter import *
from utils.Individual import Individual
class Mutation:
    def __init__(self, prob, survival_rate = None):
        self.effective = None
        self.survival_rate = survival_rate if survival_rate is not None else 0
        self.prob = prob
        self.key = random.random()
    def mutation_ind(self, parent):
        p = parent.genes
        bounds = parent.bounds
        p = (p - bounds[0]) / (bounds[1] - bounds[0])
        mp = float(1. / p.shape[0])
        u = np.random.uniform(size=[p.shape[0]])
        r = np.random.uniform(size=[p.shape[0]])             
        tmp = np.copy(p)
        for i in range(p.shape[0]):
            if r[i] < mp:
                if u[i] < 0.5:
                    delta = (2*u[i]) ** (1/(1+Parameter.mum)) - 1
                    tmp[i] = p[i] + delta * p[i]
                else:
                    delta = 1 - (2 * (1 - u[i])) ** (1/(1+Parameter.mum))
                    tmp[i] = p[i] + delta * (1 - p[i])
        tmp = np.clip(tmp, 0, 1)
        tmp = tmp * (bounds[1] - bounds[0]) + bounds[0]
        ind = Individual(len(parent.genes), bounds)
        ind.genes = tmp
        ind.cal_fitness(self.prob)
        return ind
    def search(self, pool):
        accumulate_diff = 0
        new_pool = []
        old_FEs = self.prob.FE
        for ind in pool:
            new_ind = self.mutation_ind(ind)
            diff = ind.fitness - new_ind.fitness
            if diff > 0:
                new_pool.append(new_ind)
                accumulate_diff += diff
            else:
                new_pool.append(ind)
        new_FEs = self.prob.FE
        self.effective = accumulate_diff / (new_FEs - old_FEs)
        
        return new_pool