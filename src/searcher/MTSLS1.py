import numpy as np
import copy
import sys
import os
sys.path.append(os.path.dirname(__file__))
from utils.Individual import Individual
class MTSLS1:
    def __init__(self, prob):
        self.prob = prob
        self.SR = np.array([0.2]*prob.dim)
        self.effective = float('inf')
    def __mtsls_improve_dim(self, sol, i, SR):
        best_fitness = sol.fitness
        newsol = copy.deepcopy(sol)
        newsol.genes[i] -= ((sol.bounds[1] - sol.bounds[0])*SR[i] + sol.bounds[0])
        newsol.genes = np.clip(newsol.genes, sol.bounds[0], sol.bounds[1])
        newsol.cal_fitness(self.prob)
        fitness_newsol = newsol.fitness
        if fitness_newsol < best_fitness:
            sol = newsol 
        elif fitness_newsol > best_fitness:
            newsol = copy.deepcopy(sol)
            newsol.genes[i] += (200*0.5*SR[i] - 100)
            newsol.genes = np.clip(newsol.genes, -100,100)
            newsol.cal_fitness(self.prob)
            fitness_newsol = newsol.fitness
            if fitness_newsol < best_fitness:
                sol = newsol
        return sol
    def search(self, ind, maxeval):
        dim = len(ind.genes)
        improvement = np.zeros(dim)
        dim_sorted = np.random.permutation(dim)
        improved_dim = np.zeros(dim)
        current_best = copy.deepcopy(ind)
        before_fitness = current_best.fitness
        maxevals = self.prob.FE + maxeval
        if self.prob.FE < maxevals:
            dim_sorted = np.random.permutation(dim)

            for i in dim_sorted:
                result = self.__mtsls_improve_dim(current_best, i, self.SR)
                improve = max(current_best.fitness - result.fitness, 0)
                improvement[i] = improve
                if improve:
                    improved_dim[i] = 1
                    current_best = result
                else:
                    self.SR[i] /=2
            dim_sorted = improvement.argsort()[::-1]
            d = 0
        while self.prob.FE < maxevals:
            i = dim_sorted[d]
            result = self.__mtsls_improve_dim(current_best, i, self.SR)
            improve = max(current_best.fitness- result.fitness, 0)
            improvement[i] = improve
            next_d = (d+1)%dim 
            next_i = dim_sorted[next_d]
            if improve:
                improved_dim[i] = 1
                current_best = result
                if improvement[i] < improvement[next_i]:
                    dim_sorted = improvement.argsort()[::-1]
            else:
                self.SR[i] /= 2
                d = next_d
        initial_SR = 0.2 
        self.SR[self.SR < 1e-15] = initial_SR
        after_fitness = current_best.fitness

        if after_fitness > before_fitness:
            raise ValueError(f"MTSLS1-Error: Fitness must be improved, but got {after_fitness - before_fitness}", )
        self.effective = (before_fitness - after_fitness) / abs(before_fitness)
        return current_best
    def reset(self):
        self.SR = np.array([0.5]*30)