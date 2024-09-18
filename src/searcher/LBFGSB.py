from scipy.optimize import fmin_l_bfgs_b
import copy
import numpy as np
class LBFGSB2:
    def __init__(self, prob):
        self.prob = prob
        self.effective = float('inf')
        self.stop = False
        self.grad = None
    def search(self, ind, maxeval):
        before_fitness = ind.fitness
        if self.stop:
            # if LBFGSB doesn't improve the fitness, we pass a random individual to algo
            input_genes = np.random.uniform(-100, 100, len(ind.genes))
        else:
            input_genes = ind.genes
        sol, fit, info = fmin_l_bfgs_b(self.prob.fitness_of_ind, input_genes, approx_grad=True, bounds=[(-100,100)]*len(ind.genes), maxfun=maxeval)
        #print(f'FITNESS OF IND: {fit}, RANDOM GENES: {self.stop}')
        self.grad = info["grad"]
        if fit > before_fitness:
            new_ind = ind
        else:
            new_ind = copy.deepcopy(ind)
            new_ind.genes = sol
            new_ind.fitness = fit
        after_fitness = new_ind.fitness
        if after_fitness > before_fitness:
            raise ValueError(f"LBFGSB2-Error: Fitness must be improved, but got {after_fitness - before_fitness}", )
        self.effective = (before_fitness - after_fitness) / abs(before_fitness)
        if self.effective == 0:
            self.stop = True

        return new_ind

class LBFGSB:
    def __init__(self, prob):
        self.prob = prob
        self.effective = float('inf')
    def search(self, ind, maxeval):

        before_fitness = ind.fitness
        sol, fit, info = fmin_l_bfgs_b(self.prob.fitness_of_ind, ind.genes, approx_grad=True, bounds=[(-100,100)]*len(ind.genes), maxfun=maxeval)
        # sol2, fit2, info2 = fmin_l_bfgs_b(self.prob[ind.skill_factor-1], np.random.rand(len(ind.genes)), approx_grad=True, bounds=[(0,1)]*len(ind.genes), maxfun=maxeval)
        if fit > before_fitness:
            new_ind = ind
        else:
            new_ind = copy.deepcopy(ind)
            new_ind.genes = sol
            new_ind.fitness = fit
        after_fitness = new_ind.fitness
        if after_fitness > before_fitness:
            raise ValueError(f"BFGS-Error: Fitness must be improved, but got {after_fitness - before_fitness}", )
        self.effective = (before_fitness - after_fitness) / abs(before_fitness)

        return new_ind
    def reset(self):
        pass