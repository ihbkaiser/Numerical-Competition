import random
import numpy as np
from scipy.stats import cauchy
from MFEAcode import *
from GNBG import GNBG
import copy
from scipy.optimize import fmin_l_bfgs_b
from decimal import Decimal
from typing import override
class jSO:
    def __init__(self, prob, survival_rate = None, H = 6, best_rate = 0.1):
        self.prob = prob
        self.H = H
        self.effective = None
        self.survival_rate = survival_rate if survival_rate is not None else 0
        self.mem_cr = [0.8] * self.H
        self.mem_f = [0.3] * self.H
        self.p_selection = [1/H] * H
        self.num_selection = [0] * H 
        self.num_success = [0] * H
        self.archive = []
        self.success = False
        self.mem_pos = 0
        self.p_max = 0.17
        self.p_min = 0.085
        # pbest = p * len(pool)
        # 
        self.archive_rate = 1.3
        self.control_factor = 3

    def generateFCR(self, m_f, m_cr):
        if m_cr == "TTTG":  #terminal value
            cr = 0
        else:
            cr = cauchy.rvs(loc=m_cr, scale=0.1, size=1)[0]
        if cr < 0:
            cr = 0
        if cr > 1:
            cr = 1
        while True:
            f = np.random.normal(loc=m_f, scale=0.1)
            if f>=0:
                break
        return min(f,1), cr
    def meanWL_CR(self, diff_f, s_cr):
        diff_f = np.array(diff_f)
        s_cr = np.array(s_cr)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tu = sum(weight * s_cr * s_cr)
        mau = sum(weight * s_cr)
        return tu/mau
    def meanWL_F(self, diff_f, s_f):
        diff_f = np.array(diff_f)
        s_f = np.array(s_f)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tu = sum(weight * s_f * s_f)
        mau = sum(weight * s_f)
        return tu/mau
    def updateMemory(self, s_cr, s_f, diff_f):
        if self.success:
            if self.mem_cr[self.mem_pos] == "TTTG" or max(s_cr)==0:
                self.mem_cr[self.mem_pos] = "TTTG"
            else:
                self.mem_cr[self.mem_pos] = 1/2 * (self.meanWL_CR(diff_f, s_cr) + self.mem_cr[self.mem_pos] )
            self.mem_f[self.mem_pos] = 1/2 * (self.meanWL_F(diff_f, s_f) + self.mem_f[self.mem_pos] )
            self.mem_pos = (self.mem_pos + 1) % self.H
            for i in range(self.H):
                if self.num_selection[i] != 0:
                    self.p_selection[i] = self.num_success[i] / self.num_selection[i]
                else:
                    self.p_selection[i] = 0
        else:
            self.p_selection = [1/self.H] * self.H
            self.num_selection = [0] * self.H
            self.num_success = [0] * self.H


    def search(self, pool):
        
        self.success = False
        old_FEs = Parameter.FEs
        accumulate_diff = 0
        s_cr = []
        s_f = []
        diff_f = []
        new_pool = []
        while len(self.archive) > self.archive_rate * len(pool):
            self.archive.pop(0)
        p = self.p_min + (self.p_max - self.p_min)* Parameter.FEs / Parameter.MAX_FEs
        pool.sort(key = lambda ind: ind.fitness[ind.skill_factor-1])
        best = pool[: max(  int(p * len(pool))  ,   2)]
        rank = 1 + self.control_factor * len(pool) - self.control_factor * np.arange(1, len(pool) + 1)
        Pr = rank / sum(rank)

        for ind in pool:
            r = random.choices(list(range(self.H)), weights = self.p_selection, k = 1)[0]
            self.num_selection[r] += 1
            f, cr = self.generateFCR(self.mem_f[r], self.mem_cr[r])
            if Parameter.FEs < 0.6*Parameter.MAX_FEs:
                f = min(f, 0.7)
            if Parameter.FEs < 0.25*Parameter.MAX_FEs:
                cr = max(cr, 0.7)
            if (0.25*Parameter.MAX_FEs <= Parameter.FEs) and (Parameter.FEs < 0.5*Parameter.MAX_FEs):
                cr = max(cr, 0.6)
            if Parameter.FEs < 0.2*Parameter.MAX_FEs:
                fw = 0.7*f 
            if (0.2*Parameter.MAX_FEs <= Parameter.FEs) and (Parameter.FEs < 0.4*Parameter.MAX_FEs):
                fw = 0.8*f
            if Parameter.FEs >= 0.4*Parameter.MAX_FEs:
                fw = 1.2*f
            pbest = random.sample(best, 1)[0]
            pr1 = random.choices(pool, weights = Pr, k = 1)[0]
            r2 = random.choices(pool + self.archive, k=1)[0]
            donor = ind.genes + fw * (pbest.genes - ind.genes) + f*(pr1.genes - r2.genes)
            for j in range(len(donor)):
                if donor[j] > 1:
                    donor[j] = (1 + ind.genes[j]) / 2
                elif donor[j] < 0:
                    donor[j] = ind.genes[j] / 2
            u = np.random.uniform(0, 1, size = len(ind.genes))
            mask = u < cr
            mask[np.random.choice(len(ind.genes))] = True
            new_genes = np.where(mask, donor, ind.genes)
            # assert new_genes == np.clip(new_genes, 0, 1)
            new_ind = Individual(len(ind.genes), ind.MAX_OBJS)
            new_ind.genes = new_genes
            new_ind.skill_factor = ind.skill_factor
            new_ind.cal_fitness(self.prob)
            if new_ind.fitness[new_ind.skill_factor-1] < ind.fitness[ind.skill_factor-1]:
                new_pool.append(new_ind)
                self.archive.insert(0, ind)
                self.num_success[r] += 1
                s_cr.append(cr)
                s_f.append(f)
                diff_f.append(ind.fitness[ind.skill_factor-1] - new_ind.fitness[new_ind.skill_factor-1])
                accumulate_diff += ind.fitness[ind.skill_factor-1] - new_ind.fitness[new_ind.skill_factor-1]
                self.success = True
            else:
                new_pool.append(ind)
        self.updateMemory(s_cr, s_f, diff_f)
        new_FEs = Parameter.FEs
        self.effective = accumulate_diff / (new_FEs - old_FEs)
        assert min(ind.fitness[ind.skill_factor-1] for ind in pool) >= min(new_ind.fitness[new_ind.skill_factor-1] for new_ind in new_pool), "Maximum fitness of pool must be greater than or equal to maximum fitness of new_pool"

        return new_pool
            
            

class Mutation:
    def __init__(self, prob, survival_rate = None):
        self.effective = None
        self.survival_rate = survival_rate if survival_rate is not None else 0
        self.prob = prob
        self.key = random.random()
 
    def mutation_ind(self,parent):
        p = parent.genes 
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
        ind = Individual(len(parent.genes), parent.MAX_OBJS)
        ind.genes = tmp
        ind.skill_factor = parent.skill_factor
        ind.cal_fitness(self.prob)
        return ind
    def search(self, pool):
        accumulate_diff = 0
        new_pool = []
        old_FEs = Parameter.FEs
        for ind in pool:
            new_ind = self.mutation_ind(ind)
            diff = ind.fitness[ind.skill_factor-1] - new_ind.fitness[new_ind.skill_factor-1]
            if diff > 0:
                new_pool.append(new_ind)
                accumulate_diff += diff
            else:
                new_pool.append(ind)
        new_FEs = Parameter.FEs
        self.effective = accumulate_diff / (new_FEs - old_FEs)

        assert min(ind.fitness[ind.skill_factor-1] for ind in pool) >= min(new_ind.fitness[new_ind.skill_factor-1] for new_ind in new_pool), "Maximum fitness of pool must be greater than or equal to maximum fitness of new_pool"
        
        return new_pool




class DE:
    def __init__(self, init_F, init_CR, prob, survival_rate = None, H = 6, best_rate = 0.1 ):
        self.prob = prob
        self.H = H
        self.effective = None
        self.survival_rate = survival_rate if survival_rate is not None else 0
        self.mem_cr = [init_CR] * self.H
        self.mem_f = [init_F] * self.H
        self.s_cr = []
        self.s_f = []
        self.diff_f = []
        self.archive = []
        self.mem_pos = 0
        self.key = random.random()
        self.archive_rate = 1.3
    
    def generateFCR(self, m_f, m_cr):
        while True:
            f = np.random.normal(loc=m_f, scale=0.1)
            cr = cauchy.rvs(loc=m_cr, scale=0.1, size=1)[0]
            if f>=0 and cr>=0:
                break
        return min(f,1), min(cr,1)
    def pbest_ind(self, pool, ind, cr, f, best):
        pbest = best[random.randint(0, len(best) - 1)]
        rand_idx = np.random.choice(len(pool), 2, replace=False)
        ind_ran1, ind_ran2 = pool[rand_idx[0]], pool[rand_idx[1]]
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) <= cr)
        u[np.random.choice(len(ind.genes))] = True

        new_genes = np.where(u, 
            pbest.genes + f * (ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        for j in range(len(new_genes)):
                if new_genes[j] > 1:
                    new_genes[j] = (1 + ind.genes[j]) / 2
                elif new_genes[j] < 0:
                    new_genes[j] = ind.genes[j] / 2
        new_ind = Individual(ind.MAX_NVARS, ind.MAX_OBJS)
        new_ind.genes = new_genes
        new_ind.skill_factor = ind.skill_factor
        new_ind.cal_fitness(self.prob)
        return new_ind
    def search(self, pool):
        pool.sort(key = lambda ind: ind.fitness[ind.skill_factor-1])
        best = pool[: max( int(0.1*len(pool)), 2 )]
        new_pool = []
        accumulate_diff = 0
        old_FEs = Parameter.FEs
        for ind in pool:
            r = random.randint(0, self.H - 1)
            f, cr = self.generateFCR(self.mem_f[r], self.mem_cr[r])
            new_ind = self.pbest_ind(pool, ind, cr, f, best)
            if ind.fitness[ind.skill_factor-1] > new_ind.fitness[new_ind.skill_factor-1]:
                self.diff_f.append(ind.fitness[ind.skill_factor-1] - new_ind.fitness[new_ind.skill_factor-1])
                self.s_cr.append(cr)
                self.s_f.append(f)
                self.archive.append(ind)
                new_pool.append(new_ind)
                accumulate_diff += ind.fitness[ind.skill_factor-1] - new_ind.fitness[new_ind.skill_factor-1]

            else:
                new_pool.append(ind)
        # print max fitness of new_pool 
        min_fitness1 = min(ind.fitness[ind.skill_factor-1] for ind in pool)
        min_fitness = min(ind.fitness[ind.skill_factor-1] for ind in new_pool)
        assert min_fitness <= min_fitness1, f'Fitness of new pool: {min_fitness} <= old pool:{min_fitness1} ???'
        new_FEs = Parameter.FEs
        self.effective = accumulate_diff / (new_FEs - old_FEs)
        self.updateMemory()
        return new_pool
    
    def updateMemoryCR(self, diff_f, s_cr):
        diff_f = np.array(diff_f)
        s_cr = np.array(s_cr)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tu = sum(weight * s_cr * s_cr)
        mau = sum(weight * s_cr)
        if(mau != 0):
            return tu/mau
        else:
            return 0
    def updateMemoryF(self, diff_f, s_f):
        diff_f = np.array(diff_f)
        s_f = np.array(s_f)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tu = sum(weight * s_f * s_f)
        mau = sum(weight * s_f)
        if(mau!=0):
            return tu/mau
        else:
            return 0
    def updateMemory(self):
        if len(self.s_cr) > 0:
            self.mem_cr[self.mem_pos] = self.updateMemoryCR(self.diff_f, self.s_cr)
            self.mem_f[self.mem_pos] = self.updateMemoryF(self.diff_f, self.s_f)
            self.mem_pos = (self.mem_pos + 1) % self.H
            self.s_cr = []
            self.s_f = []
            self.diff_f = []
    
        else:
            pass

class Crossover:
    def __init__(self, prob, survival_rate = None):
        self.prob = prob
        self.survival_rate = survival_rate if survival_rate is not None else 0
        self.effective = None
    def crossover(self, parent1, parent2):
        offspring = []
        p1 = parent1.genes
        p2 = parent2.genes
        D = p1.shape[0]
        cf = np.empty([D])
        u = np.random.rand(D)        

        cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (Parameter.mu + 1)))
        cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (Parameter.mu + 1)))

        c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
        c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)

        c1 = np.clip(c1, 0, 1)
        c2 = np.clip(c2, 0, 1)
        swap = parent1.skill_factor == parent2.skill_factor
        if swap:
            idx_swap = np.where(np.random.rand(parent1.MAX_NVARS) < 0.5)[0]
            c1[idx_swap], c2[idx_swap] = c2[idx_swap], c1[idx_swap]
    
        child1 = Individual(parent1.MAX_NVARS,  parent1.MAX_OBJS)
        child2 = Individual(parent1.MAX_NVARS, parent1.MAX_OBJS)
        child1.genes = c1 
        child2.genes = c2
        child1.skill_factor = parent1.skill_factor
        child2.skill_factor = parent2.skill_factor
        child1.cal_fitness(self.prob)
        offspring.extend([child1, child2])
        return offspring
    def search(self, pool):
        accumulate_diff = 0
        new_pool = []
        old_FEs = Parameter.FEs
        while(True):
            parent1, parent2 = random.sample(pool, 2)
            off, _ = self.crossover(parent1, parent2)
            diff = min(parent1.fitness[parent1.skill_factor-1] , parent2.fitness[parent2.skill_factor-1]) - off.fitness[off.skill_factor-1]
            if diff > 0:
                new_pool.append(off)
                accumulate_diff += diff
            else:
                if random.random() < self.survival_rate:
                    new_pool.append(off)
                else:
                    if parent1.fitness[parent1.skill_factor-1] < parent2.fitness[parent2.skill_factor-1]:
                        new_pool.append(parent1)
                    else:
                        new_pool.append(parent2)

            if len(new_pool) > len(pool):
                break
        
        new_FEs = Parameter.FEs
        self.effective = accumulate_diff / (new_FEs - old_FEs)
        return new_pool


        

######################## Iterative Local Search Classes ############################
class MTSLS1:
    def __init__(self, prob):
        self.prob = prob
        self.SR = np.array([0.2]*30)
        self.effective = float('inf')
    def __mtsls_improve_dim(self, sol, i, SR):
        best_fitness = sol.fitness[sol.skill_factor-1]
        newsol = copy.deepcopy(sol)
        newsol.genes[i] -= SR[i]
        newsol.genes = np.clip(newsol.genes, 0, 1)
        newsol.skill_factor = sol.skill_factor
        newsol.cal_fitness(self.prob)
        fitness_newsol = newsol.fitness[newsol.skill_factor-1]
        if fitness_newsol < best_fitness:
            sol = newsol 
        elif fitness_newsol > best_fitness:
            newsol = copy.deepcopy(sol)
            newsol.genes[i] += 0.5*SR[i]
            newsol.genes = np.clip(newsol.genes, 0,1)
            newsol.skill_factor = sol.skill_factor # do we really need this ???
            newsol.cal_fitness(self.prob)
            fitness_newsol = newsol.fitness[newsol.skill_factor-1]
            if fitness_newsol < best_fitness:
                sol = newsol
        return sol
    def search(self, ind, maxeval):
        dim = len(ind.genes)
        improvement = np.zeros(dim)
        dim_sorted = np.random.permutation(dim)
        improved_dim = np.zeros(dim)
        current_best = copy.deepcopy(ind)
        before_fitness = current_best.fitness[current_best.skill_factor-1]
        maxevals = Parameter.FEs + maxeval
        if Parameter.FEs < maxevals:
            dim_sorted = np.random.permutation(dim)

            for i in dim_sorted:
                result = self.__mtsls_improve_dim(current_best, i, self.SR)
                improve = max(current_best.fitness[current_best.skill_factor-1] - result.fitness[result.skill_factor-1], 0)
                improvement[i] = improve
                if improve:
                    improved_dim[i] = 1
                    current_best = result
                else:
                    self.SR[i] /=2
            dim_sorted = improvement.argsort()[::-1]
            d = 0
        while Parameter.FEs < maxevals:
            i = dim_sorted[d]
            result = self.__mtsls_improve_dim(current_best, i, self.SR)
            improve = max(current_best.fitness[current_best.skill_factor-1] - result.fitness[result.skill_factor-1], 0)
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
        after_fitness = current_best.fitness[current_best.skill_factor-1]

        if after_fitness > before_fitness:
            raise ValueError(f"MTSLS1-Error: Fitness must be improved, but got {after_fitness - before_fitness}", )
        self.effective = (before_fitness - after_fitness) / abs(before_fitness)
        return current_best
    def reset(self):
        self.SR = np.array([0.5]*30)

class LBFGSB:
    def __init__(self, prob):
        self.prob = prob
        self.effective = float('inf')
    def search(self, ind, maxeval):

        before_fitness = ind.fitness[ind.skill_factor-1]
        sol, fit, info = fmin_l_bfgs_b(self.prob[ind.skill_factor-1], ind.genes, approx_grad=True, bounds=[(0,1)]*len(ind.genes), maxfun=maxeval)
        # sol2, fit2, info2 = fmin_l_bfgs_b(self.prob[ind.skill_factor-1], np.random.rand(len(ind.genes)), approx_grad=True, bounds=[(0,1)]*len(ind.genes), maxfun=maxeval)
        if fit > before_fitness:
            new_ind = ind
        else:
            new_ind = copy.deepcopy(ind)
            new_ind.genes = sol
            new_ind.skill_factor = ind.skill_factor
            for i in range(len(self.prob)):
                if i == ind.skill_factor-1:
                    new_ind.fitness[i] = fit
                else:
                    new_ind.fitness[i] = float("inf")
        after_fitness = new_ind.fitness[new_ind.skill_factor-1]
        if after_fitness > before_fitness:
            raise ValueError(f"BFGS-Error: Fitness must be improved, but got {after_fitness - before_fitness}", )
        self.effective = (before_fitness - after_fitness) / abs(before_fitness)
        Parameter.FEs += info["funcalls"]
        # best_generated_fitness = min(fit, fit2)
        # best_generated_genes = sol if fit < fit2 else sol2
        # if best_generated_fitness > before_fitness:
        #     new_ind = copy.deepcopy(ind)
        # else:
        #     new_ind = Individual(len(ind.genes), ind.MAX_OBJS)
        #     new_ind.genes = best_generated_genes
        #     new_ind.skill_factor = ind.skill_factor
        #     for i in range(len(self.prob)):
        #         if i == ind.skill_factor-1:
        #             new_ind.fitness[i] = best_generated_fitness
        #         else:
        #             new_ind.fitness[i] = float("inf")
        # after_fitness = new_ind.fitness[new_ind.skill_factor-1]
        # if after_fitness > before_fitness:
        #     raise ValueError(f"BFGS-Error: Fitness must be improved, but got {after_fitness - before_fitness}", )
        # self.effective = (before_fitness - after_fitness) / abs(before_fitness)
        # Parameter.FEs += (info["funcalls"] + info2["funcalls"])
        return new_ind
    def reset(self):
        pass

class MultiDE(DE):
    def __init__(self, init_F, init_CR, prob, survival_rate = None, H = 6, best_rate = 0.1 ):
        super().__init__(init_F, init_CR, prob, survival_rate, H, best_rate)
        self.archive = []
        self.operators = [self.operator1, self.operator2, self.operator3]
    @override
    def generateFCR(self, m_f, m_cr):
        while True:
            f = np.random.normal(loc=m_f, scale=0.1)
            cr = cauchy.rvs(loc=m_cr, scale=0.1, size=1)[0]
            cr = np.clip(cr, 0, 1)
            if f>0:
                break
        return min(f,1), min(cr,1)
    def operator1(self, pool, ind, cr, f, best):
        pbest = best[random.randint(0, len(best) - 1)]
        rand_idx = np.random.choice(len(pool), 2, replace=False)
        ind_ran1, ind_ran2 = pool[rand_idx[0]], pool[rand_idx[1]]
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) <= cr)
        u[np.random.choice(len(ind.genes))] = True

        new_genes = np.where(u, 
            pbest.genes + f * (ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        for j in range(len(new_genes)):
                if new_genes[j] > 1:
                    new_genes[j] = (1 + ind.genes[j]) / 2
                elif new_genes[j] < 0:
                    new_genes[j] = ind.genes[j] / 2
        new_ind = Individual(ind.MAX_NVARS, ind.MAX_OBJS)
        new_ind.genes = new_genes
        new_ind.skill_factor = ind.skill_factor
        new_ind.cal_fitness(self.prob)
        return new_ind
    def operator2(self, pool, ind, cr, f, best):
        pbest = best[random.randint(0, len(best) - 1)]
        ind_ran1 = pool[random.randint(0, len(pool) - 1)]
        ind_ran2 = random.choices(pool + self.archive, k=1)[0]
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) <= cr)
        u[np.random.choice(len(ind.genes))] = True
        trial_vector = ind.genes + f*(pbest.genes - ind.genes + ind_ran1.genes- ind_ran2.genes)
        new_genes = np.where(u, trial_vector, ind.genes)
        for j in range(len(new_genes)):
                if new_genes[j] > 1:
                    new_genes[j] = (1 + ind.genes[j]) / 2
                elif new_genes[j] < 0:
                    new_genes[j] = ind.genes[j] / 2
        new_ind = Individual(ind.MAX_NVARS, ind.MAX_OBJS)
        new_ind.genes = new_genes
        new_ind.skill_factor = ind.skill_factor
        new_ind.cal_fitness(self.prob)
        return new_ind
    def operator3(self, pool, ind, cr, f, best):
        pbest = best[random.randint(0, len(best) - 1)]
        ind_ran1, ind_ran3 = random.choices(pool, k=2)
        ind_ran2 = random.choices(pool + self.archive, k=1)[0]
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) <= cr)
        u[np.random.choice(len(ind.genes))] = True
        trial_vector = ind_ran3.genes - f * (pbest.genes - ind.genes + ind_ran1.genes - ind_ran2.genes)
        new_genes = np.where(u, trial_vector, ind.genes)
        for j in range(len(new_genes)):
                if new_genes[j] > 1:
                    new_genes[j] = (1 + ind.genes[j]) / 2
                elif new_genes[j] < 0:
                    new_genes[j] = ind.genes[j] / 2
        new_ind = Individual(ind.MAX_NVARS, ind.MAX_OBJS)
        new_ind.genes = new_genes
        new_ind.skill_factor = ind.skill_factor
        new_ind.cal_fitness(self.prob)
        return new_ind
    # def operator4(self, pool, ind, cr, f, best):
    #     new_ind = Mutation.mutation_ind(ind)
    #     return new_ind

    def search(self, pool):
        pool.sort(key = lambda ind: ind.fitness[ind.skill_factor-1])
        best = pool[: max( int(0.1*len(pool)), 2 )]
        new_pool = []
        accumulate_diff = 0
        old_FEs = Parameter.FEs
        for ind in pool:
            r = random.randint(0, self.H - 1)
            f, cr = self.generateFCR(self.mem_f[r], self.mem_cr[r])
            operator = random.choice(self.operators)
            new_ind = operator(pool, ind, cr, f, best)
            if ind.fitness[ind.skill_factor-1] > new_ind.fitness[new_ind.skill_factor-1]:
                self.diff_f.append(ind.fitness[ind.skill_factor-1] - new_ind.fitness[new_ind.skill_factor-1])
                self.s_cr.append(cr)
                self.s_f.append(f)
                if len(self.archive) > self.archive_rate * len(pool):
                    self.archive.sort(key = lambda ind: ind.fitness[ind.skill_factor-1])
                    self.archive = self.archive[: (int(self.archive_rate * len(pool)) - 1)]
                self.archive.append(ind)
                new_pool.append(new_ind)
                accumulate_diff += ind.fitness[ind.skill_factor-1] - new_ind.fitness[new_ind.skill_factor-1]

            else:
                new_pool.append(ind)
        # print max fitness of new_pool 
        min_fitness1 = min(ind.fitness[ind.skill_factor-1] for ind in pool)
        min_fitness = min(ind.fitness[ind.skill_factor-1] for ind in new_pool)
        assert min_fitness <= min_fitness1, f'Fitness of new pool: {min_fitness} <= old pool:{min_fitness1} ???'
        new_FEs = Parameter.FEs
        self.effective = accumulate_diff / (new_FEs - old_FEs)
        self.updateMemory()
        return new_pool
