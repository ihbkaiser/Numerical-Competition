import random
import numpy as np
from scipy.stats import cauchy
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils.Individual import Individual
# this is implementation under the jSO paper for solving CEC2017
class jSOError(Exception):
    """exception class for number of generations set two low."""
    def __init__(self, message):
        super().__init__(message)
class jSO:
    def __init__(self, prob, init_F=0.5, init_CR=0.8, H=5, best_rate = 0.25, archive_rate = 2.6,
    G_max = 2500):
    # if you see jsoError and it leads you here, 
    # change G_max to a bigger number (like +100, +200) until the error is gone, but not too big
        self.prob = prob
        self.H = H
        self.effective = None
        self.G_max = G_max
        self.G = 0
        self.mem_cr = [init_CR] * self.H
        self.mem_f = [init_F] * self.H
        self.s_cr = []
        self.s_f = []
        self.diff_f = []
        self.mem_pos = 0
        self.archive_rate = archive_rate
        self.archive = []
        self.p_max = best_rate
        self.p_min = best_rate/2
        self.p = best_rate
    def generateFCR(self, m_f, m_cr):
        if m_cr < 0:
            cr = 0
        else:
            cr = np.random.normal(loc=m_cr, scale=0.1)
            cr = np.clip(cr, 0, 1)
        while True:
            f = cauchy.rvs(loc = m_f, scale=0.1, size=1 )[0]
            if f>=0:
                break
        if self.G < 0.25*self.G_max:
            f = min(f, 0.7)
            cr = max(cr, 0.5)
        elif self.G < 0.5*self.G_max:
            f = min(f, 0.8)
            cr = max(cr, 0.25)
        elif self.G < 0.75*self.G_max:
            f = min(f, 0.9)
        
        return min(f,1), min(cr,1)
    def cur_to_pbest_w_1_bin(self, pool, ind, cr, f, best):
        pbest = best[random.randint(0, len(best) - 1)]
        rand_idx = np.random.choice(len(pool), 2, replace=False)
        ind_ran1 = pool[rand_idx[0]]
        ind_ran2 = random.sample(pool+self.archive, 1)[0]
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) <= cr)
        u[np.random.choice(len(ind.genes))] = True
        if self.prob.FE < 0.2*self.prob.max_FE:
            coefficent = 0.7
        elif self.prob.FE < 0.4*self.prob.max_FE:
            coefficent = 0.8
        else:
            coefficent = 1.2
        new_genes = np.where(u,
            ind.genes + f*coefficent*(pbest.genes - ind.genes)+f*(ind_ran1.genes-ind_ran2.genes),
            ind.genes
        )
        bounds = ind.bounds
        for j in range(len(new_genes)):
                if new_genes[j] > bounds[1]:
                    new_genes[j] = (bounds[1] + ind.genes[j]) / 2
                elif new_genes[j] < bounds[0]:
                    new_genes[j] = (bounds[0] + ind.genes[j]) / 2
        new_ind = Individual(ind.MAX_NVARS, bounds)
        new_ind.genes = new_genes
        new_ind.cal_fitness(self.prob)
        return new_ind
    def addarchive(self, ind, pop_size):
        caution_size = int(self.archive_rate * pop_size)
        while len(self.archive) > caution_size:
            r = random.randint(0, len(self.archive) - 1)
            self.archive.pop(r)
        self.archive.append(ind)
    def search(self, pool):
        if self.G >= self.G_max and self.prob.FE < self.prob.max_FE:
            raise jSOError(f'Number of FEs left is {self.prob.max_FE - self.prob.FE}, but generation ended: Current G : {self.G}, max G : {self.G_max}')
        self.G += 1
        pool.sort(key = lambda x : x.fitness)
        best = pool[: max( int(self.p*len(pool)), 2 )]
        new_pool = []
        random.shuffle(pool)
        for ind in pool:
            r = random.randint(0, self.H - 1)
            if r==self.H - 1:
                self.mem_cr[r] = 0.9
                self.mem_f[r] = 0.9
            f, cr = self.generateFCR(self.mem_f[r], self.mem_cr[r])
            new_ind = self.cur_to_pbest_w_1_bin(pool, ind, cr, f, best)
            if new_ind.fitness < ind.fitness:
                self.diff_f.append(new_ind.fitness - ind.fitness)
                self.s_cr.append(cr)
                self.s_f.append(f)
                new_pool.append(new_ind)
                self.addarchive(ind, len(pool))
            else:
                new_pool.append(ind)
        self.p = (self.p_max - self.p_min) / self.prob.max_FE * self.prob.FE + self.p_min
        self.updateMemory()
        return new_pool
    def updateMemoryCR(self, diff_f, s_cr):
        diff_f = np.array(diff_f)
        s_cr = np.array(s_cr)
        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tu = sum(weight * s_cr * s_cr)
        mau = sum(weight * s_cr)
        return tu/mau
    def updateMemoryF(self, diff_f, s_f):
        diff_f = np.array(diff_f)
        s_f = np.array(s_f)
        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tu = sum(weight * s_f * s_f)
        mau = sum(weight * s_f)
        return tu/mau
    def updateMemory(self):
        if len(self.s_cr) > 0:
            if self.mem_cr[self.mem_pos] == -1 or max(self.s_cr)==0:
                self.mem_cr[self.mem_pos] = -1
            else:
                self.mem_cr[self.mem_pos] = self.updateMemoryCR(self.diff_f, self.s_cr)
            self.mem_f[self.mem_pos] = self.updateMemoryF(self.diff_f, self.s_f)
            self.mem_pos = (self.mem_pos + 1) % self.H
            self.s_cr = []
            self.s_f = []
            self.diff_f = []
        else:
            # do nothing
            pass

        