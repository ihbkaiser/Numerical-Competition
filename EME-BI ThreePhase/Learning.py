import random
import numpy as np
from scipy.stats import cauchy
from MFEAcode import *
from GNBG import GNBG
import copy
from scipy.optimize import fmin_l_bfgs_b
from Searcher import *

class SubPop():
    def __init__(self, pool, prob, searcher=None):
        self.list_tasks = prob
        self.pool = pool      #expect a list of Individuals with the same skill_factor, that is self.skill_factor
        self.searcher = searcher 
        self.fitness_improv = 0
        self.consume_fes = 0
        self.dim = 30
        self.len_tasks = len(pool[0].fitness)
        self.skill_factor = pool[0].skill_factor
        # self.task = prob[self.skill_factor - 1]
        self.scale = 0.1
    def search(self):
        return self.searcher.search(self.pool)
class LearningPhase():
    M = 2   #number of operators
    def __init__(self, is_start, prob) -> None:
        # self.list_tasks = list_tasks
        # self.num_task = len(list_tasks)
        # self.dim = dim
        self.list_tasks = prob

        self.start = is_start
        self.searcher = [ Mutation(prob)]
    def evolve(self, subpop, sigma: float, max_delta: float, divisionRate: float) :
        ################ MY TECHNIQUE (fail) #####################
        # merge_pool = []
        # for search in self.searcher:
        #     pool = search.search(subpop)
        #     merge_pool.extend(pool)
        # merge_pool.sort(key = lambda ind: ind.fitness[ind.skill_factor-1])
        # return merge_pool[:len(subpop)]
        ###################################################
        # divide subPop wrt divisionRate
    
        random.shuffle(subpop)
        new_pool = []
        sub_pools = []
        if divisionRate > 1/len(self.searcher):
            raise ValueError(f"divisionRate must be less than 1/num_searcher, but got {divisionRate} and {1/len(self.searcher)} respectively.")
        m = len(self.searcher)
        n = int ((1 - (m-1)* divisionRate) * len(subpop))
        # print("effective: ", [search.effective for search in self.searcher])
        
        for i in range(len(self.searcher)):
            if(i==0):
                sub_pools.append(SubPop(subpop[:n], self.list_tasks))
            else:
                sub_pools.append(SubPop(subpop[int(n + (i-1)*divisionRate*len(subpop)):int(n + i*divisionRate*len(subpop))], self.list_tasks))
        assert len(sub_pools) == len(self.searcher)
        

        if self.start:
            opcode = list(range(len(self.searcher)))
            random.shuffle(opcode)
            for idx, subpop in enumerate(sub_pools):
                subpop.searcher = self.searcher[opcode[idx]]
           
        else:
            opcode = np.argsort([search.effective for search in self.searcher])[::-1]
            for i, idx in enumerate(opcode):
                sub_pools[i].searcher = self.searcher[idx]
    
        for i in range(len(self.searcher)):
            pool = sub_pools[i].search()

            new_pool.extend(pool)
        return new_pool

class LearningPhaseILS():
    def __init__(self, prob):
        self.list_tasks = prob
        self.searcher = [MTSLS1(prob), LBFGSB(prob)]
        self.explorer = DE(0.5, 0.5, prob)
        self.stay = 0
        self.use_restart = True
        self.grad_threshold = 30

    def evolve(self, subpop, DE_evals, LS_evals):
        ####### Explorer part (already implemented in Phase 2) ##########
        max_DE = Parameter.FEs + DE_evals
        while Parameter.FEs < max_DE:
            subpop = self.explorer.search(subpop)
        ################################################################
        subpop.sort(key = lambda ind: ind.fitness[ind.skill_factor-1])
        if self.searcher[0].effective > self.searcher[1].effective:
            this_iteration_searcher = self.searcher[0]
        if self.searcher[0].effective < self.searcher[1].effective:
            this_iteration_searcher = self.searcher[1]
        if self.searcher[0].effective == self.searcher[1].effective:
            this_iteration_searcher = random.choice(self.searcher)
        subpop[0] = this_iteration_searcher.search(subpop[0], LS_evals)
        this_fitness = subpop[0].fitness[subpop[0].skill_factor - 1]
         ############ One more L-BFGS-B ######################
        if self.use_restart:
            ind = copy.deepcopy(subpop[0])
            sol2, fit2, info2 = fmin_l_bfgs_b(self.list_tasks[ind.skill_factor-1], np.random.rand(len(ind.genes)), approx_grad=True, bounds=[(0,1)]*len(ind.genes), maxfun=LS_evals, factr=10)
            if fit2 < this_fitness:
                new_ind = Individual(ind.MAX_NVARS, ind.MAX_OBJS)
                new_ind.genes = sol2
                new_ind.skill_factor = ind.skill_factor
                for i in range(len(self.list_tasks)):
                    if i == new_ind.skill_factor - 1:
                        new_ind.fitness[i] = fit2
                    else:
                        new_ind.fitness[i] = float('inf')
                subpop[0] = new_ind
         ##### Check if we use restart next time or not #########
            if np.linalg.norm(info2["grad"]) > self.grad_threshold:
                self.use_restart = False
            if fit2 > this_fitness:
                disimprove_ratio = (fit2 - this_fitness) / abs(this_fitness)
                if disimprove_ratio > 5:
                    self.use_restart = False
                # print(f'Disimprove ratio: {disimprove_ratio}')
   
        
        
        #####################################################
                
        # ratio_improvement = this_iteration_searcher.effective
        # if ratio_improvement > 1e-5:
        #     self.stay = 0
        # else:
        #     self.stay += 1
        #     for _searcher in self.searcher:
        #         _searcher.reset()
        #     if self.stay > 3:
        #         self.stay = 0
        #         subpop = self.restart(subpop)
        return subpop
    def restart(self, subpop):
        print("RESTART!", subpop[0].skill_factor)
        new_subpop = []
        for _searcher in self.searcher:
            _searcher.reset()
        rand_ind = subpop[0]
        rand_ind.genes += np.random.rand() * 0.1 
        rand_ind.genes = np.clip(rand_ind.genes, 0, 1)
        rand_ind.skill_factor = subpop[0].skill_factor
        rand_ind.cal_fitness(self.list_tasks)
        new_subpop.append(rand_ind)
        while len(new_subpop) != len(subpop):
            new_ind = Individual(subpop[0].MAX_NVARS, subpop[0].MAX_OBJS)
            new_ind.init()
            new_ind.skill_factor = subpop[0].skill_factor
            new_ind.cal_fitness(self.list_tasks)
            new_subpop.append(new_ind)
        return new_subpop


    




