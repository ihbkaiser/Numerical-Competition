from searcher.MTSLS1 import *
from searcher.LBFGSB import *
from searcher.IMODE import *
from searcher.Mutation import *
from utils.SubPop import *
import random
class LearningPhase():
    M = 2   #number of operators
    def __init__(self, is_start, prob) -> None:
        self.prob = prob
        self.start = is_start
        self.searcher = [ Mutation(prob)]
    def evolve(self, subpop, divisionRate = 0.4) :
        # split subpop into sub_pools which len(sub_pools)==len(self.searcher), 
        # each sub_pool is assigned an operator
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
                sub_pools.append(SubPop(subpop[:n], self.prob))
            else:
                sub_pools.append(SubPop(subpop[int(n + (i-1)*divisionRate*len(subpop)):int(n + i*divisionRate*len(subpop))], self.prob))
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
        self.prob = prob
        self.searcher = [MTSLS1(prob), LBFGSB(prob)]
        self.explorer = IMODE(0.5, 0.5, prob)
        self.stay = 0
        self.use_restart = True
        self.grad_threshold = 30

    def evolve(self, subpop, DE_evals, LS_evals):
        ####### Explorer part (already implemented in Phase 2) ##########
        max_DE = self.prob.FE + DE_evals
        while self.prob.FE < max_DE:
            subpop = self.explorer.search(subpop)
        ################################################################
        subpop.sort(key = lambda ind: ind.fitness)
        if self.searcher[0].effective > self.searcher[1].effective:
            this_iteration_searcher = self.searcher[0]
        if self.searcher[0].effective < self.searcher[1].effective:
            this_iteration_searcher = self.searcher[1]
        if self.searcher[0].effective == self.searcher[1].effective:
            this_iteration_searcher = random.choice(self.searcher)
        subpop[0] = this_iteration_searcher.search(subpop[0], LS_evals)
        this_fitness = subpop[0].fitness
         ############ One more L-BFGS-B with random starting point ######################
        if self.use_restart:
            ind = copy.deepcopy(subpop[0])
            sol2, fit2, info2 = fmin_l_bfgs_b(self.prob.fitness_of_ind, np.random.uniform(-100, 100, len(ind.genes)), approx_grad=True, bounds=[(-100,100)]*len(ind.genes), maxfun=LS_evals, factr=10)
            if fit2 < this_fitness:
                new_ind = Individual(ind.MAX_NVARS, ind.bounds)
                new_ind.genes = sol2
                new_ind.fitness = fit2
                subpop[0] = new_ind
         ##### Check if we use restart next time or not #########
            if np.linalg.norm(info2["grad"]) > self.grad_threshold:
                self.use_restart = False
            if fit2 > this_fitness:
                disimprove_ratio = (fit2 - this_fitness) / abs(this_fitness)
                if disimprove_ratio > 5:
                    self.use_restart = False
                # print(f'Disimprove ratio: {disimprove_ratio}')
        return subpop