class LearningPhase():
    M = 2
    H = 10
    def __init__(self, dim, list_tasks, task) -> None:
        self.list_tasks = list_tasks
        self.num_task = len(list_tasks)
        self.dim = dim
        self.pm = 1/self.dim
        self.task = task
        # self.sum_improv = np.zeros((LearningPhase.M))
        # self.consume_fes = np.ones((LearningPhase.M))
        # self.mem_cr = np.full((LearningPhase.H), 0.5)
        # self.mem_f = np.full((LearningPhase.H), 0.5)
        self.sum_improv = [0] * LearningPhase.M
        self.consume_fes = [0] * LearningPhase.M
        self.mem_cr = [0.5] * LearningPhase.H
        self.mem_f = [0.5] * LearningPhase.H
        self.s_cr = []
        self.s_f = []
        self.diff_f = []
        self.mem_pos = 0
        self.gen = 0
        self.best_opcode = 1
        self.scale = 0.1
        self.searcher = [self.pbest1,self.mutation]

    def mutation(self, parent: Individual, return_newInd:bool) -> Individual:
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
        ind = Individual(self.SIZE_GENES, self.NUM_TASKS)
        ind.genes = tmp
        ind.skill_factor = parent.skill_factor
        ind.cal_fitness(self.list_tasks)
        return ind
    def pbest1(self, ind, subPop, best, cr: float, f: float):
        pbest = best[random.randint(0, len(best) - 1)]
        
        rand_idx = np.random.choice(len(subPop), 2, replace=False)
        ind_ran1, ind_ran2 = subPop[rand_idx[0]], subPop[rand_idx[1]]
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (subPop.dim,))
            u[np.random.choice(subPop.dim)] = 1

        new_genes = np.where(u, 
            pbest.genes + f * (ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        new_genes = np.where(new_genes < 0, ind.genes/2, np.where(new_genes > 1, (ind.genes + 1)/2, new_genes))
        new_ind = Individual(self.dim, self.num_task)
        new_ind.genes = new_genes

        return new_ind


    def evolve(self, subPop, sigma: float, max_delta: float) :
        eval_k = 0
        
        self.gen += 1
        if self.gen > 1:
            self.best_opcode = self.__class__.updateOperator(sum_improve = self.sum_improv, 
                                                             consume_fes = self.consume_fes, 
                                                             M = LearningPhase.M)

            # self.sum_improv = np.zeros((LearningPhase.M))
            # self.consume_fes = np.ones((LearningPhase.M))
            self.sum_improv = [0.0] * LearningPhase.M
            self.consume_fes = [1.0] * LearningPhase.M

        self.updateMemory()
        
        pbest_size = max(5, int(0.15 * len(subPop)))
        idx = np.random.choice(len(subPop), pbest_size, replace=False)
        pbest = [subPop[i] for i in idx]

        for ind in subPop:
            r = random.randint(0, LearningPhase.M - 1)
            cr = np.random.normal(self.mem_cr[r], 0.1)
            f = np.random.cauchy(self.mem_f[r], 0.1)
                        
            opcode = random.randint(0, LearningPhase.M)
            if opcode == LearningPhase.M:
                opcode = self.best_opcode
            
            if opcode == 0:
                child = self.searcher[opcode](ind, subPop, pbest, cr, f)
            elif opcode == 1:
                child = self.searcher[opcode](ind, return_newInd=True)

            child.skill_factor = ind.skill_factor
            child.cal_fitness(self.list_tasks)
            
            eval_k += 1
            
            diff = ind.fitness[ind.skill_factor-1] - child.fitness[child.skill_factor-1]
            if diff > 0:
                survival = child

                self.sum_improv[opcode] += diff

                if opcode == 0:
                    self.diff_f.append(diff)
                    self.s_cr.append(cr)
                    self.s_f.append(f)
                
            elif diff == 0 or random.random() <= sigma * np.exp(diff/max_delta):
                survival = child
            else:
                survival = ind
            
            nextPop.pop.append(survival)
        
        return nextPop, eval_k
    
    def pbest1(self, ind, subPop, best, cr: float, f: float):
        pbest = best[random.randint(0, len(best) - 1)]
        
        rand_idx = np.random.choice(len(subPop), 2, replace=False)
        ind_ran1, ind_ran2 = subPop[rand_idx[0]], subPop[rand_idx[1]]
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (subPop.dim,))
            u[np.random.choice(subPop.dim)] = 1

        new_genes = np.where(u, 
            pbest.genes + f * (ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        new_genes = np.where(new_genes < 0, ind.genes/2, np.where(new_genes > 1, (ind.genes + 1)/2, new_genes))
        new_ind = Individual(self.dim, self.num_task)
        new_ind.genes = new_genes

        return new_ind


    def updateMemory(self):
        if len(self.s_cr) > 0:
        
            self.mem_cr[self.mem_pos] = self.__class__.updateMemoryCR(self.diff_f, self.s_cr)
            self.mem_f[self.mem_pos] = self.__class__.updateMemoryF(self.diff_f, self.s_f)
            
            self.mem_pos = (self.mem_pos + 1) % LearningPhase.H

            self.s_cr = []
            self.s_f = []
            self.diff_f = []

    def updateMemoryCR(diff_f: List, s_cr: List) -> float:
        diff_f = np.array(diff_f)
        s_cr = np.array(s_cr)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tmp_sum_cr = sum(weight * s_cr)
        mem_cr = sum(weight * s_cr * s_cr)
        
        if tmp_sum_cr == 0 or mem_cr == -1:
            return -1
        else:
            return mem_cr/tmp_sum_cr
        
    def updateMemoryF(diff_f: List, s_f: List) -> float:
        diff_f = np.array(diff_f)
        s_f = np.array(s_f)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tmp_sum_f = sum(weight * s_f)
        return sum(weight * (s_f ** 2)) / tmp_sum_f

  
    def updateOperator(sum_improve: List, consume_fes: List, M: int) -> int:
        sum_improve = np.array(sum_improve)
        consume_fes = np.array(consume_fes)
        eta = sum_improve / consume_fes
        best_rate = max(eta)
        best_op = np.argmax(eta)
        if best_rate > 0:
            return best_op
        else:
            return random.randint(0, M - 1)