from utils.LearningPhase import *
from utils.Individual import *
from utils.Parameter import *
from utils.Population import *
from searcher.SSA import *
class MSHO:
    def __init__(self, name, prob, gen_length, MAX_FES, new_LS=True, BASE_POPSZ=60, BASE_rmp=0.3, update_rate = 0.06, learning=True, dynamic_pop=True, phasethree=False):
        self.name = name
        self.gnbg = prob
        self.prob = prob
        self.base_popsize = BASE_POPSZ
        self.max_popsize = BASE_POPSZ
        self.min_popsize = 20
        self.inds_tasks = self.base_popsize
        self.base_rmp = BASE_rmp
        self.gen_length = gen_length
        self.MAX_FES = MAX_FES
        self.delta = None 
        self.s_rmp = None
        self.update_rate = update_rate
        if new_LS:
            self.learningPhase = LearningPhaseILSVer2(self.prob)
        else:
            self.learningPhase = LearningPhaseILS(self.prob)
        self.learningPhase2 = LearningPhase(is_start = True, prob=self.prob) 
        self.learning = learning
        self.dynamic_pop = dynamic_pop
        # self.log = [ [] for _ in range(len(prob))]
        # self.rmp_report = []
        # self.bmb = []
        self.phasethree = phasethree
        self.mutate = Mutation(self.prob)
        # self.das_crossover = das_crossover
    def run(self, checkpoint = None):
        accept = -1
        self.prob.FE = 0
        best = []
        if checkpoint is None:
            pop = Population(self.inds_tasks, self.gen_length, self.prob)
            pop.init()
        if checkpoint is not None:
            pop = checkpoint 
        
        optimized = False
        while True:
            
            this_pop_result = pop.get_result()
            print(f'At: {self.prob.FE}, found best fitness: {self.prob.best_val}, var: {self.prob.best_val - self.prob.opt_value}')
            old_fes = self.prob.FE
            # 1. reproduction population
            offsprings = self.reproduction(pop, self.inds_tasks)
            pop.pop.extend(offsprings)
            # 2. decrease pop size
            if self.dynamic_pop:
                self.inds_tasks = int(
                    int(max((self.min_popsize - self.max_popsize) * (self.prob.FE/self.prob.max_FE) + self.max_popsize, self.min_popsize))
                )
            pop.selection_EMEBI(self.inds_tasks) 
            # 3. Employ phaseThree(like phase2 in EME-BI) 
            # and phaseTwo(local search the best individual)
            #pop = self.phaseTwo(pop)
            pop = self.phaseThree(pop)
            new_fes = self.prob.FE
            if self.prob.FE > self.prob.max_FE:
                pop.pop.sort(key = lambda ind: ind.fitness)
                print(f'After {self.prob.aceps}, found error: {self.prob.best_val - self.prob.opt_value}')
                break
            # 4. If satisfy some condition, then use ASSA
            # if self.prob.FE > 500000:
            #     pop.pop.sort(key = lambda ind: ind.fitness)
            #     assa = ASSA(n = 50, max_iters = 10000,maxFes= 500000, opt_value = self.prob.opt_value,nmin = 20, dim = self.prob.dim, lb = self.prob.LB, ub=self.prob.UB)
            #     x,y = assa.optimize(self.prob.fitness_of_ind)
        
        return self.prob.best_val - self.prob.opt_value, self.prob.aceps



            

    def reproduction(self, pop, SIZE):
        offs = []
        population = pop.pop
        terminateCondition = False
        counter = 0
        while not terminateCondition:
            idx_1, idx_2 = np.random.choice(len(population), 2)
            ind1 = population[idx_1]
            ind2 = population[idx_2]
            off1, off2 = pop.crossover(ind1, ind2)
            off1.cal_fitness(self.prob)
            off2.cal_fitness(self.prob)
            offs.extend([off1, off2])
            counter += 2
            terminateCondition = (counter >= SIZE)
        return offs

    def phaseTwo(self, pop):
        newPop = []
        # evolve(pop, DE_evals , LS_evals)
        nextPop = self.learningPhase.evolve(pop.pop, 1000, 100)
        # nextPop = self.learningPhase.evolve(pop.pop, 1000)
        newPop.extend(nextPop)
        pop.pop = newPop
        return pop
    
    def phaseThree(self, pop):
        newPop = []
        nextPop = self.learningPhase2.evolve(pop.pop)
        newPop.extend(nextPop)
        pop.pop = newPop 
        return pop
