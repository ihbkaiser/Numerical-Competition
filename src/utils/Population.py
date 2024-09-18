from Individual import *
from Parameter import *
class Population:
    def __init__(self, SIZEPOP, SIZEGENES, prob, **kwargs):
        self.SIZEPOP = SIZEPOP
        #self.rand = rand
        self.SIZE_GENES = SIZEGENES
        self.pop = []
        self.prob = prob
        self.das_prob = None
    def init(self):
        for i in range(self.SIZEPOP):
            ind = Individual(self.SIZE_GENES, [-100,100])
            ind.init()
            ind.cal_fitness(self.prob)
            self.pop.append(ind)
    def get_subpops(self):
        return [self.pop]
    
    def get_result(self):
        result = np.inf
        for ind in self.pop:
            if ind.fitness < result:
                result = ind.fitness
        return result
    
    # def calculateD(self):
    #     subpops = self.get_subpops()
    #     D = np.zeros(len(self.prob))
    #     maxFit = np.zeros(len(self.prob))
    #     minFit = np.zeros(len(self.prob))
        
    #     for i in range(len(subpops)):
    #         subpops[i].sort(key=lambda ind: ind.fitness[i])
    #         maxFit[i] = subpops[i][-1].fitness[i]
    #         minFit[i] = subpops[i][0].fitness[i]
    #         sum_fitness = sum([ind.fitness[i] for ind in subpops[i]])
    #         for idx, ind in enumerate(subpops[i]):
    #             if(idx !=0):
    #                 wx = 1 - ind.fitness[i]/sum_fitness
    #                 distance = np.linalg.norm(ind.genes - subpops[i][0].genes)
    #                 D[i] += wx * distance
    #     return D, maxFit, minFit


    # def update_scalar_fitness(self):
    #     pop = self.pop
    #     for i, task in enumerate(self.prob):
    #         pop.sort(key = lambda ind: ind.fitness[i])
    #         for j in range(len(pop)):
    #             pop[j].rank[i]=j
    #     for ind in pop:
    #         _min = float('inf')
    #         _task = 0
    #         for task in range(1, len(self.prob)+1):
    #             if _min > ind.rank[task-1]:
    #                 _min = ind.rank[task-1]
    #                 _task = task 
    #             elif _min == ind.rank[task-1]:
    #                 if(random.random()<0.5):
    #                     _task = task
    #         ind.skill_factor = _task
    #         ind.scalar_fitness = 1.0/(_min + 1)
    def crossover(self, parent1, parent2):
        assert parent1.bounds == parent2.bounds
        bounds = parent1.bounds
        offspring = []
        p1 = parent1.genes
        p2 = parent2.genes
        p1 = (p1 - bounds[0] ) / (bounds[1] - bounds[0])
        p2 = (p2 - bounds[0] ) / (bounds[1] - bounds[0])
        D = p1.shape[0]
        cf = np.empty([D])
        u = np.random.rand(D)        

        cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (Parameter.mu + 1)))
        cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (Parameter.mu + 1)))

        c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
        c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)

        c1 = np.clip(c1, 0, 1)
        c2 = np.clip(c2, 0, 1)
        swap = True
        if swap:
            idx_swap = np.where(np.random.rand(self.SIZE_GENES) < 0.5)[0]
            c1[idx_swap], c2[idx_swap] = c2[idx_swap], c1[idx_swap]
      
        c1 = c1*(bounds[1] - bounds[0]) + bounds[0]
        c2 = c2*(bounds[1] - bounds[0]) + bounds[0]
        child1 = Individual(self.SIZE_GENES,  bounds = bounds)
        child2 = Individual(self.SIZE_GENES, bounds = bounds)
        child1.genes = c1 
        child2.genes = c2
        offspring.extend([child1, child2])
        return offspring
    #  for i in range(self.SIZE_GENES):
    #     v = 0.5 * ((1 + cf[i]) * parent1.genes[i] + (1 - cf[i]) * parent2.genes[i])
    #     v2 = 0.5 * ((1 - cf[i]) * parent1.genes[i] + (1 + cf[i]) * parent2.genes[i])

    #     child1.genes[i] = min(1, max(0, v))
    #     child2.genes[i] = min(1, max(0, v2))

    #  off_spring.extend([child1, child2])
    #  return off_spring


    def selection(self, pop_size):
        self.pop.sort(key = lambda ind: ind.scalar_fitness, reverse=True)
        size = len(self.pop)
        if len(self.pop) > pop_size:
            del self.pop[pop_size:size]
    
    def selection_EMEBI(self, pop_size):
        ################## PYMSOO IMPLEMENTATION #######################
        # subpops = self.get_subpops()
        # new_population = []
        # for i in range(len(inds_tasks)):
        #     subpop = subpops[i]
        #     subpop_factorial_rank = np.argsort(np.argsort([ind.fitness[i] for ind in subpop]))
        #     subpop_scalarfit = 1/(subpop_factorial_rank + 1) 

        #     Ni = min(inds_tasks[i], len(subpop))
    
        #     idx_selected = np.where(subpop_scalarfit > 1/(Ni + 1))[0].tolist()
        #     # remain_idx = np.where(subpop_scalarfit < 1/(Netil))[0].tolist()
        #     # idx_random = np.random.choice(remain_idx, size= (int(Ni - Netil), )).tolist()
        #     # idx_selected += idx_random
        #     new_population.extend([subpop[i] for i in idx_selected])
        # self.pop = new_population

         ##################   ANOTHER WAY TO IMPLEMENTATION ########################


        new_population = []
        N = min(pop_size, len(self.pop))
        self.pop.sort(key = lambda ind: ind.fitness)
        new_population.extend([self.pop[i] for i in range(N)])
        self.pop = new_population
        # for i in range(len(inds_tasks)):
        #     subpop = subpops[i]
        #     Ni = min(inds_tasks[i], len(subpop))
        #     subpop.sort(key = lambda ind : ind.fitness[i])
        #     new_population.extend([subpop[i] for i in range(Ni)])
        # self.pop = new_population


    # def variable_swap(self, p1, p2):
    #     ind1 = p1
    #     ind2 = p2

    #     for i in range(1, self.SIZE_GENES + 1):
    #         if random.random() > 0.5:
    #             temp1 = p1.genes[i - 1]
    #             temp2 = p2.genes[i - 1]
    #             ind1.genes[i - 1] = temp2
    #             ind2.genes[i - 1] = temp1
    # def gauss_mutation(self, ind):
    #     p = 0.01
    #     for i in range(len(ind.genes)):
    #         if random.random() < 1.0 / len(ind.genes):
    #             t = ind.genes[i] + random.gauss(0, 1)
    #             if t > 1:
    #                 t = ind.genes[i] + random.random() * (1 - ind.genes[i])
    #             elif t < 0:
    #                 t = random.random() * ind.genes[i]

    #             ind.genes[i] = t
