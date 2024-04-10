import random
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
class Parameter:
    reps = 30
    maxFEs = 1000000
    numRecords = 1000
    SUBPOPULATION = 100
    SIZE_POPULATION = 1000
    MAX_GENERATION = 1000

    mum = 5
    mu = 2

    rmp = 0.7
    pc = 0.8
    pm = 0.3

    num_fitness = 0
    countFitness = None
    o_rmp = None  
    FEs = 0

    @staticmethod
    def initialize_o_rmp():

        pass

class Individual:
    def __init__(self, MAX_NVARS, MAX_OBJS):
        self.MAX_NVARS = MAX_NVARS
        self.MAX_OBJS = MAX_OBJS
        self.genes = [0.0] * MAX_NVARS
        self.fitness = [float('inf')] * MAX_OBJS
        self.rank = [0] * MAX_OBJS
        self.skill_factor = 0
        self.scalar_fitness = 0.0

    def init(self):
        for i in range(self.MAX_NVARS):
            self.genes[i] = random.random()

    def compare(self, another_individual):
        if(self.scalar_fitness > another_individual.scalar_fitness):
            return -1
        elif(self.scalar_fitness < another_individual.scalar_fitness):
            return 1
        else:
            return 0
class Population:
    def __init__(self, SIZEPOP, SIZEGENES, func_list, **kwargs):
        self.SIZEPOP = SIZEPOP
        #self.rand = rand
        self.NUM_TASKS = len(func_list)
        self.SIZE_GENES = SIZEGENES
        self.pop = []
        self.prob = func_list
    def init(self):
        for i in range(self.SIZEPOP):
            ind = Individual(self.SIZE_GENES, self.NUM_TASKS)
            ind.init()
            for t in range(1, self.NUM_TASKS + 1):
                ind.fitness[t - 1] = self.prob[t-1](ind.genes)
                Parameter.FEs = Parameter.FEs + 1
            self.pop.append(ind)
    def update_scalar_fitness(self):
        pop = self.pop
        for i, task in enumerate(self.prob):
            pop.sort(key = lambda ind: ind.fitness[i])
            for j in range(len(pop)):
                pop[j].rank[i]=j
        for ind in pop:
            _min = float('inf')
            _task = 0
            for task in range(1, len(self.prob)+1):
                if _min > ind.rank[task-1]:
                    _min = ind.rank[task-1]
                    _task = task 
                elif _min == ind.rank[task-1]:
                    if(random.random()<0.5):
                        _task = task
            ind.skill_factor = _task
            ind.scalar_fitness = 1.0/(_min + 1)
    def crossover(self, parent1, parent2):
     off_spring = []
     cf = [1.0 for _ in range(self.SIZE_GENES)]
     for i in range(self.SIZE_GENES):
        u = random.random()
        if u <= 0.5:
            cf[i] = (2 * u) ** (1.0 / (Parameter.mu + 1))
        else:
            cf[i] = (2 * (1 - u)) ** (-1.0 / (Parameter.mu + 1))

     child1 = Individual(self.SIZE_GENES,  self.NUM_TASKS)
     child2 = Individual(self.SIZE_GENES, self.NUM_TASKS)
     for i in range(self.SIZE_GENES):
        v = 0.5 * ((1 + cf[i]) * parent1.genes[i] + (1 - cf[i]) * parent2.genes[i])
        v2 = 0.5 * ((1 - cf[i]) * parent1.genes[i] + (1 + cf[i]) * parent2.genes[i])

        child1.genes[i] = min(1, max(0, v))
        child2.genes[i] = min(1, max(0, v2))

     off_spring.extend([child1, child2])
     return off_spring
    def mutation(self, parent):
        ind = Individual(self.SIZE_GENES, self.NUM_TASKS)
        ind.genes = list(parent.genes)

        for i in range(1, self.SIZE_GENES + 1):
            if random.random() < 1.0 / self.SIZE_GENES:
                u = random.random()
                if u <= 0.5:
                    del_val = (2 * u) ** (1.0 / (1 + Parameter.mum)) - 1
                    ind.genes[i - 1] = ind.genes[i - 1] * (del_val + 1)
                else:
                    del_val = 1 - (2 * (1 - u)) ** (1.0 / (1 + Parameter.mum))
                    ind.genes[i - 1] = ind.genes[i - 1] + del_val * (1 - ind.genes[i - 1])

            if ind.genes[i - 1] > 1:
                ind.genes[i - 1] = parent.genes[i] + random.random() * (1 - parent.genes[i])
            elif ind.genes[i - 1] < 0:
                ind.genes[i - 1] = parent.genes[i] * random.random()

        return ind
    def direct_mutation(self, parent):
        ind = parent
        for i in range(1, self.SIZE_GENES + 1):
            if random.random() < 1.0 / self.SIZE_GENES:
                u = random.random()
                v = 0
                if u <= 0.5:
                    del_val = (2 * u) ** (1.0 / (1 + Parameter.mum)) - 1
                    v = ind.genes[i - 1] * (del_val + 1)
                else:
                    del_val = 1 - (2 * (1 - u)) ** (1.0 / (1 + Parameter.mum))
                    v = ind.genes[i - 1] + del_val * (1 - ind.genes[i - 1])

                if v > 1:
                    ind.genes[i - 1] = ind.genes[i - 1] + random.random() * (1 - ind.genes[i - 1])
                elif v < 0:
                    ind.genes[i - 1] = ind.genes[i - 1] * random.random()
    def selection(self, pop_size):
        self.pop.sort(key = lambda ind: ind.scalar_fitness, reverse=True)
        size = len(self.pop)
        if len(self.pop) > pop_size:
            del self.pop[pop_size:size]

    def variable_swap(self, p1, p2):
        ind1 = p1
        ind2 = p2

        for i in range(1, self.SIZE_GENES + 1):
            if random.random() > 0.5:
                temp1 = p1.genes[i - 1]
                temp2 = p2.genes[i - 1]
                ind1.genes[i - 1] = temp2
                ind2.genes[i - 1] = temp1
    def gauss_mutation(self, ind):
        p = 0.01
        for i in range(len(ind.genes)):
            if random.random() < 1.0 / len(ind.genes):
                t = ind.genes[i] + random.gauss(0, 1)
                if t > 1:
                    t = ind.genes[i] + random.random() * (1 - ind.genes[i])
                elif t < 0:
                    t = random.random() * ind.genes[i]

                ind.genes[i] = t

class MFEA:
    def __init__(self, prob, POPSIZE, MAX_GEN, rmp, gen_length, MAX_FES):
        self.prob = prob
        self.rmp = rmp
        self.POPSIZE = POPSIZE
        #self.maxEvals = maxEvals
        self.MAX_GEN = MAX_GEN
        self.gen_length = gen_length
        self.MAX_FES = MAX_FES

    def run(self):
        Parameter.FEs = 0

        best = []
        # data = np.ones((self.MAX_GEN+1 , len(self.prob)))
        pop = Population(self.POPSIZE, self.gen_length, self.prob)
        pop.init()
        pop.update_scalar_fitness()
        pop.selection(self.POPSIZE)
        generation = 0

        for i in range(1, len(self.prob) + 1):
            for ind in pop.pop:
                if ind.skill_factor == i:
                    # data[0][i - 1] = ind.fitness[i - 1]
                    best.append(ind)
                    break
        while(Parameter.FEs < self.MAX_FES):
            # Repopulation
            print("Current FEs: ", Parameter.FEs)
            offs = self.reproduction(pop, self.POPSIZE)
            pop.pop.extend(offs)
            pop.update_scalar_fitness()
            pop.selection(self.POPSIZE)

            for i in range(1, len(self.prob) + 1):
                for ind in pop.pop:
                    if ind.skill_factor == i:
                        if best[i - 1].fitness[i - 1] > ind.fitness[i - 1]:
                            best[i - 1] = ind
                        # data[generation][i - 1] = best[i - 1].fitness[i - 1]
                        break
        for i in range(1, len(self.prob) + 1):
            print(f"Final result: {best[i - 1].skill_factor}: {best[i - 1].fitness[i - 1]}")
        
        # for i in range(1, len(self.prob) + 1):
        #     plt.plot(data[:,i-1])
        #     plt.title(f'Task {i}')
        #     plt.xlabel('Generation')
        #     plt.ylabel('Fitness')
        #     plt.grid(True)
        #     plt.show()

      
        

       
        return best
    def reproduction(self, pop, SIZE):
        offs = []
        lst = list(range(len(pop.pop)))
        random.shuffle(lst)
        
        while len(offs) < SIZE:
            a = lst[random.randint(0, len(pop.pop) // 2)]
            b = lst[random.randint(len(pop.pop) // 2, len(pop.pop) - 1)]
            parent1 = pop.pop[a]
            parent2 = pop.pop[b]
            child = []

            if parent1.skill_factor == parent2.skill_factor or random.random() < self.rmp:
                child = pop.crossover(parent1, parent2)
                for ind in child:
                    ind.skill_factor = parent1.skill_factor if random.random() > 0.5 else parent2.skill_factor
            else:
                ind1 = pop.mutation(parent1)
                ind1.skill_factor = parent1.skill_factor
                ind2 = pop.mutation(parent2)
                ind2.skill_factor = parent2.skill_factor
                child.extend([ind2, ind1])

            offs.extend(child)

        for ind in offs:
            for i in range(1, len(self.prob) + 1):
                if i == ind.skill_factor:
                    ind.fitness[i - 1] = self.prob[i - 1](ind.genes)
                    Parameter.FEs = Parameter.FEs + 1
                else:
                    ind.fitness[i - 1] = float('inf')

        return offs
