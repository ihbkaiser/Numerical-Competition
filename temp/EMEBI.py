from MFEAcode import *
from GNBG import GNBG
from tqdm import tqdm
from scipy.stats import cauchy
from Searcher import *
from Learning import *
import random
class EMEBI:
    def __init__(self, name, prob, gen_length, MAX_FES, BASE_POPSZ=100, BASE_rmp=0.3, update_rate = 0.06, learning=True, dynamic_pop=True, phasethree=False, das_crossover=False):
        self.name = name
        self.prob = prob
        self.base_popsize = BASE_POPSZ
        self.max_popsize = BASE_POPSZ
        self.min_popsize = int(BASE_POPSZ/5)
        self.inds_tasks = [self.base_popsize] * len(prob)
        self.base_rmp = BASE_rmp
        self.gen_length = gen_length
        self.MAX_FES = MAX_FES
        self.stay = 0
        self.delta = None 
        self.s_rmp = None
        self.rmp = np.full((len(prob), len(prob)), BASE_rmp)
        self.update_rate = update_rate
        self.learningPhase = [LearningPhaseILS(prob) for _ in range(len(prob))]
        self.learningPhase2 = [LearningPhase(is_start = True, prob=prob) for _ in range(len(prob))]
        self.learning = learning
        self.dynamic_pop = dynamic_pop
        self.log = [ [] for _ in range(len(prob))]
        self.rmp_report = []
        self.bmb = []
        self.phasethree = phasethree
        self.das_crossover = das_crossover
    def run(self, checkpoint = None):
        stay = 0
        np.fill_diagonal(self.rmp, 0)
        best = []
        if checkpoint is None:
            pop = Population(sum(self.inds_tasks), self.gen_length, self.prob)
            pop.init(opposition=True)
            pop.update_scalar_fitness()
            D0, _, _ = pop.calculateD()
        if checkpoint is not None:
            pop = checkpoint 
            pop.update_scalar_fitness()
            D0, _, _ = pop.calculateD()
        this_pop_result = pop.get_result()
        for i in range(len(self.prob)):
            self.log[i].append([Parameter.FEl[i], this_pop_result[i]])
        progress_bar = tqdm(total=self.MAX_FES, unit = 'FEs', unit_scale = True)
        MFEAc = MFEA(self.prob, 100, None, 0.3, 30, 100000)
        former_FEs = Parameter.FEs
        while True:
            


            old_fes = Parameter.FEs
                 ########################  PHASE 1 - EMEBI ############################
            self.delta = [[[] for _ in range(len(self.prob))] for _ in range(len(self.prob))]
            self.s_rmp = [[[] for _ in range(len(self.prob))] for _ in range(len(self.prob))]
            offsprings = self.reproduction(pop, sum(self.inds_tasks))
            pop.pop.extend(offsprings)
            if self.dynamic_pop:
                self.inds_tasks = [int(
                    int(max((self.min_popsize - self.max_popsize) * ((Parameter.FEs - former_FEs)/(self.MAX_FES - former_FEs)) + self.max_popsize, self.min_popsize))
                )] * len(self.prob)
            pop.selection_EMEBI(self.inds_tasks)  #each skill-factor choose some best
            # pop.update_scalar_fitness()                 # choose best of the whole pop using scalar-fitness
            # pop.selection(sum(self.inds_tasks))
                  
            self.updateRMP(self.update_rate)
               #########################  PHASE 1 - MFEA2 ############################
            # offsprings, sample_rmp = MFEAc.reproduction_MFEA_2(pop = pop, SIZE = sum(self.inds_tasks))
            # self.rmp_report.append(sample_rmp)
            # pop.pop.extend(offsprings)
            # if self.dynamic_pop:
            #     self.inds_tasks = [int(
            #         int(max((self.min_popsize - self.max_popsize) * (Parameter.FEs/self.MAX_FES) + self.max_popsize, self.min_popsize))
            #     )] * len(self.prob)
            # pop.selection(sum(self.inds_tasks))
               ###########################  PHASE 2 - LEARNING ############################
            if self.learning:
                self.phaseThree(D0, pop)
            if self.phasethree:
                self.phaseTwo(D0, pop)
            new_fes = Parameter.FEs
            
            if Parameter.FEs <= self.MAX_FES:
                this_pop_result = pop.get_result()
                for i in range(len(self.prob)):
                    self.log[i].append([Parameter.FEl[i], this_pop_result[i]])
                result = "Current fitness: "
                for i in range(len(self.prob)):
                    result += f"{i+1}: {this_pop_result[i]} "
                progress_bar.set_description(result)
                progress_bar.update(new_fes - old_fes)
            else:
                break
        return self.log, pop
    def restart(self, pop):
        self.inds_tasks = [self.base_popsize] * len(self.prob)
            


    def reproduction(self, pop, SIZE):
        sub_size = int(SIZE/len(self.prob))
        offs = []
        population = pop.pop
        counter = np.zeros((len(self.prob)))
        terminateCondition = False
        while not terminateCondition:
            idx_1, idx_2 = np.random.choice(len(population), 2)
            ind1 = population[idx_1]
            ind2 = population[idx_2]
            t1 = ind1.skill_factor
            t2 = ind2.skill_factor
            if counter[t1-1] >= sub_size and counter[t2-1] >= sub_size:
                continue 
            rmpValue = np.random.normal(loc = max(self.rmp[t1-1][t2-1], self.rmp[t2-1][t1-1]), scale = 0.1)
            if t1 == t2:
                if not self.das_crossover:
                    off1, off2 = pop.crossover(ind1, ind2)
                    off1.skill_factor = t1
                    off2.skill_factor = t2
                if self.das_crossover:
                    off1, off2 = pop.das_crossover(ind1, ind2, t1, t2)
                off1.cal_fitness(self.prob)
                off2.cal_fitness(self.prob)
                offs.extend([off1, off2])
                counter[t1-1] += 2
                # ### Choose one from the two crossovers below, on your choice ###
                # ################  Normal crossover ###############
                # # off1, off2 = pop.crossover(ind1, ind2)
                # # off1.skill_factor = t1
                # # off2.skill_factor = t2
                # ############### Das crossover ###################
                # off1, off2 = pop.das_crossover(ind1, ind2, t1, t2)
                # off1.cal_fitness(self.prob)
                # off2.cal_fitness(self.prob)
                # offs.extend([off1, off2])
                # counter[t1-1] += 2
            elif random.random() < rmpValue:
                if not self.das_crossover:
                    this_offs = pop.crossover(ind1, ind2)
                if self.das_crossover:
                    this_offs = pop.das_crossover(ind1, ind2, t1, t2)

                for off in this_offs:
                    if counter[t1-1] < sub_size and random.random() < self.rmp[t1-1][t2-1] / (self.rmp[t1-1][t2-1] + self.rmp[t2-1][t1-1]):
                        off.skill_factor = t1
                        off.cal_fitness(self.prob)
                      
                        offs.append(off)
                        counter[t1-1] += 1
                        if ind1.fitness[t1-1] > off.fitness[t1-1]:
                            self.bmb.append(Parameter.FEs)
                            self.delta[t1-1][t2-1].append(ind1.fitness[t1-1] - off.fitness[t1-1])
                            self.s_rmp[t1-1][t2-1].append(rmpValue)
    
                    elif counter[t2-1]< sub_size:
                        off.skill_factor = t2 
                        off.cal_fitness(self.prob)
                    
                     
                        offs.append(off)
                        counter[t2-1] +=1
                        if ind2.fitness[t2-1] > off.fitness[t2-1]:
                            self.bmb.append(Parameter.FEs)
                            self.delta[t2-1][t1-1].append(ind2.fitness[t2-1] - off.fitness[t2-1])
                            self.s_rmp[t2-1][t1-1].append(rmpValue)
            else:
                if counter[t1 - 1] < sub_size:
                            sf_list = np.array( [pop.skill_factor for pop in population] )
                           
                            idx_same_sf = np.where(sf_list == t1)[0]
                   
                            random_idx = np.random.choice(idx_same_sf)
                            ind11 = population[random_idx]
                            assert ind11.skill_factor == t1
                            off1, _ = pop.crossover(ind1, ind11)
                            off1.skill_factor = t1
                            off1.cal_fitness(self.prob)
                            offs.append(off1)
                            counter[t1-1] += 1
                if counter[t2 - 1] < sub_size:
                            sf_list = np.array( [pop.skill_factor for pop in population] )
                            idx_same_sf = np.where(sf_list == t2)[0]
                            random_idx = np.random.choice(idx_same_sf)
                            ind22 = population[random_idx]
                            assert ind22.skill_factor == t2
                            _, off2 = pop.crossover(ind2, ind22)
                            off2.skill_factor = t2
                            off2.cal_fitness(self.prob)
                            offs.append(off2)
                            counter[t2-1] += 1
            terminateCondition = sum(counter >= sub_size) == len(self.prob)
        return offs
    def updateRMP(self, update_rate):
            for i in range(len(self.prob)):
                for j in range(len(self.prob)):
                    if i==j:
                        continue 
                    if len(self.delta[i][j]) > 0:
                        self.rmp[i][j] += update_rate * Lehmer_mean(self.delta[i][j], self.s_rmp[i][j])
                    else:
                        self.rmp[i][j] = (1-update_rate)*self.rmp[i][j]
                    self.rmp[i][j] = max(0.1, min(1, self.rmp[i][j]))
    # def phaseTwo(self, D0, pop):
    #     D, maxFit, minFit = pop.calculateD()
    #     maxDelta = maxFit - minFit + 1e-99
    #     sigma = np.where(D > D0 , 0 , 1 - D/D0)
    #     newPop = []
    #     subpops = pop.get_subpops()
    #     for i in range(len(self.prob)):
    #         nextPop = self.learningPhase[i].evolve(subpops[i], sigma[i], maxDelta[i], divisionRate =0.4)
    #         newPop.extend(nextPop)
    #         self.learningPhase[i].start = False
    #     pop.pop = newPop

    def phaseTwo(self, D0, pop):
        newPop = []
        subpops = pop.get_subpops()
        for i in range(len(self.prob)):
            nextPop = self.learningPhase[i].evolve(subpops[i], 1000, 1000)
            newPop.extend(nextPop)
        pop.pop = newPop
    
    def phaseThree(self, D0, pop):
        D, maxFit, minFit = pop.calculateD()
        maxDelta = maxFit - minFit + 1e-99
        sigma = np.where(D > D0 , 0 , 1 - D/D0)
        newPop = []
        subpops = pop.get_subpops()
        for i in range(len(self.prob)):
            nextPop = self.learningPhase2[i].evolve(subpops[i], sigma[i], maxDelta[i], divisionRate =0.4)
            newPop.extend(nextPop)
            self.learningPhase[i].start = False
        pop.pop = newPop

        

def Lehmer_mean(delta, s_rmp):
        delta = np.array(delta)
        s_rmp = np.array(s_rmp)
        sum_delta = sum(delta)
        tmp = (delta/sum_delta) * s_rmp
        meanS = sum(tmp * s_rmp)
        return meanS/sum(tmp)








