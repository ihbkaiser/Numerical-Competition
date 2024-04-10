from GNBG import GNBG
import numpy as np
from tqdm import tqdm
def DE(idx, MAX_FES):
    problem = GNBG(idx)
    threshold = 1e-8
    PopulationSize = 100
    LB = problem.LB 
    UB = problem.UB 
    dim = problem.dim
    X = LB + (UB - LB) * np.random.rand(PopulationSize, dim)
    fitness = problem.fitness_of_pop(X)
    Donor = np.empty((PopulationSize, dim))
    Cr = 0.9
    F = 0.5
    FEs = 0
    BestID = np.argmin(fitness)
    BestPosition = X[BestID,:]
    BestValue = fitness[BestID]
    progress = tqdm(total=MAX_FES)
    while problem.FE < MAX_FES:
        old_fe = problem.FE
        BestID = np.argmin(fitness)
        if fitness[BestID] < BestValue:
            BestPosition = X[BestID, :]
            BestValue = fitness[BestID]
        if(abs(BestValue - problem.opt_value) < threshold):
            print(f'at FE {problem.FE}, found best value : {BestValue}')
            break
        # Mutation :
        R = np.full((PopulationSize, 3), 1)
        for ii in range(PopulationSize):
            tmp = np.random.permutation(PopulationSize)
            tmp = tmp[tmp!=ii]
            R[ii,:] = tmp[:3]
        Donor = X[R[:,0],:] + F * ( X[R[:,1],:]  - X[R[:,2],:])
        # Crossover :
        OffspringPosition = X.copy()
        i = np.column_stack(((np.arange(PopulationSize)), np.random.randint(0, dim, size=PopulationSize)))
        OffspringPosition[i[:,0], i[:,1]] = Donor[i[:,0], i[:,1]]
        CrossoverBinomial = np.random.rand(PopulationSize, dim) < Cr
        OffspringPosition[CrossoverBinomial] = Donor[CrossoverBinomial]
        # Boundary Checking:
        LB_tmp1 = OffspringPosition < LB 
        UB_tmp1 = OffspringPosition > UB 
        LB_tmp2 = ((LB+X)*LB_tmp1)/2
        UB_tmp2 = ((UB+X)*UB_tmp1)/2
        OffspringPosition[LB_tmp1] = LB_tmp2[LB_tmp1]
        OffspringPosition[UB_tmp1] = UB_tmp2[UB_tmp1]
        OffspringFitness = problem.fitness_of_pop(OffspringPosition)
        better = OffspringFitness < fitness
        X[better.flatten()] = OffspringPosition[better.flatten()]
        fitness[better] = OffspringFitness[better]
        new_fe = problem.FE
        progress.update(new_fe - old_fe)
        progress.set_description(f'Best Value: {BestValue}')
        if problem.FE >= 500000:
            print(f'at FE {500000}, found best value: {BestValue}')

            break