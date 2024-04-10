import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as scio
class GNBG:
    def __init__(self, i):
        # get file path './fi.mat/' 
        self.idx = i
        self.file_path = './f' + str(i) + '.mat'
        self.GNBG = self.get_GNBG_from_path(self.file_path)
        self.LB = self.GNBG['MinCoordinate'][0][0][0][0]
        self.UB = self.GNBG['MaxCoordinate'][0][0][0][0]
        self.dim = self.GNBG['Dimension'][0][0][0][0]
        self.num_com = self.GNBG['o'][0][0][0][0]
        self.opt_value = self.GNBG['OptimumValue'][0][0][0][0]
        self.opt_position = self.GNBG['OptimumPosition'][0][0]
    def get_GNBG_from_path(self, file_path):
        data = scio.loadmat(file_path)
        GNBG = data['GNBG']
        return GNBG
    def show(self):
        print("ID:", self.idx)
        print("Optimal value: ", self.opt_value)
        print("Num components: ", self.num_com)
        print("Dimension: ", self.dim)
        print("LB: ", self.LB)
        print("UB: ", self.UB)
        
    def check(self):
        ind = self.fitness_of_ind(self.opt_position)
        val = self.opt_value
        print("Fitness of our code", ind)
        print("Fitness of the author", val)
    def fitness_of_pop(self, X):
        return fitness_of_pop(X, self.GNBG)
    def fitness_of_ind(self, X):
        return fitness_of_ind(X, self.GNBG)
    def fitness_mfea(self, X):
        return fitness_of_ind(self.LB + np.array(X)*(self.UB-self.LB), self.GNBG)

import numpy as np

def fitness_of_pop(X, GNBG):  
    '''
    Input : A matrix of size (N,d), 
    N: number of individuals, d: number of dimensions, 
    kth row (1<=k<=N) represents kth individual vector.
    Output: a matrix (N,1) representing the fitness of N individuals.

    '''
    ### read the MATLAB file and call a few indices to get the same shape as the original competition file
    num_com = GNBG['o'][0][0][0][0]
    dim = GNBG['Dimension'][0][0][0][0]
    min_pos = GNBG['Component_MinimumPosition'][0][0].reshape(num_com, dim)
    rot_mat = GNBG['RotationMatrix'][0][0].reshape(dim,dim,num_com)
    sigma = GNBG['ComponentSigma'][0,0].reshape(num_com,1)
    hehe = GNBG['Component_H'][0,0].reshape(num_com,dim)
    mu = GNBG['Mu'][0][0].reshape(num_com,2)
    omega = GNBG['Omega'][0][0].reshape(num_com,4)
    lamb = GNBG['lambda'][0][0].reshape(num_com,1)
    SolutionNumber, _ = X.shape
    result = np.empty((SolutionNumber, 1))
    ### Converts the input vector format and returns the fitness list of individuals
    for jj in range(SolutionNumber):
        x = X[jj, :].reshape(-1,1)
        f = np.empty((1, num_com))
        
        for k in range(num_com):
            inp = x - min_pos[k,:].reshape(-1,1)
            a = Transform( inp.T  @  rot_mat[:,:,k] , mu[k,:], omega[k,:])
            b = Transform(rot_mat[:,:,k] @ inp, mu[k,:], omega[k,:] )
            f[0, k] = sigma[k] + (np.abs(a @ np.diag(hehe[k, :]) @ b))**lamb[k]

        result[jj] = np.min(f)
        
        # if GNBG['FE'] > GNBG['MaxEvals']:
        #     return result, GNBG
        
        # GNBG['FE'] += 1
        # GNBG['FEhistory'][GNBG['FE']] = result[jj]
        
        # if GNBG['BestFoundResult'] > result[jj]:
        #     GNBG['BestFoundResult'] = result[jj]
            
        # if (abs(GNBG['FEhistory'][GNBG['FE']] - GNBG['OptimumValue']) < GNBG['AcceptanceThreshold'] and
        #         np.isinf(GNBG['AcceptanceReachPoint'])):
        #     GNBG['AcceptanceReachPoint'] = GNBG['FE']
    
    return result

def Transform(X, Alpha, Beta):
    Y = X.copy()
    tmp = (X > 0)
    Y[tmp] = np.log(X[tmp])
    Y[tmp] = np.exp(Y[tmp] + Alpha[0] * (np.sin(Beta[0] * Y[tmp]) + np.sin(Beta[1] * Y[tmp])))
    
    tmp = (X < 0)
    Y[tmp] = np.log(-X[tmp])
    Y[tmp] = -np.exp(Y[tmp] + Alpha[1] * (np.sin(Beta[2] * Y[tmp]) + np.sin(Beta[3] * Y[tmp])))
    
    return Y

def fitness_of_ind(X,GNBG):
    ### read the MATLAB file and call a few indices to get the same shape as the original competition file
    num_com = GNBG['o'][0][0][0][0]
    dim = GNBG['Dimension'][0][0][0][0]
    min_pos = GNBG['Component_MinimumPosition'][0][0].reshape(num_com, dim)
    rot_mat = GNBG['RotationMatrix'][0][0].reshape(dim,dim,num_com)
    sigma = GNBG['ComponentSigma'][0,0].reshape(num_com,1)
    hehe = GNBG['Component_H'][0,0].reshape(num_com,dim)
    mu = GNBG['Mu'][0][0].reshape(num_com,2)
    omega = GNBG['Omega'][0][0].reshape(num_com,4)
    lamb = GNBG['lambda'][0][0].reshape(num_com,1)
    ### Converts the input vector format and returns the fitness list of individuals
    x = X.reshape(-1,1)
    f = np.empty((1, num_com))
    for k in range(num_com):
        inp = x - min_pos[k, :].reshape(-1,1)
        a = Transform( inp.T  @  rot_mat[:,:,k],mu[k, :],omega[k,:])
        b = Transform(rot_mat[:,:,k] @ inp, mu[k, :], omega[k, :] )
        f[0, k] = sigma[k] + (np.abs(a @ np.diag(hehe[k, :]) @ b))**lamb[k]


    result = np.min(f)
        
        # if GNBG['FE'] > GNBG['MaxEvals']:
        #     return result, GNBG
        
        # GNBG['FE'] += 1
        # GNBG['FEhistory'][GNBG['FE']] = result[jj]
        
        # if GNBG['BestFoundResult'] > result[jj]:
        #     GNBG['BestFoundResult'] = result[jj]
            
        # if (abs(GNBG['FEhistory'][GNBG['FE']] - GNBG['OptimumValue']) < GNBG['AcceptanceThreshold'] and
        #         np.isinf(GNBG['AcceptanceReachPoint'])):
        #     GNBG['AcceptanceReachPoint'] = GNBG['FE']
    
    return result
    

def khoitao(N, dim, ub, lb):
    return (ub - lb) * np.random.rand(N, dim) + lb
N = 100  
Max_iter = 1000
for idx in range(7, 25):
    problem = GNBG(idx)
    problem.show()
    problem.check()
    hammuctieu = problem.fitness_of_ind
    lb = problem.LB
    ub = problem.UB
    dim = problem.dim
    def SMA(N, Max_iter, lb, ub, dim, hammuctieu):
        print('Vui lòng chờ')
        bestPositions = np.zeros(dim)
        Destination_fitness = np.inf  
        fitness_list = np.ones(N)  
        weight = np.ones((N, dim))  
        X = khoitao(N, dim, ub, lb)
        Ve = np.zeros(Max_iter)
        it = 1  
        lb = np.ones(dim) * lb  
        ub = np.ones(dim) * ub  
        z = 0.03  
        with tqdm(total=Max_iter, desc="Tiến trình") as pbar:
            while it <= Max_iter:
                for i in range(N):
                    for j in range(dim):
                        if(X[i][j] > ub[0]):
                            X[i][j] = ub[0]
                        if(X[i][j] < lb[0]):
                            X[i][j] = lb[0]
                    fitness_list[i] = hammuctieu(X[i])
                SmellIndex = np.argsort(fitness_list)
                worstFitness = fitness_list[SmellIndex[-1]]
                bestFitness = fitness_list[SmellIndex[0]]
                S = bestFitness - worstFitness + np.finfo(float).eps  
                for i in range(N):
                    for j in range(dim):
                        if i <= (N / 2):
                            weight[SmellIndex[i], j] = 1 + np.random.rand() * np.log10((bestFitness - fitness_list[i]) / S + 1)
                        else:
                            weight[SmellIndex[i], j] = 1 - np.random.rand() * np.log10((bestFitness - fitness_list[i]) / S + 1)
                if bestFitness < Destination_fitness:
                    bestPositions = np.copy(X[SmellIndex[0]])
                    Destination_fitness = bestFitness
                a = np.arctanh(-(it / Max_iter) + 1)
                b = 1 - it / Max_iter
                for i in range(N):
                    if np.random.rand() < z:
                        X[i] = (ub - lb) * np.random.rand(dim) + lb
                    else:
                        p = np.tanh(abs(fitness_list[i] - Destination_fitness))
                        vb = np.random.uniform(-a, a, dim)
                        vc = np.random.uniform(-b, b, dim)
                        for j in range(dim):
                            r = np.random.rand()
                            A, B = np.random.choice(N, 2)
                            if r < p:
                                X[i, j] = bestPositions[j] + vb[j] * (weight[i, j] * X[A, j] - X[B, j])
                            else:
                                X[i, j] = vc[j] * X[i, j]
                Ve[it - 1] = Destination_fitness
                it += 1
                pbar.update(1)
        return Destination_fitness, bestPositions, Ve
    Destination_fitness, bestPositions, Ve = SMA(N, Max_iter, lb, ub, dim, hammuctieu)
    print("Giá trị nhỏ nhất là:", Destination_fitness)
    print("Đạt được tại", bestPositions)
# if hammuctieu(bestPositions) == Destination_fitness:
#     print("Thuật toán chạy đúng")
# else:
#     print("Thuật toán chạy sai")
# plt.plot(range(Max_iter), Ve, marker='o', linestyle='-', color = 'r')
# plt.title('Đường hội tụ')
# plt.xlabel('Vòng lặp')
# plt.ylabel('Giá trị hàm mục tiêu')
# plt.grid(True)
# plt.show()