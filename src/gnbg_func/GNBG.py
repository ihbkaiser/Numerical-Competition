import numpy as np
import scipy.io as scio
import hashlib
from functools import lru_cache 
class GNBG:
    def __init__(self, i):
        # get file path './fi.mat/' 
        if(i < 1 or i > 24):
            raise ValueError("problem index must be in [1, 24]")
        self.idx = i
        self.file_path = './gnbg_func/f' + str(i) + '.mat'
        self.GNBG = self.get_GNBG_from_path(self.file_path)
        self.LB = self.GNBG['MinCoordinate'][0][0][0][0]
        self.UB = self.GNBG['MaxCoordinate'][0][0][0][0]
        self.dim = self.GNBG['Dimension'][0][0][0][0]
        self.num_com = self.GNBG['o'][0][0][0][0]
        self.opt_value = self.GNBG['OptimumValue'][0][0][0][0]
        self.opt_position = self.GNBG['OptimumPosition'][0][0]
        self.best_val = np.inf
        self.FE = 0
        self.aceps = np.inf
        self.fitness_map = {}
        self.max_FE = 500000 if i <= 15 else 1000000
    def get_GNBG_from_path(self, file_path):
        data = scio.loadmat(file_path)
        GNBG = data['GNBG']
        # print("ID:", self.idx)
        # print("Optimal value: ", GNBG['OptimumValue'][0][0][0][0])
        # print("Num components: ", GNBG['o'][0][0][0][0])
        # print("Dimension: ", GNBG['Dimension'][0][0][0][0])
        # print("LB: ", GNBG['MinCoordinate'][0][0][0][0])
        # print("UB: ", GNBG['MaxCoordinate'][0][0][0][0])
        return GNBG
    def check(self):
        ind = self.fitness_of_ind(self.opt_position)
        val = self.opt_value
        print("Fitness of our code", ind)
        print("Fitness of the author", val)
    def fitness_of_pop(self, X):
        '''
        Input : A matrix of size (N,d), 
        N: number of individuals, d: number of dimensions, 
        kth row (1<=k<=N) represents kth individual vector.
        Output: a matrix (N,1) representing the fitness of N individuals.

        '''
        ### read the MATLAB file and call a few indices to get the same shape as the original competition file
        num_com = self.GNBG['o'][0][0][0][0]
        dim = self.GNBG['Dimension'][0][0][0][0]
        min_pos = self.GNBG['Component_MinimumPosition'][0][0].reshape(num_com, dim)
        rot_mat = self.GNBG['RotationMatrix'][0][0].reshape(dim,dim,num_com)
        sigma = self.GNBG['ComponentSigma'][0,0].reshape(num_com,1)
        hehe = self.GNBG['Component_H'][0,0].reshape(num_com,dim)
        mu = self.GNBG['Mu'][0][0].reshape(num_com,2)
        omega = self.GNBG['Omega'][0][0].reshape(num_com,4)
        lamb = self.GNBG['lambda'][0][0].reshape(num_com,1)
        SolutionNumber, _ = X.shape
        result = np.empty((SolutionNumber, 1))
        ### Converts the input vector format and returns the fitness list of individuals
        for jj in range(SolutionNumber):
            self.FE = self.FE + 1
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

    def fitness_of_ind(self,X):
        # genes_hash = hashlib.sha256(X.tobytes()).hexdigest()
        # if genes_hash in self.fitness_map:
        #     print("Hit")
        #     return self.fitness_map[genes_hash]
        
        num_com = self.GNBG['o'][0][0][0][0]
        dim = self.GNBG['Dimension'][0][0][0][0]
        min_pos = self.GNBG['Component_MinimumPosition'][0][0].reshape(num_com, dim)
        rot_mat = self.GNBG['RotationMatrix'][0][0].reshape(dim,dim,num_com)
        sigma = self.GNBG['ComponentSigma'][0,0].reshape(num_com,1)
        hehe = self.GNBG['Component_H'][0,0].reshape(num_com,dim)
        mu = self.GNBG['Mu'][0][0].reshape(num_com,2)
        omega = self.GNBG['Omega'][0][0].reshape(num_com,4)
        lamb = self.GNBG['lambda'][0][0].reshape(num_com,1)
        ### Converts the input vector format and returns the fitness list of individuals
        self.FE = self.FE + 1
        x = X.reshape(-1,1)
        f = np.empty((1, num_com))
        for k in range(num_com):
            inp = x - min_pos[k, :].reshape(-1,1)
            a = Transform( inp.T  @  rot_mat[:,:,k],mu[k, :],omega[k,:])
            b = Transform(rot_mat[:,:,k] @ inp, mu[k, :], omega[k, :] )
            f[0, k] = sigma[k] + (np.abs(a @ np.diag(hehe[k, :]) @ b))**lamb[k]

        
        result = np.min(f)
        self.best_val = min(self.best_val, result)
        # add result and corresponding genes to the map
        # genes_hash = hashlib.sha256(X.tobytes()).hexdigest()
        # self.fitness_map[genes_hash] = result
        if abs(result - self.opt_value) < 1e-8 and np.isinf(self.aceps):
            self.aceps = self.FE
            
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
        
    def fitness_mfea(self, X):
        # ##########################################################################
        # Due to float preicision, this function may not return the right value  #
        ##########################################################################
        return self.fitness_of_ind(self.LB + np.array(X)*(self.UB-self.LB))

def Transform(X, Alpha, Beta):
    Y = X.copy()
    tmp = (X > 0)
    Y[tmp] = np.log(X[tmp])
    Y[tmp] = np.exp(Y[tmp] + Alpha[0] * (np.sin(Beta[0] * Y[tmp]) + np.sin(Beta[1] * Y[tmp])))
    
    tmp = (X < 0)
    Y[tmp] = np.log(-X[tmp])
    Y[tmp] = -np.exp(Y[tmp] + Alpha[1] * (np.sin(Beta[2] * Y[tmp]) + np.sin(Beta[3] * Y[tmp])))
    
    return Y


