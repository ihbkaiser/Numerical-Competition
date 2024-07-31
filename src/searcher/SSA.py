import numpy as np
import copy, math

class Salp(object):
    def __init__(self):
        self.X = []
        self.fitness = np.inf
    
    # def __repr__(self):
    #     return str(list(self.X))
    
    def getX(self):
        return self.X
    
    def setX(self , X):
        self.X = X

    def get_fitness(self):
        return self.fitness
    
    def set_fitness(self, f):
        self.fitness = f

class ASSA(object):
    def __init__(self, n = 50, max_iters = 10000, seed = None, former_leader=None, verbose = True, opt_value = 0, maxFes = 500000,
                 nmin = 20, dim = 30, lb = -1, ub = 1):
        if seed is not None:
            np.random.seed(seed)
        self.dim = dim
        self.former_leader = former_leader
        self.lb = lb
        self.ub = ub
        self._F = None
        self.c1 = 1
        self.n = n
        self.nmax = n 
        self.nmin = nmin
        self.max_iters = max_iters
        self.iteration = 1
        self.Salps = []
        self.verbose = verbose
        self.opt_value = opt_value
        self.FEs = 0
        self.maxFEs = maxFes
        self.funct = None

    def _initialise(self):
        return np.random.uniform(0,1,self.dim)*(self.ub - self.lb) + self.lb
    def update_fitness(self):
        try:
            for s in self.Salps[:self.n]:
                fitness = self.funct(s.getX())
                s.set_fitness(fitness)
        except:
            print("* Error! The function", self.funct, "is not defined. Please, provide a valid function.")
            exit(-1)
    
    def _create_salps(self):
        # we can add a individual from IMODE to this population pool
        if self.former_leader is not None:
            s = Salp()
            s.setX(self.former_leader)
            self.Salps.append(s)
            for _ in range(self.n-1):
                s = Salp()
                s.setX(self._initialise())
                self.Salps.append(s)
        else:
            for _ in range(self.n):
                s = Salp()
                s.setX(self._initialise())
                self.Salps.append(s)
        
        self.update_fitness()
        self.FEs = self.FEs + self.n

        self.Salps = sorted(self.Salps, key= lambda x: x.get_fitness(), reverse= False)

        self._F = copy.deepcopy(self.Salps[0])

        if self.verbose:
            print(f'* {self.n} salps have been created')
            print(f'* Iter {self.iteration}: best fitness {self._F.get_fitness()}')

    def _termination_criterion(self):
        if self.FEs > self.maxFEs-self.nmin:
            if self.verbose:
                print(f"The number of FEs has been used: {self.FEs}")
            return True
        else:
            return False
        
    def _updateC1(self):
        # tmp = np.random.rand() + (10*self.iteration)/self.max_iters
        # self.c1 = self.cmax + (self.cmin - self.cmax)*math.log10(tmp)

        self.c1 = 2 * math.exp(-((4 * self.iteration / self.max_iters) ** 2))

    def _updateN(self):
        tmp = ((self.nmin - self.nmax)/self.maxFEs)*self.FEs + self.nmax
        self.n = int(tmp)
    
    def _update_leader(self, index):
        c2 = np.random.uniform(0, 1, self.dim)
        c3 = np.random.uniform(0, 1, self.dim)

        F = copy.deepcopy(self._F.getX())
        X = np.zeros(self.dim)

        tmp2 = self.c1*((self.ub - self.lb)*c2 + self.lb)
        X[c3<0.5] = F[c3<0.5] + tmp2[c3<0.5]
        X[c3>=0.5] = F[c3>=0.5] - tmp2[c3>=0.5]

        self.Salps[index].setX(X)

    def _update_salps(self):
        for i in range(self.n):
            if (i<self.n /2):
                self._update_leader(i)

            else:
                X1= copy.deepcopy(self.Salps[i].getX())
                X2 = copy.deepcopy(self.Salps[i-1].getX())
                X = 0.5 *(X1+X2)
                self.Salps[i].setX(X)

        for s in self.Salps[:self.n]:
            X = copy.deepcopy(s.getX())
            for i in range(self.dim):
                X[i] = np.clip(X[i], self.lb, self.ub)
            
            s.setX(X)

    def _iterate(self):
        self._updateC1()
        self._updateN()

        self._update_salps()

        self.update_fitness()
        self.FEs = self.FEs + self.n

        for s in self.Salps:
            if s.get_fitness() < self._F.get_fitness():
                self._F = copy.deepcopy(s)

        if(self.verbose) & (self.iteration%100 == 0):
            print(f'* Iteration {self.iteration}: best fitness {self._F.get_fitness()}--- var {self._F.get_fitness()- self.opt_value}')
        
        self.iteration += 1

    def optimize(self, function):  # Return best position and best value
        self.funct = function
        if self.verbose:
            print("Starting ASSA code...")
        
        self._create_salps()

        while not self._termination_criterion():
            self._iterate()

        if self.verbose:
            print(f'Process terminated')

        return self._F.getX(), self._F.get_fitness()