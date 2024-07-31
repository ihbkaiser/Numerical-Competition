from sklearn.decomposition import TruncatedSVD
import numpy as np
from MFEAcode import *
np.random.seed(0)
class SVD:
    def __init__(self, epsilon, pop, prob):
        self.epsilon = epsilon 
        genes_pool = [ind.genes for ind in pop]
        self.pop_mat = np.array(genes_pool)
        self.dim = pop[0].dim
        self.prob = prob
        self.N = len(pop)
    def fit(self, epsilon):
        new_pop = Population(self.N, self.dim, self.prob)
        new_dim = int(4/(epsilon**2/2 - epsilon**3/3) * np.log(self.N))
        svd = TruncatedSVD(n_components=new_dim)
        assert new_dim < self.dim
        U, S, V = np.linalg.svd(self.pop_mat)
        new_pop_mat = svd.fit(self.pop_mat).transform(self.pop_mat)
        

