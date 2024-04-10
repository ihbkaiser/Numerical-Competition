import numpy as np
from copy import deepcopy
from scipy.stats import norm
from scipy.optimize import fminbound
class Model:
  def __init__(self, mean, std, num_sample):
    self.mean        = mean
    self.std         = std
    self.num_sample  = num_sample

  def density(self, subpop):
    N, D = subpop.shape
    prob = np.ones([N])
    for d in range(D):
      prob *= norm.pdf(subpop[:, d], loc=self.mean[d], scale=self.std[d])
    return prob

def log_likelihood(rmp, prob_matrix, K):
  posterior_matrix = deepcopy(prob_matrix)
  value = 0
  for k in range(2):
    for j in range(2):
      if k == j:
        posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * (1 - 0.5 * (K - 1) * rmp / float(K))
      else:
        posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * 0.5 * (K - 1) * rmp / float(K)
    value = value + np.sum(-np.log(np.sum(posterior_matrix[k], axis=1)))
  return value

def learn_models(subpops):
  K = len(subpops)
  D = subpops[0].shape[1]
  models = []
  for k in range(K):
    subpop            = subpops[k]
    num_sample        = len(subpop)
    num_random_sample = int(np.floor(0.1 * num_sample))
    rand_pop          = np.random.rand(num_random_sample, D)
    mean              = np.mean(np.concatenate([subpop, rand_pop]), axis=0)
    std               = np.std(np.concatenate([subpop, rand_pop]), axis=0)
    models.append(Model(mean, std, num_sample))
  return models

def learn_rmp(subpops, D):
  K          = len(subpops)
  rmp_matrix = np.eye(K)
  models = learn_models(subpops)

  for k in range(K - 1):
    for j in range(k + 1, K):
      probmatrix = [np.ones([models[k].num_sample, 2]), 
                    np.ones([models[j].num_sample, 2])]
      probmatrix[0][:, 0] = models[k].density(subpops[k])
      probmatrix[0][:, 1] = models[j].density(subpops[k])
      probmatrix[1][:, 0] = models[k].density(subpops[j])
      probmatrix[1][:, 1] = models[j].density(subpops[j])

      rmp = fminbound(lambda rmp: log_likelihood(rmp, probmatrix, K), 0, 1)
      rmp += np.random.randn() * 0.01
      rmp = np.clip(rmp, 0, 1)
      rmp_matrix[k, j] = rmp
      rmp_matrix[j, k] = rmp

  return rmp_matrix