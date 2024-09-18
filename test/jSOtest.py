

import pyade.jso
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from cec2017.CEC2017 import CEC2017
algo = pyade.jso
prob = CEC2017(i=8, dim=30).fitness_of_ind
params = algo.get_default_params(dim=30)
params['bounds'] = np.array([[-100, 100]] * 30)
params['func'] = prob
sol, fit = algo.apply(**params)
print(fit)