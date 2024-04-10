import multiprocessing
import numpy as np
import cma

from concurrent.futures import ProcessPoolExecutor
from SpaceDecomposition import SpaceDecomposition
from DataFrame import gnbg_instances
from cma.evolution_strategy import fmin
from functools import partial

def fitness_function(i, x):
    return gnbg_instances[i].fitness(np.array([x])).item()

# Tạo các wrapper cho mỗi hàm fitness với i được cố định
fitness_wrapper = [partial(fitness_function, i) for i in range(24)]

num_points = 39  # N

def run_cma_es_on_subspace(j, space_decomp, fitness_wrapper_i, is_best_subspace=False, best_sol=None):
    lower_bounds = [space_decomp[j][k][0] for k in range(30)]
    upper_bounds = [space_decomp[j][k][1] for k in range(30)]
    if is_best_subspace:
        options = {
            'bounds': [lower_bounds, upper_bounds],
            'verbose': -9
        }
        x0 = best_sol[j]
        sigma0 = min([(upper_bounds[k] - lower_bounds[k]) / 9 for k in range(30)])
        res = fmin(fitness_wrapper_i, x0, sigma0, options, restarts=2, restart_from_best=True, incpopsize=2)
    else:
        options = {
            'bounds': [lower_bounds, upper_bounds],
            'verbose': -9
        }
        x0 = [(lower_bounds[k] + upper_bounds[k]) / 8 for k in range(30)]
        sigma0 = min([(upper_bounds[k] - lower_bounds[k]) / 2 for k in range(30)])
        res = fmin(fitness_wrapper_i, x0, sigma0, options)
    return j, res[0], res[1] 

if _name_ == '_main_':

    specific_tasks = [2, 5, 6, 9, 14, 15, 18, 20, 21, 22, 24]
    tasks_index = [(i - 1) for i in specific_tasks]

    for i in tasks_index:     
        space_decomp = SpaceDecomposition([-100, 100], 30, num_points, 1.5).extension()
        
        # Chạy song song lần đầu tiên
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [executor.submit(run_cma_es_on_subspace, j, space_decomp, fitness_wrapper[i]) for j in range(num_points + 1)]
            initial_results = [f.result() for f in futures]

        # Lấy ra 10 không gian con tốt nhất
        best_sol = {res[0]: res[1] for res in initial_results}
        best_values = {res[0]: res[2] for res in initial_results}
        best_indices = np.argsort(list(best_values.values()))[:10]
        
        # Chạy song song lần thứ hai cho 10 không gian con tốt nhất
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [executor.submit(run_cma_es_on_subspace, j, space_decomp, fitness_wrapper[i], True, best_sol) for j in best_indices]
            best_results = [f.result() for f in futures]

        # Lấy ra giá trị tốt nhất
        best_value = float('inf')
        best_solution = None
        for _, sol, val in best_results:
            if val < best_value:
                best_value = val
                best_solution = sol

        final = np.linalg.norm(gnbg_instances[i].OptimumPosition - best_solution)
        
        print(f'Distance of f{i + 1}: {final} - Error: {best_value - gnbg_instances[i].OptimumValue}')