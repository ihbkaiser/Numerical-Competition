import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
class Execute:
    def __init__(self, algo, problems, num_run):
        self.problems = problems
        self.bestValue = [prob.opt_value for prob in problems]
        self.algo = algo
        self.num_run = num_run
        self.attempt = np.zeros((num_run, len(self.algo), len(self.problems)))
        self.logs = []
    def run(self):
        for i in range(self.num_run):
            this_algo = copy.deepcopy(self.algo)
            tmp_log = []
            for idx, algo in enumerate(this_algo):
                algo.run()
                self.attempt[i][idx] = [self.bestValue[t] - algo.log[t][-1][-1] for t in range(len(self.problems))]
                tmp_log.append(algo.log)
            self.logs.append(tmp_log)
    def convergence_plot(self, Task):
        plt.figure()
        for i in range(len(self.algo)):
            LOG = self.logs[0][i][Task - 1]
            FEs = [logg[0] for logg in LOG]
            fitness = [logg[1] for logg in LOG]
            plt.plot(FEs, fitness, label= self.algo[i].name)
        # plt.yscale("log")
        plt.legend()
        plt.show()
    def statistic(self):
        for idx, algo in enumerate(self.algo):
            print(self.algo[i].name + ": " + "\n")
            attempt = self.attempt[:, idx, :]
            mean_values = attempt.mean(axis=0)
            std_values = attempt.std(axis=0)
            for col_idx, (mean, std) in enumerate(zip(mean_values, std_values)):
                print(f"Task {col_idx+1}:")
                print(f"Mean: {mean}")
                print(f"Std: {std}")
                print("-----------------------")
            print("----------------------------\n")
