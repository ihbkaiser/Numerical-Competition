import sys
import os
sys.path.append(os.path.dirname(__file__))
from searcher.MTSLS1 import *
from searcher.LBFGSB import *
from searcher.MutationLS import *
from searcher.IMODE import *
from searcher.IMODE_LLM import *
from searcher.Mutation import *
from searcher.jSO import *
from SubPop import *
from utils.GeminiModel import Gemini
import random
class LearningPhase():
    M = 2   #number of operators
    def __init__(self, is_start, prob) -> None:
        self.prob = prob
        self.start = is_start
        self.searcher = [ Mutation(prob)]
    def evolve(self, subpop, divisionRate = 0.4) :
        # split subpop into sub_pools which len(sub_pools)==len(self.searcher), 
        # each sub_pool is assigned an operator
        random.shuffle(subpop)
        new_pool = []
        sub_pools = []
        if divisionRate > 1/len(self.searcher):
            raise ValueError(f"divisionRate must be less than 1/num_searcher, but got {divisionRate} and {1/len(self.searcher)} respectively.")
        m = len(self.searcher)
        n = int ((1 - (m-1)* divisionRate) * len(subpop))
        # print("effective: ", [search.effective for search in self.searcher])
        
        for i in range(len(self.searcher)):
            if(i==0):
                sub_pools.append(SubPop(subpop[:n], self.prob))
            else:
                sub_pools.append(SubPop(subpop[int(n + (i-1)*divisionRate*len(subpop)):int(n + i*divisionRate*len(subpop))], self.prob))
        assert len(sub_pools) == len(self.searcher)
        

        if self.start:
            opcode = list(range(len(self.searcher)))
            random.shuffle(opcode)
            for idx, subpop in enumerate(sub_pools):
                subpop.searcher = self.searcher[opcode[idx]]
           
        else:
            opcode = np.argsort([search.effective for search in self.searcher])[::-1]
            for i, idx in enumerate(opcode):
                sub_pools[i].searcher = self.searcher[idx]
    
        for i in range(len(self.searcher)):
            pool = sub_pools[i].search()

            new_pool.extend(pool)
        return new_pool

class LearningPhaseILS():
    def __init__(self, prob):
        self.prob = prob
        self.searcher = [MTSLS1(prob), LBFGSB2(prob), MutationLS(prob)]
        self.explorer = IMODE_LLM(0.5,0.5,prob) 
        # self.explorer = jSO(prob)
        self.stay = 0
        self.use_restart = True
        self.grad_threshold = 30

    def evolve(self, subpop, DE_evals, LS_evals):
        ####### Explorer part (already implemented in Phase 2) ##########
        max_DE = self.prob.FE + DE_evals
        while self.prob.FE < max_DE:
            subpop = self.explorer.search(subpop)
        ################################################################
        subpop.sort(key = lambda ind: ind.fitness)
        # if self.searcher[0].effective > self.searcher[1].effective:
        #     this_iteration_searcher = self.searcher[0]
        # if self.searcher[0].effective < self.searcher[1].effective:
        #     this_iteration_searcher = self.searcher[1]
        # if self.searcher[0].effective == self.searcher[1].effective:
        #     this_iteration_searcher = random.choice(self.searcher)
        self.searcher.sort(key = lambda search: search.effective, reverse=True)
        highest_effective = self.searcher[0].effective
        top_searchers = [s for s in self.searcher if s.effective == highest_effective]
        this_iteration_searcher = random.choice(top_searchers)
        #print(f'This iter use {this_iteration_searcher.__class__.__name__}, effective: {this_iteration_searcher.effective}')


        subpop[0] = this_iteration_searcher.search(subpop[0], LS_evals)
        this_fitness = subpop[0].fitness
         ############ One more L-BFGS-B with random starting point ######################
        if self.use_restart:
            ind = copy.deepcopy(subpop[0])
            sol2, fit2, info2 = fmin_l_bfgs_b(self.prob.fitness_of_ind, np.random.uniform(-100, 100, len(ind.genes)), approx_grad=True, bounds=[(-100,100)]*len(ind.genes), maxfun=LS_evals, factr=10)
            if fit2 < this_fitness:
                new_ind = Individual(ind.MAX_NVARS, ind.bounds)
                new_ind.genes = sol2
                new_ind.fitness = fit2
                subpop[0] = new_ind
         ##### Check if we use restart next time or not #########
            if np.linalg.norm(info2["grad"]) > self.grad_threshold:
                self.use_restart = False
            if fit2 > this_fitness:
                disimprove_ratio = (fit2 - this_fitness) / abs(this_fitness)
                if disimprove_ratio > 5:
                    self.use_restart = False
                # print(f'Disimprove ratio: {disimprove_ratio}')
        return subpop

class LearningPhaseILSVer2():
    def __init__(self, prob):
        self.prob = prob
        self.searcher = [MTSLS1(prob), LBFGSB2(prob), MutationLS(prob)]
        self.explorer = jSO(prob)
        self.stagnation_threshold = [40, 40, 40]
        self.stagnation = [0]*len(self.searcher)
        self.LS_useful = True
        # self.explorer = jSO(prob)
        self.stay = 0
        self.use_restart = True
        self.grad_threshold = 30
    def evolve(self, subpop, DE_evals, LS_evals):
        max_DE = self.prob.FE + DE_evals
        while self.prob.FE < max_DE:
            subpop = self.explorer.search(subpop)
        subpop.sort(key = lambda ind: ind.fitness)
        # calculate coefficent of local searchs:
        coefficent = [0]*len(self.searcher)
        sum_stagnation = sum(self.stagnation)
        for i in range(len(self.searcher)):
            if sum_stagnation != 0 and self.stagnation[i] <= self.stagnation_threshold[i]:
                coefficent[i] = 1 - (self.stagnation[i])/sum_stagnation
            elif sum_stagnation == 0:
                coefficent[i] = 1
            else:
                coefficent[i] = 0
        # if all local searchs exceed threshold, then we shut down local search
        if(sum(coefficent) == 0):
            self.LS_useful = False
        if self.LS_useful:
            selected_idx = random.choices(range(len(self.searcher)), weights=coefficent, k=1)[0]
            # print(f'Use {self.searcher[selected_idx].__class__.__name__} -- stagnation {self.stagnation[selected_idx]}')
            selected_local_search = self.searcher[selected_idx]
            new_ind = selected_local_search.search(subpop[0], LS_evals)
            assert new_ind.fitness <= subpop[0].fitness, f'New fitness after LS: {new_ind.fitness} > Old fitness: {subpop[0].fitness}'
            # if fitness doesn't improve, we incrase stagnation
            if new_ind.fitness == subpop[0].fitness:
                if selected_local_search.__class__.__name__ == 'MTSLS1':
                    self.stagnation[selected_idx] += 8
                elif selected_local_search.__class__.__name__ == 'LBFGSB2':
                    if np.linalg.norm(selected_local_search.grad) > self.grad_threshold:
                        self.stagnation[selected_idx] += 8
                    else:
                        self.stagnation[selected_idx] += 1
                else:
                    self.stagnation[selected_idx] += 1
            else:
                subpop[0] = new_ind
        return subpop
        
class LearningPhaseLLM:
    def __init__(self, prob):
        self.prob = prob
        self.explorer = IMODE(0.5, 0.5, prob)
        self.searchers = {
            'MTSLS1': MTSLS1(prob),
            'LBFGSB2': LBFGSB2(prob),
            'MutationLS': MutationLS(prob)
        }
        self.ls_counts = {name: 0 for name in self.searchers}
        self.ls_effective = {name: 0.0 for name in self.searchers}
        self.ls_lastFE = {name: 0 for name in self.searchers}
        # Restart config
        self.use_restart = True
        self.grad_threshold = 30.0
        self.llm = Gemini(temperature=0) 

    def evolve(self, subpop, DE_evals, LS_evals):
        max_fe = self.prob.FE + DE_evals
        while self.prob.FE < max_fe:
            subpop = self.explorer.search(subpop)

        subpop.sort(key=lambda ind: ind.fitness)

        for name, searcher in self.searchers.items():
            self.ls_effective[name] = searcher.effective or 0.0

        router_stats = []
        for name in self.searchers:
            router_stats.append({
                'name': name,
                'usage': self.ls_counts[name],
                'effective': self.ls_effective[name],
                'last_FE': self.ls_lastFE[name]
            })

        prompt = (
            "You are a router deciding which local search method to apply next.\n"
            f"Here are stats for the three methods:\n{json.dumps(router_stats, indent=2)}\n"
            "Task: Select one method to use next, balancing equal usage counts, recent performance improvements, "
            "and considering FE budget used previously.\n"
            "IMPORTANT: Return strictly a JSON object with keys 'method' and 'explanation'. "
            "Explanation should start with 'Let's think step by step:'."
        )

        try:
            text, _, _ = self.llm.prompt([{'content': prompt}])
            raw = text.strip()
            start = raw.find('{')
            end = raw.rfind('}')
            json_str = raw[start:end+1] if start != -1 and end != -1 else raw
            print(f"================LLM response: {text}======================")
            choice = json.loads(json_str)
            method_name = choice['method']
            explanation = choice.get('explanation', '')
            print(f"=== LLM routing decision: {method_name} ===")
            if method_name not in self.searchers:
                raise KeyError('Invalid method')
        except Exception as e:
            print(f"LLM routing failed: {e}. Using fallback strategy.")
            sorted_ls = sorted(
                self.ls_counts.items(),
                key=lambda kv: (kv[1], -self.ls_effective[kv[0]])
            )
            method_name = sorted_ls[0][0]
            explanation = (
                "Let's think step by step: LLM routing failed or invalid. "
                "Choosing method with lowest usage and best recent performance."
            )


        self.ls_counts[method_name] += 1
        self.ls_lastFE[method_name] = LS_evals

        best_ind = subpop[0]
        ls = self.searchers[method_name]
        improved = ls.search(best_ind, LS_evals)
        subpop[0] = improved

        # Optional restart
        if self.use_restart:
            ind = copy.deepcopy(subpop[0])
            x0 = np.random.uniform(-100, 100, size=len(ind.genes))
            bounds = [(-100, 100)] * len(ind.genes)
            sol2, fit2, info2 = fmin_l_bfgs_b(
                self.prob.fitness_of_ind,
                x0,
                approx_grad=True,
                bounds=bounds,
                maxfun=LS_evals,
                factr=10
            )
            if fit2 < subpop[0].fitness:
                subpop[0].genes = sol2
                subpop[0].fitness = fit2
            grad = info2.get('grad')
            # gradient too high --> noise problem
            if grad is not None and np.linalg.norm(grad) > self.grad_threshold:
                self.use_restart = False
            

        return subpop
        
    