import random
import numpy as np
from utils.Individual import Individual
from utils.GeminiModel import Gemini
from scipy.stats import cauchy
import json

class IMODE_LLM:
    def __init__(self, init_F, init_CR, prob, survival_rate = None, H = 6, best_rate = 0.1, arc_rate = 2.6 ):
        self.prob = prob
        self.H = H
        self.effective = None
        self.survival_rate = survival_rate if survival_rate is not None else 0
        self.mem_cr = [init_CR] * self.H
        self.mem_f = [init_F] * self.H
        self.s_cr = []
        self.s_f = []
        self.diff_f = []
        self.archive = []
        self.ratio = [1/3 , 1/3, 1/3]
        self.mem_pos = 0
        self.arc_rate = arc_rate
        self.stats_history = []
        self.llm_interval = 5
        self.llm = Gemini(temperature=0.0)
        self.key = random.random()
    
    def generateFCR(self, m_f, m_cr):
        if m_cr == "terminal":
            cr = 0
        else:
            cr = np.random.normal(loc=m_cr, scale=0.1)
        if cr < 0:
            cr = 0
        if cr > 1:
            cr = 1
        while True:
            f = cauchy.rvs(loc = m_f, scale=0.1, size=1 )[0]
            if f>=0:
                break
        return min(f,1), min(cr,1)
    def cur_topbest(self, pool, ind, cr, f, best):
        pbest = best[random.randint(0, len(best) - 1)]
        rand_idx = np.random.choice(len(pool), 2, replace=False)
        ind_ran1, ind_ran2 = pool[rand_idx[0]], pool[rand_idx[1]]
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) <= cr)
        u[np.random.choice(len(ind.genes))] = True
        new_genes = np.where(u,
            ind.genes + f * (pbest.genes - ind.genes + ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        bounds = ind.bounds
        for j in range(len(new_genes)):
                if new_genes[j] > bounds[1]:
                    new_genes[j] = (bounds[1] + ind.genes[j]) / 2
                elif new_genes[j] < bounds[0]:
                    new_genes[j] = (bounds[0] + ind.genes[j]) / 2

        new_ind = Individual(ind.MAX_NVARS, bounds)
        new_ind.genes = new_genes
        new_ind.cal_fitness(self.prob)
        return new_ind
    def cur_topbestav(self, pool, ind, cr, f, best):
        pbest = best[random.randint(0, len(best) - 1)]
        ind_ran1, ind_ran2 = random.sample(pool + self.archive, 2)
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) <= cr)
        u[np.random.choice(len(ind.genes))] = True
        new_genes = np.where(u,
            ind.genes + f * (pbest.genes - ind.genes + ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        bounds = ind.bounds
        for j in range(len(new_genes)):
                if new_genes[j] > bounds[1]:
                    new_genes[j] = (bounds[1] + ind.genes[j]) / 2
                elif new_genes[j] < bounds[0]:
                    new_genes[j] = (bounds[0] + ind.genes[j]) / 2

        new_ind = Individual(ind.MAX_NVARS, bounds)
        new_ind.genes = new_genes
        new_ind.cal_fitness(self.prob)
        return new_ind


    def pbest_ind(self, pool, ind, cr, f, best):
        pbest = best[random.randint(0, len(best) - 1)]
        rand_idx = np.random.choice(len(pool), 2, replace=False)
        ind_ran1, ind_ran2 = pool[rand_idx[0]], pool[rand_idx[1]]
        u = (np.random.uniform(0, 1, size=(len(ind.genes)) ) <= cr)
        u[np.random.choice(len(ind.genes))] = True
        new_genes = np.where(u, 
            pbest.genes + f * (ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        bounds = ind.bounds
        for j in range(len(new_genes)):
                if new_genes[j] > bounds[1]:
                    new_genes[j] = (bounds[1] + ind.genes[j]) / 2
                elif new_genes[j] < bounds[0]:
                    new_genes[j] = (bounds[0] + ind.genes[j]) / 2

        new_ind = Individual(ind.MAX_NVARS, bounds)
        new_ind.genes = new_genes
        new_ind.cal_fitness(self.prob)
        return new_ind
    def split(self, pool):
        pop_list = list(len(pool) * np.array(self.ratio))
        pop_list = [int(x) for x in pop_list]
        max_idx = np.argmax(pop_list)
        min_idx = np.argmin(pop_list)
        if sum(pop_list) > len(pool):
            pop_list[max_idx] -= (sum(pop_list) - len(pool))
        if sum(pop_list) < len(pool):
            pop_list[min_idx] += (len(pool) - sum(pop_list))
        max_idx = np.argmax(pop_list)
        min_idx = np.argmin(pop_list)
        # print(pop_list)
        assert sum(pop_list) == len(pool), f"Sum of pop_list: {sum(pop_list)} != len(pool): {len(pool)}"
        while 0 in pop_list:
            zero_idx = pop_list.index(0)
            if max_idx != zero_idx:
                pop_list[zero_idx] += 1
                pop_list[max_idx] -= 1
            else:
                raise ValueError("Cannot split pool: all ratios are 0")
        
        assert sum(pop_list) == len(pool), f"Sum of pop_list: {sum(pop_list)} != len(pool): {len(pool)}"
        Pop_list = []
        start = 0
        for size in pop_list:
            end = start + size
            Pop_list.append(pool[start: end])
            start = end
        # assert sum(pop_list) == len(pool), f"Sum of pop_list: {sum(pop_list)} != len(pool): {len(pool)}"
        return Pop_list
    def addarchive(self, ind, pop_size):
        caution_size = int(self.arc_rate * pop_size)
        while len(self.archive) > caution_size:
            r = random.randint(0, len(self.archive) - 1)
            self.archive.pop(r)
        self.archive.append(ind)
    def _build_prompt(self):
        recent = self.stats_history[-self.llm_interval:]
        prompt_lines = [
            "You are a meta-optimizer for an adaptive Differential Evolution algorithm.",
            f"Here are the last {len(recent)} generations of subpopulation stats:"  
        ]
        for entry in recent:
            for idx, sub in enumerate(entry['subpop_stats'], start=1):
                prompt_lines.append(
                    f" Subpop {idx}:"
                    f" Best={sub['best']:.4f},"
                    f" Mean={sub['mean']:.4f},"
                    f" Diversity={sub['diversity']:.2f}"
                )
        prompt_lines.extend([
            "Definitions: 'ratios' specify how to split the entire population into three subpopulations for local searches.",
            f"Current ratios = r1: {self.ratio[0]:.3f}, r2: {self.ratio[1]:.3f}, r3: {self.ratio[2]:.3f}",
            "Task: Based solely on these subpopulation performances, suggest new ratios.",
            "IMPORTANT: Respond strictly with a JSON object containing four keys:",
            "  'r1', 'r2', 'r3' for the proportions (which must sum to 1),",
            "  and 'explanation' as a brief chain-of-thought starting with \"Let's think step by step:\"",
            "Example output: { \"r1\": 0.5, \"r2\": 0.3, \"r3\": 0.2, \"explanation\": \"Let's think step by step: ...\" }",
            "Do not include any additional text."
        ])
        return '\n'.join(prompt_lines)
    
    def _call_llm(self, prompt: str) -> dict:
        text, _, _ = self.llm.prompt([{"content": prompt}])
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            data = json.loads(text[start:end+1])
        return data
    


    def search(self, pool):
        pool.sort(key = lambda ind: ind.fitness)
        best = pool[: max( int(0.1*len(pool)), 2 )]
        new_pool = []
        operators = [self.cur_topbest, self.cur_topbestav, self.pbest_ind]
        quality = [0]*len(operators)
        div = [0]*len(operators)
        improv_rate = [0]*len(operators)
        accumulate_diff = 0
        old_FEs = self.prob.FE
        random.shuffle(pool)
        pop_list = self.split(pool)
        # build statistics
        sub_stats = []
        for sub in pop_list:
            sub_stats.append({
                "best": min(ind.fitness for ind in sub),
                "mean": float(np.mean([ind.fitness for ind in sub])),
                "diversity": float(np.mean([np.linalg.norm(ind.genes - sub[0].genes) for ind in sub]))
            })
        stats = {
            "subpop_stats": sub_stats
        }
        self.stats_history.append(stats)
        
        # print(pop_list)
        for idx, pop in enumerate(pop_list):
            for ind in pop:
                r = random.randint(0, self.H - 1)
                f, cr = self.generateFCR(self.mem_f[r], self.mem_cr[r])
                new_ind = operators[idx](pool, ind, cr, f, best)
                if ind.fitness > new_ind.fitness:
                    self.diff_f.append(ind.fitness - new_ind.fitness)
                    self.s_cr.append(cr)
                    self.s_f.append(f)
                    self.addarchive(ind, len(pool))
                    new_pool.append(new_ind)
                    accumulate_diff += ind.fitness- new_ind.fitness

                else:
                    new_pool.append(ind)
            # get information of operators
            pop.sort(key = lambda ind: ind.fitness)
            quality[idx] = pop[0].fitness
            div[idx] = np.sum([np.linalg.norm(ind.genes - pop[0].genes) for ind in pop])
        try: 
            # try using suggest from LLM
            prompt = self._build_prompt()
            # print(f"==================IMODE_LLM prompt:\n{prompt}======================")
            data = self._call_llm(prompt)
            print(f"==================IMODE_LLM response:\n{data}======================")
            if not all(key in data for key in ['r1', 'r2', 'r3', 'explanation']):
                raise ValueError("LLM response missing required keys.")
            r1, r2, r3 = data['r1'], data['r2'], data['r3']
            if not (0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1 and abs(r1 + r2 + r3 - 1) < 1e-6):
                raise ValueError("LLM response ratios are not valid.")
            self.ratio = [r1, r2, r3]

        except Exception as e:
            # then use default IMODE strategy
            print(f"LLM suggestion failed: {e}. Using default IMODE ratios.")
            qual_sum = sum(quality)
            div_sum = sum(div)
            for i in range(len(quality)):
                quality[i] = quality[i]/qual_sum if qual_sum != 0 else 0 
                div[i] = div[i]/div_sum if div_sum != 0 else 0
            for i in range(len(operators)):
                improv_rate[i] = (1 - quality[i]) + div[i]
            sum_improv_rate = sum(improv_rate)
            if sum_improv_rate == 0:
                self.ratio = [1/3 , 1/3, 1/3]
            else:
                self.ratio = [max(0.1, min(0.9, x/sum_improv_rate)) for x in improv_rate]
        # print(self.ratio)
        min_fitness1 = min(ind.fitness for ind in pool)
        min_fitness2 = min(ind.fitness for ind in new_pool)
        assert min_fitness2 <= min_fitness1, f'Fitness of new pool: {min_fitness2} > old pool:{min_fitness1} ???'
        new_fes = self.prob.FE
        self.effective = accumulate_diff / (new_fes - old_FEs)
        self.updateMemory()
        return new_pool
    def updateMemoryCR(self, diff_f, s_cr):
        diff_f = np.array(diff_f)
        s_cr = np.array(s_cr)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tu = sum(weight * s_cr * s_cr)
        mau = sum(weight * s_cr)
        return tu/mau
    def updateMemoryF(self, diff_f, s_f):
        diff_f = np.array(diff_f)
        s_f = np.array(s_f)

        sum_diff = sum(diff_f)
        weight = diff_f/sum_diff
        tu = sum(weight * s_f * s_f)
        mau = sum(weight * s_f)
        return tu/mau
    def updateMemory(self):
        if len(self.s_cr) > 0:
            if self.mem_cr[self.mem_pos] == "terminal" or max(self.s_cr)==0:
                self.mem_cr[self.mem_pos] = "terminal"
            else:
                self.mem_cr[self.mem_pos] = self.updateMemoryCR(self.diff_f, self.s_cr)
            self.mem_f[self.mem_pos] = self.updateMemoryF(self.diff_f, self.s_f)
            self.mem_pos = (self.mem_pos + 1) % self.H
            self.s_cr = []
            self.s_f = []
            self.diff_f = []
    
        else:
            # do nothing
            pass