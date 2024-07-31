class SubPop():
    def __init__(self, pool, prob, searcher=None):
        self.prob= prob
        self.pool = pool     
        self.searcher = searcher 
        self.fitness_improv = 0
        self.consume_fes = 0
        self.dim = 30
        # self.task = prob[self.skill_factor - 1]
        self.scale = 0.1
    def search(self):
        return self.searcher.search(self.pool)