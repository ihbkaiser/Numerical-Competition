class Parameter:
    reps = 30
    MAX_FEs = 60000
    numRecords = 1000
    SUBPOPULATION = 100
    SIZE_POPULATION = 1000
    MAX_GENERATION = 1000

    mum = 5
    mu = 2

    rmp = 0.7
    pc = 0.8
    pm = 0.3

    num_fitness = 0
    countFitness = None
    o_rmp = None  
    FEs = None
    FEl = None

    @staticmethod
    def initialize_o_rmp():

        pass