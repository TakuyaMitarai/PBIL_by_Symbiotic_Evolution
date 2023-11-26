import numpy as np

# ハイパーパラメータ
WPOP_SIZE = 400
PPOP_SIZE = 400
MAX_GENERATION = 1000
WCROSSOVER_PROB = 0.9
PCROSSOVER_PROB = 0.99
WMUTATE_PROB = 0.1
WCHROM_LEN = 10
PCHROM_LEN = 10

# 部分解個体
class PartialIndividual:
    def __init__(self):
        self.chrom = np.random.randint(0, 2, PCHROM_LEN)
        self.fitness = 1000000

# 部分解集団
class PartialPopulation:
    def __init__(self):
        self.population = []
        self.prob_vector = np.full(PCHROM_LEN, 0.5)
        for i in range(PPOP_SIZE):
            individual = PartialIndividual()
            self.population.append(individual)
    
    def update_prob_vector(self, alpha):
        top_individuals = self.population[:int(PPOP_SIZE * 0.3)]
        for i in range(PCHROM_LEN):
            avg_bit = np.mean([ind.chrom[i] for ind in top_individuals])
            self.prob_vector[i] = (1 - alpha) * self.prob_vector[i] + alpha * avg_bit

    def sample_new_population(self, crossover_prob):
        num_new_individuals = int(PPOP_SIZE * crossover_prob)
        new_individuals = []
        for _ in range(num_new_individuals):
            new_individual = PartialIndividual()
            new_individual.chrom = np.array([1 if np.random.rand() < p else 0 for p in self.prob_vector])
            new_individuals.append(new_individual)
        self.population[-num_new_individuals:] = new_individuals

    def evainit(self):
        for i in range(PPOP_SIZE):
            self.population[i].fitness = 1000000

# 全体解個体
class WholeIndividual:
    def __init__(self):
        self.chrom = []
        for _ in range(WCHROM_LEN):
            index = np.random.randint(0, PPOP_SIZE)
            self.chrom.append(ppop.population[index])
        self.fitness = 1000000
    
    def crossover(self, parent1, parent2, index1, index2):
        if index1 > index2:
            tmp = index1
            index1 = index2
            index2 = tmp
        for i in range(0, index1):
            self.chrom[i] = parent1.chrom[i]
        for i in range(index1, index2):
            self.chrom[i] = parent2.chrom[i]
        for i in range(index2, WCHROM_LEN):
            self.chrom[i] = parent1.chrom[i]
        self.mutate()
    
    def mutate(self):
        for i in range(WCHROM_LEN):
            if np.random.rand() < WMUTATE_PROB:
                index = np.random.randint(0, PPOP_SIZE)
                self.chrom[i] = ppop.population[index]

# 全体解集団
class WholePopulation:
    def __init__(self):
        self.population = []
        for i in range(WPOP_SIZE):
            individual = WholeIndividual()
            self.population.append(individual)
    
    def crossover(self):
        for i in range(int(WPOP_SIZE * (1 - WCROSSOVER_PROB)), WPOP_SIZE):
            # 二点交叉
            parent1 = np.random.randint(0, int(WPOP_SIZE/4))
            parent2 = np.random.randint(0, int(WPOP_SIZE/4))
            index1 = np.random.randint(0, WCHROM_LEN)
            index2 = np.random.randint(0, WCHROM_LEN)
            self.population[i].crossover(self.population[parent1], self.population[parent2], index1, index2)

    def evainit(self):
        for i in range(WPOP_SIZE):
            self.population[i].fitness = 1000000

# floyd問題
def evaluate_fitness():
    for i in range(WPOP_SIZE):
        fitness = 0.0
        for j in range(WCHROM_LEN):
            for k in range(PCHROM_LEN):
                fitness += (wpop.population[i].chrom[j].chrom[k] * 2 - 1) * np.sqrt(j*PCHROM_LEN+k+1)
        wpop.population[i].fitness = np.abs(fitness)
        for j in range(WCHROM_LEN):
            if wpop.population[i].chrom[j].fitness > wpop.population[i].fitness:
                wpop.population[i].chrom[j].fitness = wpop.population[i].fitness
    ppop.population.sort(key=lambda individual: individual.fitness)
    wpop.population.sort(key=lambda individual: individual.fitness)

# 初期化
ppop = PartialPopulation()
wpop = WholePopulation()
evaluate_fitness()

best = []

# 世代交代
for i in range(MAX_GENERATION):
    print(f"第{i+1}世代 最良個体適応度: {wpop.population[0].fitness}")
    best.append(wpop.population[0].fitness)

    # PBIL: 確率分布の更新
    ppop.update_prob_vector(alpha=0.001)

    # PBIL: 新しい個体のサンプリング
    ppop.sample_new_population(crossover_prob=PCROSSOVER_PROB)

    # 全体解集団の交叉
    wpop.crossover()

    # 適応度初期化
    ppop.evainit()
    wpop.evainit()

    # 適応度算出
    evaluate_fitness()
