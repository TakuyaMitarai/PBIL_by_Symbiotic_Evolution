import numpy as np

# ハイパーパラメータ
WPOP_SIZE = 400
PPOP_SIZE = 400
MAX_GENERATION = 1000
WCROSSOVER_PROB = 0.3
PCROSSOVER_PROB = 0.9
WMUTATE_PROB = 0.07
WCHROM_LEN = 10
PCHROM_LEN = 10
DISTRIBUTION_PROB = 0.1
ALPHA = 0.5
effective_observation = 20

# 部分解個体
class PartialIndividual:
    _id_counter = 0  # クラス変数としてIDカウンターを追加

    def __init__(self):
        self.id = PartialIndividual._id_counter  # 各インスタンスに一意のIDを割り当てる
        PartialIndividual._id_counter += 1  # IDカウンターをインクリメント
        self.chrom = np.random.randint(0, 2, PCHROM_LEN)
        self.fitness = 1000000


# 部分解集団
class PartialPopulation:
    def __init__(self):
        self.population = []
        self.prob_vectors = [np.full(PCHROM_LEN, 0.5) for _ in range(WCHROM_LEN)]
        for i in range(PPOP_SIZE):
            individual = PartialIndividual()
            self.population.append(individual)

    def update_prob_vectors(self, alpha, whole_population):
        # 全体解集団の上位30%から確率ベクトルを更新
        top_individuals = whole_population.population[:int(WPOP_SIZE * DISTRIBUTION_PROB)]
        for i in range(WCHROM_LEN):
            for j in range(PCHROM_LEN):
                avg_bit = np.mean([ind.chrom[i].chrom[j] for ind in top_individuals])
                self.prob_vectors[i][j] = (1 - alpha) * self.prob_vectors[i][j] + alpha * avg_bit

    def sample_new_population(self, crossover_prob, whole_population):
        num_new_individuals = int(PPOP_SIZE * crossover_prob)
        gene_usage_counts = np.zeros((PPOP_SIZE, WCHROM_LEN))
        for whole_ind in whole_population.population:
            for gene_index, part_ind in enumerate(whole_ind.chrom):
                gene_usage_counts[part_ind.id][gene_index] += 1

        individuals_to_update = self.population[-num_new_individuals:]

        for ind in individuals_to_update:
            posterior_probs = np.random.dirichlet(gene_usage_counts[ind.id] + effective_observation)
            selected_whole_gene_indices = np.random.choice(range(WCHROM_LEN), size=WCHROM_LEN, p=posterior_probs)
            selected_whole_gene_index = np.argmax(np.bincount(selected_whole_gene_indices))

            for gene_index in range(PCHROM_LEN):
                ind.chrom[gene_index] = 1 if np.random.rand() < whole_population.prob_vectors[selected_whole_gene_index][gene_index] else 0

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
        self.prob_vectors = [np.full(PCHROM_LEN, 0.5) for _ in range(WCHROM_LEN)]
        for i in range(WPOP_SIZE):
            individual = WholeIndividual()
            self.population.append(individual)
    
    def crossover(self):
        for i in range(int(WPOP_SIZE * (1 - WCROSSOVER_PROB)), WPOP_SIZE):
            parent1 = np.random.randint(0, int(WPOP_SIZE/8))
            parent2 = np.random.randint(0, int(WPOP_SIZE/8))
            index1 = np.random.randint(0, WCHROM_LEN)
            index2 = np.random.randint(0, WCHROM_LEN)
            self.population[i].crossover(self.population[parent1], self.population[parent2], index1, index2)

    def evainit(self):
        for i in range(WPOP_SIZE):
            self.population[i].fitness = 1000000

# 適応度評価
def evaluate_fitness(whole_population):
    for i in range(WPOP_SIZE):
        fitness = 0.0
        for j in range(WCHROM_LEN):
            for k in range(PCHROM_LEN):
                fitness += (whole_population.population[i].chrom[j].chrom[k] * 2 - 1) * np.sqrt(j*PCHROM_LEN+k+1)
        whole_population.population[i].fitness = np.abs(fitness)
        for j in range(WCHROM_LEN):
            if whole_population.population[i].chrom[j].fitness > whole_population.population[i].fitness:
                whole_population.population[i].chrom[j].fitness = whole_population.population[i].fitness
    ppop.population.sort(key=lambda individual: individual.fitness)
    whole_population.population.sort(key=lambda individual: individual.fitness)

# 初期化
ppop = PartialPopulation()
wpop = WholePopulation()
evaluate_fitness(wpop)

best = []

# 世代交代
for i in range(MAX_GENERATION):
    print(f"第{i+1}世代 最良個体適応度: {wpop.population[0].fitness}")
    best.append(wpop.population[0].fitness)

    # PBIL: 確率分布の更新
    ppop.update_prob_vectors(alpha=ALPHA, whole_population=wpop)

    # PBIL: 新しい個体のサンプリング
    ppop.sample_new_population(crossover_prob=PCROSSOVER_PROB, whole_population=wpop)

    # 全体解集団の交叉
    wpop.crossover()

    # 適応度初期化
    ppop.evainit()
    wpop.evainit()

    # 適応度算出
    evaluate_fitness(wpop)
