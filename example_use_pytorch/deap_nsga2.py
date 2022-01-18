import array
import random
from deap import base
from deap import creator
from deap import tools
import numpy as np
import matplotlib.pyplot as plt
import time

def ZDT1(Vars, NDIM): # 目标函数
    ObjV1 = Vars[:, 0]
    gx = 1 + 9 * np.sum(Vars[:, 1:], 1) / (NDIM - 1)
    hx = 1 - np.sqrt(np.abs(ObjV1) / gx) # 取绝对值是为了避免浮点数精度异常带来的影响
    ObjV2 = gx * hx
    return np.array([ObjV1, ObjV2]).T

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()
# Problem definition
BOUND_LOW, BOUND_UP = 0.0, 1.0
NDIM = 30
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def main(seed=None):
    random.seed(seed)
    NGEN = 300
    MU = 40
    CXPB = 1.0
    pop = toolbox.population(n=MU)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    ObjV = ZDT1(np.array(invalid_ind), NDIM)
    for ind, i in zip(invalid_ind, range(MU)):
        ind.fitness.values = ObjV[i, :]
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        ObjV = ZDT1(np.array([ind for ind in offspring if not ind.fitness.valid]), NDIM)
        for ind, i in zip(invalid_ind, range(MU)):
            ind.fitness.values = ObjV[i, :]
        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
    return pop
if __name__ == "__main__":
    start = time.time()
    pop = main()
    end = time.time()
    p = np.array([ind.fitness.values for ind in pop])
    plt.scatter(p[:, 0], p[:, 1], marker="o", s=10)
    plt.grid(True)
    plt.show()
    print('耗时：', end - start, '秒')

