from math import factorial
import random
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3d
import time

# Problem definition
NOBJ = 3
K = 5
NDIM = NOBJ + K - 1
P = 12
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 0.0, 1.0

def DTLZ1(Vars, NDIM): # 目标函数
    XM = Vars[:,(NOBJ-1):]
    g = 100 * (NDIM - NOBJ + 1 + np.sum(((XM - 0.5)**2 - np.cos(20 * np.pi * (XM - 0.5))), 1, keepdims = True))
    ones_metrix = np.ones((Vars.shape[0], 1))
    f = 0.5 * np.hstack([np.fliplr(np.cumprod(Vars[:,:NOBJ-1], 1)), ones_metrix]) * np.hstack([ones_metrix, 1 - Vars[:, range(NOBJ - 2, -1, -1)]]) * (1 + g)
    return f

##
# Algorithm parameters
MU = int(H)
NGEN = 500
CXPB = 1.0
MUTPB = 1.0/NDIM
##
# Create uniform reference point
ref_points = tools.uniform_reference_points(NOBJ, P)
# Create classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)
# Toolbox initialization
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
##
def main(seed=None):
    random.seed(seed)
    # Initialize statistics object
    pop = toolbox.population(n=MU)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    ObjV = DTLZ1(np.array(invalid_ind), NDIM)
    for ind, i in zip(invalid_ind, range(MU)):
        ind.fitness.values = ObjV[i, :]
    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        ObjV = DTLZ1(np.array([ind for ind in offspring if not ind.fitness.valid]), NDIM)
        for ind, i in zip(invalid_ind, range(MU)):
            ind.fitness.values = ObjV[i, :]
        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)
    return pop

if __name__ == "__main__":
    start = time.time()
    pop = main()
    end = time.time()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = np.array([ind.fitness.values for ind in pop])
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="o", s=10)
    ax.view_init(elev=30, azim=45)
    plt.grid(True)
    plt.show()
    print('耗时：', end - start, '秒')
