import numpy as np
import random



''' initial without chaos '''
#
# def initial(pop, dim, ub, lb):
#     X = np.zeros([pop, dim])
#     for i in range(pop):
#         for j in range(dim):
#             X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
#
#     return X, lb, ub

''' Tent映射 '''

def Tent(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()  # 初始点
    a = 0.7  # 参数a的值
    for i in range(Max_iter - 1):
        if x[i] < a:
            x[i + 1] = x[i] / a
        if x[i] >= a:
            x[i + 1] = (1 - x[i]) / (1 - a)

    return x


''' Logistic映射 '''
def Logistic(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    if x[0] == 0. :
        x[0] = random.random()
    elif x[0] == 0.25 :
        x[0] = random.random()
    elif x[0] == 0.5 :
        x[0] = random.random()
    elif x[0] == 0.75 :
        x[0] = random.random()
    else :
        x[0] = x[0]
    u = 4    # 这个4很重要 由logistic混沌映射决定的,具体可以看logistic的性质
    for i in range(Max_iter - 1):
        x[i+1] = u * x[i] * (1-x[i])

    return x

''' Gussian映射 '''

# def Gussian(Max_iter):
#     x = np.zeros([Max_iter, 1])
#     x[0] = random.random()
#     for i in range(Max_iter - 1):
#         if x[i] == 0. :
#             x[i+1] = 0.
#         else :
#             x[i + 1] = 1 / (x[i]*    )

''' Chebyshev映射 '''

import math

def Chebyshev(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    for i in range(Max_iter - 1):
        a = 5.78
        x[i+1] = math.cos(a*math.acos(x[i]))
    return x

''' Singer映射 '''

def Singer(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    for i in range(Max_iter - 1):
        u = 1.07  # u /in [0.9,1.08]
        x[i+1] = u * (7.86 * x[i] - 23.31 * (x[i] ** 2) + 28.75 * (x[i] ** 3) - 13.302875 *(x[i] ** 4))
    return x   # x /in [0,1]


''' Sine映射 '''

import math

def Sine(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    for i in range(Max_iter - 1):
        a=3  # a /in (0,4]
        x[i + 1] = 4/a * math.sin(math.pi*x[i])

    return x  # x /in [-1,1]

''' Cubic映射 '''
def Cubic(Max_iter):
    x = np.zeros([Max_iter, 1])
    x[0] = random.random()
    for i in range(Max_iter - 1):
        a = 4
        b = 3
        x[i+1] = a * x[i]**3 - b * x[i]

    return x  # x /in [-1,1]




''' 种群初始化函数 '''

# def initial(pop, dim, ub, lb):
#     X = np.zeros([pop, dim])
#     for i in range(pop):
#         # TentValue = Tent(dim)
#         # LogValue = Logistic(dim)
#         ChebyValue = Chebyshev(dim)
#         for j in range(dim):
#             # X[i, j] = TentValue[j] * (ub[j] - lb[j]) + lb[j]
#             # X[i, j] = LogValue[j] * (ub[j] - lb[j]) + lb[j]
#             X[i,j] = ChebyValue[j] * ((ub[j] - lb[j])/2)
#             if X[i, j] > ub[j]:
#                 X[i, j] = ub[j]
#             if X[i, j] < lb[j]:
#                 X[i, j] = lb[j]
#
#     return X

''' 种群初始化函数--改进版本 '''
def ipinitial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for j in range(dim):
        # ChebyValue = Chebyshev(pop)
        # TentValue = Tent(pop)
        # LogValue = Logistic(pop)
        SingerValue = Singer(pop)
        for i in range(pop):
            # X[i, j] = ChebyValue[i] * ((ub[j] - lb[j])/2)
            # X[i, j] = TentValue[i] * (ub[j] - lb[j]) + lb[j]
            # X[i, j] = LogValue[i] * (ub[j] - lb[j]) + lb[j]
            X[i, j] = SingerValue[i] * (ub[j] - lb[j]) + lb[j]
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]

    return X



if __name__ == "__main__":
    pop=50
    dim=3
    ub=10*np.ones([dim,1])
    lb=-10*np.ones([dim,1])
    # X= initial(pop, dim, ub, lb)
    X = ipinitial(pop, dim, ub, lb)
    print(X)


