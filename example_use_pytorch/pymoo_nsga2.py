from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import time
problem = get_problem("ZDT1")
algorithm = NSGA2(pop_size=40, elimate_duplicates=False)
start = time.time()
res = minimize(problem,
               algorithm,
               ('n_gen', 300),
               verbose=False)
end = time.time()
plt.scatter(res.F[:, 0], res.F[:, 1], marker="o", s=10)
plt.grid(True)
plt.show()
print('耗时：', end - start, '秒')
