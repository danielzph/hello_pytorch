from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import time

# create the reference directions to be used for the optimization
M = 3
ref_dirs = get_reference_directions("das-dennis", M, n_partitions=12)
N = ref_dirs.shape[0]
# create the algorithm object
algorithm = NSGA3(pop_size=N,
                  ref_dirs=ref_dirs)
start = time.time()
# execute the optimization
res = minimize(get_problem("dtlz1", n_obj = M),
               algorithm,
               termination=('n_gen', 500))
end = time.time()
Scatter().add(res.F).show()
print('耗时：', end - start, '秒')
