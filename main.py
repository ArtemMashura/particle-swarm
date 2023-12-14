import random
import math
import numpy as np
from mealpy import PSO, SA, FloatVar


# objective function


def objective_clust(ob):
    x = np.reshape(ob, (-1, 2))
    n_cust = cust.shape[0]
    n_ob = x.shape[0]
    f = 0;
    d = np.zeros((n_ob, n_cust), dtype=float)
    for i in range(n_ob):
        for j in range(n_cust):
            d[i][j] = math.dist(x[i][:], cust[j][:])
    bin_matr = np.zeros((n_ob, n_cust), dtype=int)
    for i in range(n_cust):
        ind = np.argmin(d, axis=0)[i]
        bin_matr[ind][i] = 1
    f = f + np.sum(d * bin_matr)
    return f


N = 4
r = np.zeros(N, dtype=float)
r0 = 0.01
for k in range(N):
    r[k] = r0*(k+1)
n = 100
n1 = round(math.sqrt(n))
cust = np.zeros((n, 2), dtype=float)
for i in range(n):
    x_coord = (i // n1) * (1 / (n1 - 1))
    y_coord = (i % n1) * (1 / (n1 - 1))
    cust[i][0] = x_coord
    cust[i][1] = y_coord
x = np.random.rand(N,2)
problem_dict = {
    "bounds": FloatVar(lb=(0.,) * 2*N, ub=(1.,) * 2*N, name="delta"),
    "obj_func": objective_clust,
    "minmax": "min",
}

# model = SA.OriginalSA(epoch=100, pop_size=50, temp_init=100, step_size=0.01)
model = PSO.OriginalPSO(epoch=1000, pop_size=50, temp_init=100, step_size=0.01)
g_best = model.solve(problem_dict)
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")