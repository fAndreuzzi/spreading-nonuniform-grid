import numpy as np
import matplotlib.pyplot as plt
from mapped_distance_matrix import mapped_distance_matrix
from pycsou.linop.sampling import MappedDistanceMatrix
from time import time

t = np.linspace(0, 2, 256)
rng = np.random.default_rng(seed=2)
x, y = np.meshgrid(t, t)
samples1 = np.stack((x.flatten(), y.flatten()), axis=-1)
samples2 = np.stack((2 * rng.random(size=4), 2 * rng.random(size=4)), axis=-1)
alpha = np.ones(samples2.shape[0])
sigma = 1 / 12
func = lambda x: np.exp(-(x ** 2) / (2 * sigma ** 2))

start = time()
m = mapped_distance_matrix(samples1, samples2, 0.1, func, should_vectorize=False)
print(time() - start)

start = time()
mdm = MappedDistanceMatrix(samples1=samples1, samples2=samples2, function=func, operator_type='dask')
print(time() - start)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.contourf(x, y, m.dot(alpha).reshape(t.size, t.size), 50)
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.subplot(1, 2, 2)
plt.contourf(x, y, (mdm * alpha).reshape(t.size, t.size), 50)
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.show()
