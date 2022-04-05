import numpy as np
import matplotlib.pyplot as plt
from mapped_distance_matrix import mapped_distance_matrix

ux = np.linspace(0, 1, 5)
uy = np.linspace(0, 1, 5)
upts = np.reshape(np.meshgrid(ux, uy), (2, -1)).T
print(upts)

x = np.random.rand(5,2)
print(x)


def func(d):
    return 0 if d == 0 else 1 / d
max_distance = 0.2


m = mapped_distance_matrix(x, upts, max_distance, func)

plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(x[:,0], x[:,1])
plt.scatter(upts[:,0], upts[:,1])
plt.grid()

for idx, xi in enumerate(x):
    plt.annotate(idx, xi+0.02)
    circle = plt.Circle(xi, max_distance, color='g', alpha=0.3)
    plt.gca().add_patch(circle)

for idx, uxi in enumerate(upts):
    plt.annotate(idx, uxi+0.02, color='r')

plt.subplot(1,2,2)
plt.pcolormesh(m)
plt.colorbar()

plt.show()
