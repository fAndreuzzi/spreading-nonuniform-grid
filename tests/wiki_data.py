import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2, 256)
rng = np.random.default_rng(seed=2)
x,y = np.meshgrid(t,t)
samples1 = np.stack((x.flatten(), y.flatten()), axis=-1)
samples2 = np.stack((2 * rng.random(size=4), 2 * rng.random(size=4)), axis=-1)
alpha = np.ones(samples2.shape[0])
sigma = 1 / 12
func = lambda x: np.exp(-x ** 2 / (2 * sigma ** 2))
