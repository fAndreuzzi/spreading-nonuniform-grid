import numpy as np

from dask.distributed import Client

from functools import partial, reduce
from itertools import product
from math import ceil, floor, log10

import operator
import sys
import time


def matrix_to_valuelist(mat):
    return mat.flatten(order="c")


def axes_to_list(x, y):
    return np.vstack(
        [np.vstack([np.array([xi, yi])[None, :] for yi in y]) for xi in x]
    )


def load_data():
    x = np.load("../data/x.npy")
    y = np.load("../data/y.npy")
    f = np.load("../data/f.npy")

    pts = axes_to_list(x, y)
    f = matrix_to_valuelist(f)

    assert f.shape[0] == pts.shape[0]

    return pts, f


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def compute_w(epsilon):
    return ceil(log10(1 / epsilon)) + 1


def compute_beta(epsilon):
    return 2.3 * compute_w(epsilon)


def kernel(z, beta):
    return np.exp(beta * (np.sqrt(1 - z * z) - 1))


def nonuniform_grid_size(pts):
    return np.array([len(np.unique(pts[:, i])) for i in range(pts.shape[1])])


def fine_grid_size(nonuniform_grid_size, w, upsampling_factor=2):
    # TODO fix
    return np.ceil(
        np.maximum(
            upsampling_factor * nonuniform_grid_size,
            2 * w * np.ones_like(nonuniform_grid_size),
        )
    ).astype(int)


def fine_grid_spacing(n):
    return 2 * np.pi / n


def compute_alpha(w, n):
    return w * np.pi / n


# smallest location of the grid of w^2 points in the
# uniform grid being influenced by the non-uniform
# point x
def solution1(x, h, alpha):
    return np.ceil((x - alpha) / h).astype(int)


# biggest location of the grid of w^2 points in the
# uniform grid being influenced by the non-uniform
# point x
def solution2(x, h, alpha):
    return np.floor((x + alpha) / h).astype(int)


def worker(nonuniform_idx, pts, f, kernel, n, h, alpha):
    x = pts[nonuniform_idx]
    c = f[nonuniform_idx]

    b = np.zeros(n, dtype=float)

    start = solution1(x, h, alpha)
    start[start < 0] = 0
    end = solution2(x, h, alpha)
    end[end > n - 1] = n[end > n - 1] - 1

    krn_transformation = lambda l: np.multiply(l, h) - x

    # kernel evaluated in the uniform grid (translated with the
    # non-uniform coordinates)
    krn_vals = np.zeros((len(n), np.max(n)), dtype=float)
    for i in range(len(start)):
        if start[i] <= end[i]:
            krn_vals[i, start[i] : (end[i] + 1)] = kernel(
                (h[i] * np.arange(start[i], end[i] + 1) - x[i]) / alpha[i]
            )

    for cmb in product(
        *[range(start[i], end[i] + 1) for i in range(len(start))]
    ):
        b[cmb[0], cmb[1]] += c * prod(
            krn_vals[i][cmb[i]] for i in range(len(cmb))
        )

    return b


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_workers = int(sys.argv[1])
    else:
        n_workers = 8

    if len(sys.argv) > 2:
        epsilon = float(sys.argv[2])
    else:
        epsilon = 1.0e-10

    client = Client(processes=True, n_workers=n_workers)

    pts, f = load_data()
    N = nonuniform_grid_size(pts)

    beta = compute_beta(epsilon)
    w = compute_w(epsilon)
    prt_kernel = partial(kernel, beta=beta)
    vec_krn = np.vectorize(prt_kernel)

    n = fine_grid_size(N, w)
    h = fine_grid_spacing(n)

    alpha = compute_alpha(w, n)

    start = time.time_ns()
    b = np.sum(
        np.array(
            client.gather(
                [
                    client.submit(worker, i, pts, f, vec_krn, n, h, alpha)
                    for i in range(len(pts))
                ]
            )
        ),
        axis=0,
    )
    print("took {} seconds".format((time.time_ns() - start) / 1.0e9))

    np.save("../data/b_gm.npy", b)
