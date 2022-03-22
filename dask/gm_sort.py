import numpy as np

from dask.distributed import Client

from functools import partial, reduce
from itertools import product
from math import ceil, floor, log10

import operator
import sys
import time

from gm import (
    prod,
    compute_alpha,
    compute_beta,
    compute_w,
    kernel,
    fine_grid_size,
    fine_grid_spacing,
    solution1,
    solution2,
)

# h is the uniform spacing
def nonuniform_coord_to_uniform(pts, idx, h):
    return np.floor(np.divide(pts[idx] + np.pi, h))


# return a vector of bin coordinates ([0,1] means
# the first bin of the second row)
def which_bin(uniform_coord, bin_size):
    return uniform_coord // bin_size


def bin_index(nbins_per_direction, bin_coords):
    return sum(
        [
            bin_coords[i] * np.prod(nbins_per_direction[:i])
            for i in range(len(nbins_per_direction) - 1, -1, -1)
        ]
    )


# find a permutation of the points (represented by a
# list of integers which can be used to re-arrange
# a NumPy array) such that t(1),...,t(M1) belong to R1,
# t(M1+1),...,t(M2) belong to R2 and so on
def find_optimal_permutation(nbins, pts, h, bin_size):
    bins = [[] for _ in range(nbins)]
    for j,pt in enumerate(pts):
        uf_coords = nonuniform_coord_to_uniform(pts, h)
        bin_coords = which_bin(uniform_coord, bin_size)
        bins[bin_index(bin_size, bin_coords)].append(j)
    # flatten list
    return functools.reduce(operator.iconcat, bins, [])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_workers = int(sys.argv[1])
    else:
        n_workers = 8

    if len(sys.argv) > 2:
        epsilon = float(sys.argv[2])
    else:
        epsilon = 1.0e-10

    if len(sys.argv) > 3:
        m1 = int(sys.argv[3])
    else:
        m1 = 32

    if len(sys.argv) > 4:
        m2 = int(sys.argv[4])
    else:
        m2 = m1

    client = Client(processes=True, n_workers=n_workers)

    pts = np.load("../data/points.npy")
    N = np.array([len(np.unique(pts[:, i])) for i in range(pts.shape[1])])
    f = np.load("../data/function_values.npy")
    assert f.shape[0] == pts.shape[0]

    beta = compute_beta(epsilon)
    w = compute_w(epsilon)
    prt_kernel = partial(kernel, beta=beta)
    vec_krn = np.vectorize(prt_kernel)

    n = fine_grid_size(N, w)
    nbins = np.prod(np.ceil(np.divide(n, [m1, m2])))
    h = fine_grid_spacing(n)

    alpha = compute_alpha(w, n)

    t = find_optimal_permutation(nbins, pts, h, np.array([m1,m2]))
    pts = pts[t]
    f = f[t]

    remote_f = client.scatter(f)
    remote_pts = client.scatter(pts)

    start = time.time_ns()
    b = np.sum(
        np.array(
            client.gather(
                [
                    client.submit(
                        worker, i, remote_pts, remote_f, vec_krn, h, alpha
                    )
                    for i in range(len(pts))
                ]
            )
        ),
        axis=0,
    )
    print("took {} seconds".format((time.time_ns() - start) / 1.0e9))

    np.save("../data/b.npy", b)
