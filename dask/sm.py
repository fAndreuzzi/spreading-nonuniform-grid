import numpy as np

from dask.distributed import Client
import dask.array as da

from functools import partial, reduce
from itertools import product
from math import ceil, floor, log10

import operator
import sys
import time

from gm_sort import find_bins
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
    load_data,
)

# return the modes extremal top left point and bottom right point of the fine
# grid generated from the given (non-uniform) points.
def fine_grid_boundaries(pts, h, alpha, n):
    s1 = solution1(pts, h, alpha)
    s1[np.where(s1 < 0)] = 0
    for i in range(len(n)):
        s1[np.where(s1 > n[i])] = n[i]

    s2 = solution2(pts, h, alpha)
    s2[np.where(s2 < 0)] = 0
    for i in range(len(n)):
        s2[np.where(s2 > n[i])] = n[i]

    return np.min(s1, axis=0), np.max(s2, axis=0)


def shape_from_boundaries(top_left, bottom_right):
    incr = lambda x: x + 1
    return tuple(map(incr, tuple(bottom_right - top_left)))


def interval_to_slice(start, size):
    return tuple(
        slice(start[i], start[i] + size[i], 1) for i in range(len(start))
    )


# given two vectors (first and last points for each component)
# prduces all the possible points whose components are all inside
# for instance if starts = [1,1] and lasts = [2,3] the points
# returned are (1,1), (2,1), (1,2), (2,2), (1,3), (2,3)
def all_combinations(starts, lasts):
    assert len(starts) == len(lasts)
    # produce a range from the first and the last values
    def rng(start, last):
        return range(start, last + 1)

    return map(np.array, product(*(map(rng, starts, lasts))))


def worker(nonuniform_idx, pts, f, kernel, n, h, w, alpha, sub_b, offset):
    x = pts[nonuniform_idx]
    c = f[nonuniform_idx]

    start = solution1(x, h, alpha)
    start[start < 0] = 0
    end = solution2(x, h, alpha)
    end[end > n - 1] = n[end > n - 1] - 1

    # kernel evaluated in the uniform grid (translated with the
    # non-uniform coordinates).
    # the translation occurs here
    krn_vals = np.zeros((len(n), np.max(n)), dtype=float)
    for i in range(len(start)):
        if start[i] <= end[i]:
            krn_vals[i, start[i] : (end[i] + 1)] = kernel(
                (h[i] * np.arange(start[i], end[i] + 1) - x[i]) / alpha[i]
            )

    for cmb in all_combinations(start, end):
        sub_b[tuple(cmb - offset)] += c * prod(
            krn_vals[i, cmb[i]] for i in range(len(cmb))
        )

    return sub_b, offset


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

    if len(sys.argv) > 5:
        msub = int(sys.argv[5])
    else:
        msub = 8

    client = Client(processes=True, n_workers=n_workers)

    pts, f = load_data()
    N = nonuniform_grid_size(pts)

    beta = compute_beta(epsilon)
    w = compute_w(epsilon)
    prt_kernel = partial(kernel, beta=beta)
    vec_krn = np.vectorize(prt_kernel)

    n = fine_grid_size(N, w)

    m = np.array([m1, m2])
    p = (m + 2 * np.ceil(w / 2)).astype(int)

    bin_dims = np.array([m1, m2])
    n_bins_axes = np.ceil(np.divide(n, bin_dims)).astype(int)
    nbins = np.prod(n_bins_axes)

    h = fine_grid_spacing(n)

    alpha = compute_alpha(w, n)

    coarse_bins = find_bins(nbins, pts, h, n_bins_axes, bin_dims)
    # impose msub maximum size for subproblems
    bins = list(
        map(lambda arr: np.split(arr, msub), map(np.array, coarse_bins))
    )
    # flatten list
    bins = reduce(operator.iconcat, bins, [])

    start = time.time_ns()

    futures = []
    for bn in bins:
        top_left, bottom_right = fine_grid_boundaries(pts[bn], h, alpha, n)
        # shape of sub_b (called "padded" in the paper)
        sub_b_shape = shape_from_boundaries(top_left, bottom_right)
        # offset for the translation from sub_b to the full b
        sub_b_offset = tuple(top_left)

        sub_b = da.from_array(np.zeros(sub_b_shape, dtype=float))

        for j in bn:
            futures.append(
                client.submit(
                    worker,
                    j,
                    pts=pts,
                    f=f,
                    kernel=vec_krn,
                    n=n,
                    h=h,
                    w=w,
                    alpha=alpha,
                    sub_b=sub_b,
                    offset=sub_b_offset,
                )
            )

    # merge all the sub-sums into b
    b = np.zeros(n, dtype=float)
    for sub_b, offset in client.gather(futures):
        b[interval_to_slice(offset, sub_b.shape)] += sub_b.compute()

    print("took {} seconds".format((time.time_ns() - start) / 1.0e9))

    np.save("../data/b_sm.npy", b)
