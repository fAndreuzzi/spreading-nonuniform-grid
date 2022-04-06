import numpy as np
import matplotlib.pyplot as plt

from dask.distributed import Client
from dask.distributed import as_completed

from functools import partial

# approximated uniform coordinates of non-uniform points
# h: uniform spacing
# L: length of the uniform region
def rounded_uniform_coordinates(pts, h):
    return np.floor(np.divide(pts, h)).astype(int)


# return a matrix such that each row corresponds to the coords of the bin
# in which the corresponding point in rounded_uniform_coords should be
# placed
def compute_bin_coords(rounded_uniform_coords, bin_dims):
    return rounded_uniform_coords // bin_dims


# top-left and bottom-right
def bounds(bin_pts):
    return np.min(bin_pts, axis=0)[None, :], np.max(bin_pts, axis=0)[None, :]


def group_by(a):
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


# build a list of lists, where each list contains the points in pts contained
# inside the bin corresponding to a certain (linearized) coordinate. For more on
# linearized bin coordinates see bin_coords and linearized_bin_coords.
# pts is the matrix of points.
# h is the granularity of the uniform grid we intend to build.
# bin_dims is the number of uniform points to be included in each (non-padded)
#   bin, in each direction.
# region_dimension is the dimension of the region used to enclose the points.
# it is preferable that bin_dims * h divides region_dimension exactly in each
# direction.
def fill_bins(pts, h, bin_dims, region_dimension, dask_client):
    bins_per_axis = np.ceil((region_dimension / h / bin_dims)).astype(int)
    nbins = np.prod(bins_per_axis)

    indexes_inside_bins = [[] for _ in range(nbins)]
    # rounded uniform coordinates for each non-uniform point
    uf_coords = rounded_uniform_coordinates(pts, h)
    # coordinates of the bin for a given non-uniform point
    bin_coords = compute_bin_coords(uf_coords, bin_dims)

    # moves to the last bin of the axis any point which is outside the region
    # defined by pts2.
    for axis_idx in range(bin_coords.shape[1]):
        bin_coords[bin_coords[:, axis_idx] >= bins_per_axis[axis_idx]] = (
            bins_per_axis[axis_idx] - 1
        )

    # for each non-uniform point, gives the linearized coordinate of the
    # appropriate bin
    shifted_nbins_per_axis = np.ones_like(bins_per_axis)
    shifted_nbins_per_axis[:-1] = bins_per_axis[1:]
    linearized_bin_coords = np.sum(bin_coords * shifted_nbins_per_axis, axis=1)

    # add a second column containing the index in pts1
    linearized_bin_coords = np.hstack(
        [linearized_bin_coords[:, None], np.arange(len(pts))[:, None]]
    )
    indexes_inside_bins = group_by(linearized_bin_coords)

    future_bins = dask_client.map(
        lambda idxes_in_bin: pts[idxes_in_bin], indexes_inside_bins
    )
    return future_bins, indexes_inside_bins


# for all the bins, return the top left and bottom right coords of the point
# representing the enclosing padded rectangle at distance max_distance
def compute_padded_bin_bounds(boundaries, distance):
    top_left, bottom_right = boundaries
    padded_top_left = top_left - distance
    padded_bottom_right = bottom_right + distance
    return padded_top_left, padded_bottom_right


# given a set of bins bounds and a set of points, find which points are inside
# which bins (a point could belong to multiple bins)
def points_inside_bin(bin_bounds, points):
    inclusion_vector = np.full(len(points), False)
    inside_bin = np.logical_and(
        np.all(bin_bounds[0] < points, axis=1),
        np.all(points < bin_bounds[1], axis=1),
    )
    inclusion_vector[inside_bin] = True
    return inclusion_vector


def compute_distance(pts1, pts2):
    return np.linalg.norm(pts1[:, None, :] - pts2[None, ...], axis=-1)


def compute_mapped_distance_matrix(
    bin,
    indexes_inside_bin,
    inclusion_vector,
    pts1,
    pts2,
    max_distance,
    func,
):
    n_pts1_in_bin = len(bin)
    n_pts2_in_bin = np.count_nonzero(inclusion_vector)
    if n_pts1_in_bin == 0 or n_pts2_in_bin == 0:
        return

    submatrix = np.zeros((n_pts1_in_bin, n_pts2_in_bin))
    distances = compute_distance(
        pts1[indexes_inside_bin], pts2[inclusion_vector]
    )

    # indexes is the list of indexes in pts1 that belong to this bin
    indexes = np.asarray(indexes_inside_bin)
    # a new layer of selection, may want to disable to improve performance
    nearby = distances < max_distance

    for pt1_idx in range(n_pts1_in_bin):
        pts2_indexes = nearby[pt1_idx]
        if np.any(pts2_indexes):
            submatrix[pt1_idx, pts2_indexes] = func(
                distances[pt1_idx, pts2_indexes]
            )

    return submatrix, indexes, inclusion_vector


def mapped_distance_matrix(
    pts1,
    pts2,
    max_distance,
    func,
    h=None,
    bin_dims=None,
    chunks="auto",
    should_vectorize=True,
):
    client = Client(processes=True)

    region_dimension = np.max(pts2, axis=0) - np.min(pts2, axis=0)

    # not using np.vectorize if the function is already vectorized allows us
    # to save some time
    if should_vectorize:
        func = np.vectorize(func)

    if not h:
        # 1000 points in each direction
        h = region_dimension / 1000
    if not bin_dims:
        bin_dims = np.full(region_dimension.shape, 100)

    if bin_dims.dtype != int:
        raise ValueError("The number of points in each bin must be an integer")

    ndims = pts1.shape[1]

    bins, indexes_inside_bins = fill_bins(
        pts1, h, bin_dims, region_dimension, client
    )
    bins_bounds = client.map(bounds, bins)

    pcompute_padded_bin_bounds = partial(
        compute_padded_bin_bounds, distance=max_distance
    )
    padded_bin_bounds = client.map(pcompute_padded_bin_bounds, bins_bounds)

    ppoints_inside_bin = partial(points_inside_bin, points=pts2)
    inclusion_vectors = client.map(ppoints_inside_bin, padded_bin_bounds)

    pcompute_mapped_distance_matrix = partial(
        compute_mapped_distance_matrix,
        pts1=pts1,
        pts2=pts2,
        max_distance=max_distance,
        func=func,
    )
    sub_results = client.map(
        pcompute_mapped_distance_matrix,
        bins,
        indexes_inside_bins,
        inclusion_vectors,
    )

    matrix = np.zeros((len(pts1), len(pts2)), dtype=float)
    for _, submatrix, pt1_idxes, pt2_idxes in as_completed(
        sub_results, with_results=True
    ):
        matrix[pt1_idxes, pt2_idxes] += submatrix

    return matrix
