import numpy as np
from math import prod
import matplotlib.pyplot as plt

# approximated uniform coordinates of non-uniform points
# h: uniform spacing
# L: length of the uniform region
def rounded_uniform_coordinates(pts, h):
    return np.floor(np.divide(pts, h)).astype(int)


# return a matrix such that each row corresponds to the coords of the bin
# in which the corresponding point in rounded_uniform_coords should be
# placed
def bin_coords(rounded_uniform_coords, bin_dims):
    return rounded_uniform_coords // bin_dims


def linearized_bin_coords(nbins):
    return np.arange(prod(nbins)).reshape((nbins), order="C")


# top-left and bottom-right
def bounds(bn):
    return np.vstack(
        [np.min(bn, axis=0)[None, :], np.max(bn, axis=0)[None, :]]
    )


# return a tensor of shape N x 2 x D where N is the number of bins, 2 is the
# number of bounds (top-left and bottom-right) and D is the dimensionality of
# the space
def compute_bins_bounds(bins):
    nbins = len(bins)
    bin_bounds = np.zeros((nbins, 2, pts.shape[1]), dtype=float)
    for bin_idx in range(nbins):
        bin_as_arr = np.array(bins[bin_idx])
        bins[bin_idx] = bin_as_arr
        if bin_as_arr.shape[0] > 0:
            bin_bounds[bin_idx] = bounds(bin_as_arr)
    return bin_bounds


def fill_bins(pts, h, bin_dims, uniform_points):
    L = np.max(uniform_points, axis=0) - np.min(uniform_points, axis=0)
    bins_per_axis = np.ceil((L / h / bin_dims)).astype(int)
    nbins = np.prod(bins_per_axis)

    bins = [[] for _ in range(nbins)]
    indexes_inside_bins = [[] for _ in range(nbins)]
    # rounded uniform coordinates for each non-uniform point
    uf_coords = rounded_uniform_coordinates(pts, h)
    # coordinates of the bin for a given non-uniform point
    bn_coords = bin_coords(uf_coords, bin_dims)

    # maps bin coords to a linear array
    linear_bn_map = linearized_bin_coords(bins_per_axis)
    # for each non-uniform point, gives the linearized coordinate of the
    # appropriate bin
    linearized_bn_coords = np.apply_along_axis(
        lambda row: linear_bn_map[tuple(row)], axis=1, arr=bn_coords
    )

    # put each non-uniform point into the appopriate bin
    for j in range(pts.shape[0]):
        linear_bin_coord = linearized_bn_coords[j]
        bins[linear_bin_coord].append(pts[j])
        indexes_inside_bins[linear_bin_coord].append(j)

    return bins, indexes_inside_bins


# for all the bins, return the top left and bottom right coords of the point
# representing the enclosing padded rectangle at distance max_distance
def compute_padded_bin_bounds(boundaries, distance):
    top_left = boundaries[:, 0] - distance
    bottom_right = boundaries[:, 1] + distance
    return np.concatenate([top_left[:, None], bottom_right[:, None]], axis=1)


def match_points_and_bins(bins_bounds, points):
    # this has one row for each pt in points, and one column for each bin.
    # True if the point in a given row belongs to the bin in a given column.
    inclusion_matrix = np.full((points.shape[0], bins_bounds.shape[0]), False)
    # we now need to check which uniform points are in which padded bin
    for bin_idx, bin_bounds in enumerate(bins_bounds):
        inside_bin = np.logical_and(
            np.all(bin_bounds[0] < points, axis=1),
            np.all(points < bin_bounds[1], axis=1),
        )
        inclusion_matrix[inside_bin, bin_idx] = True

    return inclusion_matrix


def compute_distance(pts1, pts2):
    return np.linalg.norm(pts1[:, None, :] - pts2[None, ...], axis=-1)


def compute_mapped_distance_matrix(
    bins, indexes_inside_bins, pts1, pts2, inclusion_matrix, max_distance, func
):
    func = np.vectorize(func)

    matrix = np.zeros((pts1.shape[0], pts2.shape[0]), dtype=float)
    for bin_idx in filter(
        lambda idx: len(bins) > 0, range(inclusion_matrix.shape[1])
    ):
        bin_pts1 = bins[bin_idx]

        pts2_in_bin = inclusion_matrix[:, bin_idx]
        padded_bin_pts2 = pts2[pts2_in_bin]
        # not needed if (1) is disabled
        bin_pts2_indexing_to_full = np.arange(pts2.shape[0])[pts2_in_bin]

        distances = compute_distance(bin_pts1, padded_bin_pts2)

        indexes = np.asarray(indexes_inside_bins[bin_idx])
        # a new layer of selection, may want to disable to improve performance
        nearby = distances < max_distance  # (1)

        for pt1_idx in range(bin_pts1.shape[0]):
            idx = indexes[pt1_idx]
            matrix[idx, bin_pts2_indexing_to_full[nearby[pt1_idx]]] = func(
                distances[pt1_idx, nearby[pt1_idx]]
            )
    return matrix


bin_dims = np.array([20, 20])
h = np.array([0.001, 0.001])
distance = 0.05

pts = np.random.rand(10000, 2)

ux = np.arange(1001) * h[0]
uy = np.arange(1001) * h[1]
uniform_points = np.reshape(np.meshgrid(ux, uy), (2, -1)).T

bins, indexes_inside_bins = fill_bins(pts, h, bin_dims, uniform_points)
bins_bounds = compute_bins_bounds(bins)
padded_bin_bounds = compute_padded_bin_bounds(bins_bounds, distance)

assert padded_bin_bounds.shape == bins_bounds.shape

inclusion_matrix = match_points_and_bins(padded_bin_bounds, uniform_points)
mapped_distance = compute_mapped_distance_matrix(
    bins,
    indexes_inside_bins,
    pts,
    uniform_points,
    inclusion_matrix,
    0.15,
    lambda x: x * x,
)

plt.figure(figsize=(20, 10))
plt.pcolormesh(mapped_distance)
plt.show()