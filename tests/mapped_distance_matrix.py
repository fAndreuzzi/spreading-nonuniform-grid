import numpy as np
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


# top-left and bottom-right
def bounds(bn):
    return np.min(bn, axis=0)[None, :], np.max(bn, axis=0)[None, :]


# return a tensor of shape N x 2 x D where N is the number of bins, 2 is the
# number of bounds (top-left and bottom-right) and D is the dimensionality of
# the space
def compute_bins_bounds(bins, ndims):
    nbins = len(bins)
    bin_bounds = np.zeros((nbins, 2, ndims), dtype=float)
    for bin_idx in range(nbins):
        # until now the bin is a Python list. we want it to be a NumPy array
        bin_as_arr = np.array(bins[bin_idx])
        bins[bin_idx] = bin_as_arr

        # we don't do anything if the bin is empty
        if len(bin_as_arr) > 0:
            top_left, bottom_right = bounds(bin_as_arr)
            bin_bounds[bin_idx, 0] = top_left
            bin_bounds[bin_idx, 1] = bottom_right
    return bin_bounds


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
def fill_bins(pts, h, bin_dims, region_dimension):
    bins_per_axis = np.ceil((region_dimension / h / bin_dims)).astype(int)
    nbins = np.prod(bins_per_axis)

    bins = [[] for _ in range(nbins)]
    indexes_inside_bins = [[] for _ in range(nbins)]
    # rounded uniform coordinates for each non-uniform point
    uf_coords = rounded_uniform_coordinates(pts, h)
    # coordinates of the bin for a given non-uniform point
    bn_coords = bin_coords(uf_coords, bin_dims)

    # moves to the last bin of the axis any point which is outside the region
    # defined by pts2.
    for axis_idx in range(bn_coords.shape[1]):
        bn_coords[bn_coords[:, axis_idx] >= bins_per_axis[axis_idx]] = (
            bins_per_axis[axis_idx] - 1
        )

    # for each non-uniform point, gives the linearized coordinate of the
    # appropriate bin
    shifted_nbins_per_axis = np.ones_like(bins_per_axis)
    shifted_nbins_per_axis[:-1] = bins_per_axis[1:]
    linearized_bin_coords = np.sum(bn_coords * shifted_nbins_per_axis, axis=1)
    # the above is much more efficient than this:
    # lbc = np.arange(nbins).reshape(bins_per_axis, order="C")
    # linearized_bin_coords = np.apply_along_axis(
    #     lambda row: lbc[tuple(row)], axis=1, arr=bn_coords
    # )

    # put each non-uniform point into the appopriate bin
    for j in range(len(pts)):
        linear_bin_coord = linearized_bin_coords[j]
        bins[linear_bin_coord].append(pts[j])
        indexes_inside_bins[linear_bin_coord].append(j)

    return bins, indexes_inside_bins


# for all the bins, return the top left and bottom right coords of the point
# representing the enclosing padded rectangle at distance max_distance
def compute_padded_bin_bounds(boundaries, distance):
    top_left = boundaries[:, 0] - distance
    bottom_right = boundaries[:, 1] + distance
    return np.concatenate([top_left[:, None], bottom_right[:, None]], axis=1)


# given a set of bins bounds and a set of points, find which points are inside
# which bins (a point could belong to multiple bins)
def match_points_and_bins(bins_bounds, points):
    # this has one row for each pt in points, and one column for each bin.
    # True if the point in a given row belongs to the bin in a given column.
    inclusion_matrix = np.full((len(points), len(bins_bounds)), False)
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

    def filter_empty_bins(bin_idx):
        return len(bins[bin_idx]) > 0 and np.any(inclusion_matrix[:, bin_idx])

    # we filter away empty bins
    bins_indexes = filter(filter_empty_bins, range(inclusion_matrix.shape[1]))
    matrix = np.zeros((len(pts1), len(pts2)), dtype=float)

    for bin_idx in bins_indexes:
        bin_pts1 = bins[bin_idx]

        pts2_in_bin = inclusion_matrix[:, bin_idx]
        padded_bin_pts2 = pts2[pts2_in_bin]
        # not needed if (1) is disabled
        bin_pts2_indexing_to_full = np.arange(len(pts2))[pts2_in_bin]

        distances = compute_distance(bin_pts1, padded_bin_pts2)

        indexes = np.asarray(indexes_inside_bins[bin_idx])
        # a new layer of selection, may want to disable to improve performance
        nearby = distances < max_distance  # (1)

        def filter_zero_nearby(pt1_idx):
            return np.any(nearby[pt1_idx])

        for pt1_idx in filter(filter_zero_nearby, range(len(bin_pts1))):
            idx = indexes[pt1_idx]
            matrix[idx, bin_pts2_indexing_to_full[nearby[pt1_idx]]] = func(
                distances[pt1_idx, nearby[pt1_idx]]
            )
    return matrix


def mapped_distance_matrix(
    pts1, pts2, max_distance, func, h=None, bin_dims=None, chunks="auto"
):
    region_dimension = np.max(pts2, axis=0) - np.min(pts2, axis=0)

    if not h:
        # 1000 points in each direction
        h = region_dimension / 1000
    if not bin_dims:
        bin_dims = np.full(region_dimension.shape, 100)

    if bin_dims.dtype != int:
        raise ValueError("The number of points in each bin must be an integer")

    ndims = pts1.shape[1]

    bins, indexes_inside_bins = fill_bins(pts1, h, bin_dims, region_dimension)
    bins_bounds = compute_bins_bounds(bins, ndims)
    padded_bin_bounds = compute_padded_bin_bounds(bins_bounds, max_distance)

    assert padded_bin_bounds.shape == bins_bounds.shape

    inclusion_matrix = match_points_and_bins(padded_bin_bounds, pts2)
    mapped_distance = compute_mapped_distance_matrix(
        bins,
        indexes_inside_bins,
        pts1,
        pts2,
        inclusion_matrix,
        max_distance,
        func,
    )
    assert mapped_distance.shape == (len(pts1), len(pts2))

    return mapped_distance
