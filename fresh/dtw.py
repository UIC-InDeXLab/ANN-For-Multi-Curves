"""Exact cumulative Euclidean DTW and bottleneck matching of multi-curves.

Curves are arrays (m, d), multi-curves (k, m, d), and collections (n, k, m, d).
The two arguments may have different vertex counts. There is no normalization,
warping band, or change to the min-bijection/max-curve aggregation.
Pruning combines endpoint sums, box separation times minimum path length, and
vertex-to-box sums. Ground truth precomputes all curve boxes once per collection.
"""
import operator
import time

import numpy as np
from numba import njit, prange

from curveset import core


__all__ = ["dtw", "dtw_bounded", "dtw_distance", "dtw_endpoint_lb",
           "dtw_ground_truth", "dtw_verify"]


@njit(cache=True)
def _validate_finite(A):
    for value in A.flat:
        if not np.isfinite(value):
            raise ValueError("DTW coordinates must be finite")


@njit(cache=True)
def _validate_curves(A, B):
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("DTW curves must have shape (vertices, dimensions)")
    if len(A) == 0 or len(B) == 0:
        raise ValueError("DTW curves must not be empty")
    if A.shape[1] == 0 or A.shape[1] != B.shape[1]:
        raise ValueError("DTW curves must have equal, positive dimensions")
    _validate_finite(A)
    _validate_finite(B)


@njit(cache=True)
def _validate_sets(P, Q):
    if P.ndim != 3 or Q.ndim != 3:
        raise ValueError("DTW multi-curves must have shape (curves, vertices, dimensions)")
    if len(P) == 0 or len(P) != len(Q):
        raise ValueError("DTW multi-curves must have equal, positive curve counts")
    if P.shape[1] == 0 or Q.shape[1] == 0:
        raise ValueError("DTW curves must not be empty")
    if P.shape[2] == 0 or P.shape[2] != Q.shape[2]:
        raise ValueError("DTW curves must have equal, positive dimensions")
    _validate_finite(P)
    _validate_finite(Q)


@njit(cache=True)
def _validate_collection_shapes(X, Qs):
    if X.ndim != 4 or Qs.ndim != 4:
        raise ValueError("DTW collections must have shape (parents, curves, vertices, dimensions)")
    if len(X) == 0:
        raise ValueError("DTW data collection must not be empty")
    if X.shape[1] == 0 or X.shape[1] != Qs.shape[1]:
        raise ValueError("DTW multi-curves must have equal, positive curve counts")
    if X.shape[2] == 0 or Qs.shape[2] == 0:
        raise ValueError("DTW curves must not be empty")
    if X.shape[3] == 0 or X.shape[3] != Qs.shape[3]:
        raise ValueError("DTW curves must have equal, positive dimensions")


@njit(cache=True)
def _validate_ids(ids, size):
    if ids.ndim != 1:
        raise ValueError("DTW parent ids must be a one-dimensional array")
    for value in ids:
        if not np.isfinite(value) or value < 0 or value >= size or value != int(value):
            raise ValueError("DTW parent ids must be integers within the data collection")


@njit(cache=True)
def _euclidean(a, b):
    squared = 0.0
    for d in range(len(a)):
        difference = float(a[d]) - float(b[d])
        squared += difference * difference
    return np.sqrt(squared)


@njit(cache=True)
def _dtw_bounded(A, B, limit):
    previous = np.full(len(B), np.inf)
    current = np.empty(len(B))
    for i in range(len(A)):
        row_minimum = np.inf
        for j in range(len(B)):
            if i == 0 and j == 0:
                prior = 0.0
            elif i == 0:
                prior = current[j - 1]
            elif j == 0:
                prior = previous[j]
            else:
                prior = min(previous[j], previous[j - 1], current[j - 1])
            value = prior + _euclidean(A[i], B[j])
            current[j] = value
            row_minimum = min(row_minimum, value)
        # Every complete warping path crosses this row; future costs are nonnegative.
        if row_minimum > limit:
            return np.inf
        previous, current = current, previous
    value = previous[-1]
    return value if value <= limit else np.inf


@njit(cache=True)
def dtw(A, B):
    """Unconstrained DTW: sum of Euclidean local costs along a minimum-cost path."""
    _validate_curves(A, B)
    return _dtw_bounded(A, B, np.inf)


@njit(cache=True)
def dtw_bounded(A, B, limit):
    """Return exact DTW when it is <= limit, otherwise infinity (inclusive cutoff)."""
    _validate_curves(A, B)
    if np.isnan(limit):
        raise ValueError("DTW limit must not be NaN")
    return _dtw_bounded(A, B, limit)


@njit(cache=True)
def _endpoint_cost(A, B):
    start = _euclidean(A[0], B[0])
    if len(A) == 1 and len(B) == 1:
        return start
    return start + _euclidean(A[-1], B[-1])


@njit(cache=True)
def _endpoint_matrix(P, Q):
    k = len(Q)
    bounds = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            bounds[i, j] = _endpoint_cost(Q[i], P[j])
    return bounds


@njit(cache=True)
def dtw_endpoint_lb(P, Q):
    """Bottleneck bound from mandatory start/end costs, once for two singletons."""
    _validate_sets(P, Q)
    return core.bottleneck(_endpoint_matrix(P, Q))


@njit(cache=True)
def _curve_box(A):
    box = np.empty((2, A.shape[1]))
    for d in range(A.shape[1]):
        lower, upper = float(A[0, d]), float(A[0, d])
        for i in range(1, len(A)):
            value = float(A[i, d])
            lower, upper = min(lower, value), max(upper, value)
        box[0, d], box[1, d] = lower, upper
    return box


@njit(cache=True)
def _set_boxes(P):
    boxes = np.empty((len(P), 2, P.shape[2]))
    for i in range(len(P)):
        boxes[i] = _curve_box(P[i])
    return boxes


@njit(cache=True, parallel=True)
def _collection_boxes(X):
    boxes = np.empty((len(X), X.shape[1], 2, X.shape[3]))
    for p in prange(len(X)):
        boxes[p] = _set_boxes(X[p])
    return boxes


@njit(cache=True)
def _box_cost(A_box, B_box, minimum_cells):
    squared = 0.0
    for d in range(A_box.shape[1]):
        gap = max(0.0, A_box[0, d] - B_box[1, d], B_box[0, d] - A_box[1, d])
        squared += gap * gap
    # Every path has at least max(mA, mB) cells, each at least this box distance.
    return np.sqrt(squared) * minimum_cells


@njit(cache=True)
def _vertex_box_cost(A, B_box, limit):
    total = 0.0
    for i in range(len(A)):
        squared = 0.0
        for d in range(A.shape[1]):
            value = float(A[i, d])
            gap = max(0.0, B_box[0, d] - value, value - B_box[1, d])
            squared += gap * gap
        total += np.sqrt(squared)
        if total > limit:
            break
    return total


@njit(cache=True)
def _box_matrix(P, Q, P_boxes, Q_boxes):
    k = len(Q)
    bounds = np.empty((k, k))
    minimum_cells = max(P.shape[1], Q.shape[1])
    for i in range(k):
        for j in range(k):
            bounds[i, j] = max(_endpoint_cost(Q[i], P[j]),
                               _box_cost(Q_boxes[i], P_boxes[j], minimum_cells))
    return bounds


@njit(cache=True)
def _refine_vertex_bounds(P, Q, P_boxes, Q_boxes, bounds, cutoff):
    # Each vertex appears in a path cell. The two directional sums may overlap,
    # so combine them (and endpoints/box bounds) with max, never addition.
    for i in range(len(Q)):
        row_minimum = np.inf
        for j in range(len(P)):
            if bounds[i, j] <= cutoff:
                value = _vertex_box_cost(Q[i], P_boxes[j], cutoff)
                if value <= cutoff:
                    value = max(value, _vertex_box_cost(P[j], Q_boxes[i], cutoff))
                bounds[i, j] = max(bounds[i, j], value)
            row_minimum = min(row_minimum, bounds[i, j])
        if row_minimum > cutoff:
            return False
    return True


@njit(cache=True)
def _pruning_limit(limit):
    # Use the existing relative 1e-9 comparison margin only for global pruning.
    # Values and the public per-curve cutoff remain exact, not tolerance-clamped.
    return np.nextafter(limit * (1.0 + 1e-9), np.inf)


@njit(cache=True)
def _distance_upto(P, Q, limit):
    return _distance_with_boxes(P, Q, _set_boxes(P), _set_boxes(Q), limit)


@njit(cache=True)
def _distance_with_boxes(P, Q, P_boxes, Q_boxes, limit):
    """Exact within a conservatively widened cutoff, else inf; also report DP work."""
    k = len(Q)
    cutoff = _pruning_limit(limit)
    bounds = _box_matrix(P, Q, P_boxes, Q_boxes)
    for i in range(k):
        if np.min(bounds[i]) > cutoff:
            return np.inf, False
    for j in range(k):
        if np.min(bounds[:, j]) > cutoff:
            return np.inf, False
    row_of_col = np.empty(k, np.int64)
    col_of_row = np.empty(k, np.int64)
    parent = np.empty(k, np.int64)
    visited = np.empty(k, np.bool_)
    queue = np.empty(k, np.int64)
    if not core._has_perfect_matching(
            bounds, cutoff, row_of_col, col_of_row, parent, visited, queue):
        return np.inf, False
    if not _refine_vertex_bounds(P, Q, P_boxes, Q_boxes, bounds, cutoff):
        return np.inf, False
    if not core._has_perfect_matching(
            bounds, cutoff, row_of_col, col_of_row, parent, visited, queue):
        return np.inf, False

    lower = core.bottleneck(bounds)
    if not core._has_perfect_matching(
            bounds, lower, row_of_col, col_of_row, parent, visited, queue):
        raise ValueError("DTW lower-bound bottleneck failed to produce a matching")
    distances = np.full((k, k), np.inf)
    computed = np.zeros((k, k), np.bool_)
    upper = 0.0
    for i in range(k):
        j = col_of_row[i]
        value = _dtw_bounded(Q[i], P[j], cutoff)
        distances[i, j] = value
        bounds[i, j] = value
        computed[i, j] = True
        upper = max(upper, value)
    cutoff = _pruning_limit(min(upper, limit))
    if not core._has_perfect_matching(
            bounds, cutoff, row_of_col, col_of_row, parent, visited, queue):
        return np.inf, True

    # Seed distances give a feasible upper bound only if all its edges survived.
    # Uncomputed bounds never substitute for DTW distances.
    for i in range(k):
        for j in range(k):
            if not computed[i, j] and bounds[i, j] <= cutoff:
                value = _dtw_bounded(Q[i], P[j], cutoff)
                distances[i, j] = value
                bounds[i, j] = value
        if not core._has_perfect_matching(
                bounds, cutoff, row_of_col, col_of_row, parent, visited, queue):
            return np.inf, True
    return core.bottleneck(distances), True


@njit(cache=True)
def dtw_distance(P, Q):
    """Exact min-bijection/max-per-curve DTW distance, retaining multiplicities."""
    _validate_sets(P, Q)
    value, _ = _distance_upto(P, Q, np.inf)
    return value


@njit(cache=True, parallel=True)
def dtw_ground_truth(X, Qs, initial_ids):
    """Scan every parent in parallel over queries, using initial ids only as bounds.

    Return (ids, distances); ties select the lowest parent id. A planted parent
    does not certify nearest-neighbor status. All rejected edges/parents have
    lower bounds above an incumbent, or lack a perfect matching at that cutoff.
    Curve boxes are precomputed once for X and Qs, shared by all query scans.
    """
    _validate_collection_shapes(X, Qs)
    _validate_finite(X)
    _validate_finite(Qs)
    _validate_ids(initial_ids, len(X))
    if len(initial_ids) != len(Qs):
        raise ValueError("DTW initial ids must contain one parent per query")
    ids = np.empty(len(Qs), np.int64)
    distances = np.empty(len(Qs))
    X_boxes, Q_boxes = _collection_boxes(X), _collection_boxes(Qs)
    for q in prange(len(Qs)):
        initial = int(initial_ids[q])
        best, _ = _distance_with_boxes(X[initial], Qs[q], X_boxes[initial], Q_boxes[q], np.inf)
        best_id = initial
        for p in range(len(X)):
            if p == initial:
                continue
            value, _ = _distance_with_boxes(X[p], Qs[q], X_boxes[p], Q_boxes[q], best)
            if value < best or (value == best and p < best_id):
                best, best_id = value, p
        ids[q], distances[q] = best_id, best
    return ids, distances


@njit(cache=True)
def _verify_chunk(X, Q, Q_boxes, parents, best, best_id):
    checked = 0
    for parent in parents:
        p = int(parent)
        _validate_finite(X[p])
        value, evaluated = _distance_with_boxes(X[p], Q, _set_boxes(X[p]), Q_boxes, best)
        checked += int(evaluated)
        if value < best:
            best, best_id = value, p
    return best, best_id, checked


def dtw_verify(X, Q, parents, config, started):
    """Like fresh.regimes.verify: (best parent, distance, checked, timeout).

    ``checked`` counts parents reaching DTW DP after lower-bound/matching pruning.
    Deadline checks use actual elapsed time before each verification chunk and
    after the final chunk; a timed-out result never exposes a successful id.
    """
    chunk = operator.index(config["verification_chunk"])
    if chunk <= 0:
        raise ValueError("DTW verification_chunk must be positive")
    deadline = float(config["query_deadline_seconds"])
    if np.isnan(deadline) or deadline < 0 or not np.isfinite(started):
        raise ValueError("DTW deadline must be nonnegative and started must be finite")
    _validate_collection_shapes(X, Q[None])
    _validate_finite(Q)
    _validate_ids(parents, len(X))
    Q_boxes = _set_boxes(Q)
    best, best_id, checked = np.inf, -1, 0
    for start in range(0, len(parents), chunk):
        if time.perf_counter() - started > deadline:
            return -1, np.inf, checked, True
        best, best_id, count = _verify_chunk(
            X, Q, Q_boxes, parents[start:start + chunk], best, best_id)
        checked += count
    timeout = time.perf_counter() - started > deadline
    return int(best_id) if not timeout else -1, float(best), checked, timeout
