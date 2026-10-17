"""Independent full-DP/permutation oracles for cumulative Euclidean DTW."""
from itertools import permutations
import math
import time
from unittest.mock import patch

import numpy as np
import pytest

from curveset import core
from fresh import dtw as dtw_module
from fresh.dtw import (dtw, dtw_bounded, dtw_distance, dtw_endpoint_lb,
                       dtw_ground_truth, dtw_verify)


def reference_dtw(A, B):
    costs = [[math.inf] * (len(B) + 1) for _ in range(len(A) + 1)]
    costs[0][0] = 0.0
    for i, a in enumerate(A, 1):
        for j, b in enumerate(B, 1):
            costs[i][j] = math.dist(a, b) + min(
                costs[i - 1][j], costs[i][j - 1], costs[i - 1][j - 1])
    return costs[-1][-1]


def reference_distance(P, Q):
    return min(max(reference_dtw(Q[i], P[j]) for i, j in enumerate(order))
               for order in permutations(range(len(P))))


def reference_endpoint_lb(P, Q):
    def cost(A, B):
        start = math.dist(A[0], B[0])
        return start if len(A) == len(B) == 1 else start + math.dist(A[-1], B[-1])

    return min(max(cost(Q[i], P[j]) for i, j in enumerate(order))
               for order in permutations(range(len(P))))


def reference_box_bounds(A, B):
    A_lower, A_upper = A.min(axis=0), A.max(axis=0)
    B_lower, B_upper = B.min(axis=0), B.max(axis=0)
    gaps = [max(0.0, a_low - b_high, b_low - a_high)
            for a_low, a_high, b_low, b_high in zip(A_lower, A_upper, B_lower, B_upper)]
    box = math.dist(gaps, [0.0] * len(gaps)) * max(len(A), len(B))

    def vertex_sum(C, lower, upper):
        return sum(math.dist(vertex, [min(max(x, lo), hi) for x, lo, hi in
                                      zip(vertex, lower, upper)]) for vertex in C)

    return box, vertex_sum(A, B_lower, B_upper), vertex_sum(B, A_lower, A_upper)


def combined_bounds(P, Q):
    P_boxes, Q_boxes = dtw_module._set_boxes(P), dtw_module._set_boxes(Q)
    bounds = dtw_module._box_matrix(P, Q, P_boxes, Q_boxes)
    assert dtw_module._refine_vertex_bounds(P, Q, P_boxes, Q_boxes, bounds, np.inf)
    return bounds


def curve(values):
    return np.array(values, dtype=float).reshape(-1, 1)


def config(chunk=2, deadline=60.0):
    return dict(verification_chunk=chunk, query_deadline_seconds=deadline)


@pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
def test_curve_dp_matches_independent_reference(dimensions):
    rng = np.random.default_rng(120 + dimensions)
    for _ in range(50):
        A = rng.normal(size=(int(rng.integers(1, 8)), dimensions))
        B = rng.normal(size=(int(rng.integers(1, 8)), dimensions))
        expected = reference_dtw(A, B)
        assert dtw(A, B) == pytest.approx(expected, rel=1e-13, abs=1e-13)
        assert dtw(B, A) == pytest.approx(expected, rel=1e-13, abs=1e-13)
        for limit in (0.0, expected * .5, expected, expected * 1.01, np.inf):
            value = dtw_bounded(A, B, limit)
            if expected <= limit or math.isclose(expected, limit, rel_tol=1e-14):
                if math.isfinite(value):
                    assert value == pytest.approx(expected, rel=1e-13, abs=1e-13)
                else:
                    assert dtw(A, B) > limit
            else:
                assert value == np.inf


@pytest.mark.parametrize("A,B,expected", [
    ([0], [3], 3.0),
    ([0], [1, 2, 3], 6.0),
    ([1, 2, 3], [0], 6.0),
    ([0, 0, 1], [0, 1, 1], 0.0),
    ([0, 1], [1, 2], 2.0),
    ([0, 10, 0], [0, 0, 0], 10.0),
    ([0, 0, 0], [2, 2, 2], 6.0),
    ([0, 1, 2], [0, 0, 1, 2], 0.0),
])
def test_singletons_repeated_vertices_and_interior_costs(A, B, expected):
    A, B = curve(A), curve(B)
    assert dtw(A, B) == expected
    assert dtw_bounded(A, B, expected) == expected
    assert dtw_bounded(A, B, np.nextafter(expected, -np.inf)) == np.inf
    assert dtw_bounded(A, B, np.nextafter(expected, np.inf)) == expected


def test_euclidean_local_cost_is_not_squared_or_normalized():
    A = np.zeros((3, 2))
    B = np.tile([3.0, 4.0], (3, 1))
    assert dtw(A, B) == 15.0
    assert dtw_bounded(A, B, 14.0) == np.inf


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("dimensions", [1, 2, 4])
def test_global_bottleneck_and_bounds_match_exhaustive_permutations(k, dimensions):
    rng = np.random.default_rng(k * 100 + dimensions)
    for _ in range(12):
        P = rng.normal(size=(k, int(rng.integers(1, 6)), dimensions))
        Q = rng.normal(size=(k, int(rng.integers(1, 6)), dimensions))
        expected = reference_distance(P, Q)
        assert dtw_distance(P, Q) == pytest.approx(expected, rel=1e-13, abs=1e-13)
        assert dtw_distance(Q, P) == pytest.approx(expected, rel=1e-13, abs=1e-13)
        bound = dtw_endpoint_lb(P, Q)
        assert bound == pytest.approx(reference_endpoint_lb(P, Q), rel=1e-13, abs=1e-13)
        assert bound <= expected + 1e-12
        assert dtw_distance(P[rng.permutation(k)], Q[rng.permutation(k)]) == pytest.approx(expected)


def test_bottleneck_not_minimum_sum_assignment():
    # Distances [[5, 0], [8, 5]]: sum assignment picks max 8, bottleneck picks 5.
    P = np.array([[[0.0]], [[5.0]]])
    Q = np.array([[[5.0]], [[10.0]]])
    assert dtw_distance(P, Q) == 5.0
    pairs = np.array([[reference_dtw(A, B) for B in P] for A in Q])
    sum_choice = min(permutations(range(2)), key=lambda p: sum(pairs[i, p[i]] for i in range(2)))
    # This line example has a sum tie; use a planar strict counterexample below.
    assert max(pairs[i, sum_choice[i]] for i in range(2)) >= 5.0
    P = np.array([[[0.0, 0.0]], [[5.0, 0.0]]])
    Q = np.array([[[5.0, 0.0]], [[5.0, 5.0]]])
    pairs = np.array([[reference_dtw(A, B) for B in P] for A in Q])
    sum_choice = min(permutations(range(2)), key=lambda p: sum(pairs[i, p[i]] for i in range(2)))
    assert max(pairs[i, sum_choice[i]] for i in range(2)) == pytest.approx(math.sqrt(50))
    assert dtw_distance(P, Q) == 5.0


def test_multiplicities_require_a_bijection_and_matching_pruning():
    Q = np.array([curve([0, 0]), curve([0, 0]), curve([10, 10])])
    P = np.array([curve([0, 0]), curve([10, 10]), curve([10, 10])])
    assert dtw_endpoint_lb(P, Q) == 20.0
    assert dtw_distance(P, Q) == 20.0
    # Every row and column has a zero edge, but no perfect matching at zero.
    X = np.stack([Q, P])
    ids, distances = dtw_ground_truth(X, Q[None], np.array([1]))
    assert ids.tolist() == [0]
    assert distances.tolist() == [0.0]
    p, value, checked, timeout = dtw_verify(X, Q, np.array([0, 1]), config(), time.perf_counter())
    assert (p, value, checked, timeout) == (0, 0.0, 1, False)


def test_endpoints_do_not_replace_cumulative_dp_or_skip_equal_edges():
    P, Q = curve([0, 1])[None], curve([1, 2])[None]
    assert core.dmatch(P, Q) == 1.0
    assert dtw_endpoint_lb(P, Q) == 2.0
    assert dtw_distance(P, Q) == 2.0
    P, Q = curve([0, 10, 0])[None], curve([0, 0, 0])[None]
    assert dtw_endpoint_lb(P, Q) == 0.0
    assert dtw_distance(P, Q) == 10.0
    for k in (1, 2, 4):
        P = np.tile(curve([0, 1])[None], (k, 1, 1))
        Q = np.tile(curve([1, 2])[None], (k, 1, 1))
        assert dtw_distance(P, Q) == 2.0
        assert dtw_distance(P, P[::-1]) == 0.0


def test_endpoint_bound_counts_once_only_when_both_singletons():
    P, Q = curve([0])[None], curve([3])[None]
    assert dtw_endpoint_lb(P, Q) == dtw_distance(P, Q) == 3.0
    P, Q = curve([0])[None], curve([3, 3])[None]
    assert dtw_endpoint_lb(P, Q) == dtw_distance(P, Q) == 6.0


@pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
def test_each_box_bound_is_sound_against_full_dp(dimensions):
    rng = np.random.default_rng(1700 + dimensions)
    for _ in range(80):
        A = rng.normal(size=(int(rng.integers(1, 8)), dimensions))
        B = rng.normal(size=(int(rng.integers(1, 8)), dimensions))
        if rng.random() < .5:
            B += 5.0
        A_box, B_box = dtw_module._curve_box(A), dtw_module._curve_box(B)
        np.testing.assert_array_equal(A_box, np.stack([A.min(axis=0), A.max(axis=0)]))
        np.testing.assert_array_equal(B_box, np.stack([B.min(axis=0), B.max(axis=0)]))
        expected = reference_box_bounds(A, B)
        actual = (dtw_module._box_cost(A_box, B_box, max(len(A), len(B))),
                  dtw_module._vertex_box_cost(A, B_box, np.inf),
                  dtw_module._vertex_box_cost(B, A_box, np.inf))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        assert max(actual) <= reference_dtw(A, B) + 1e-12
        for limit in (0.0, expected[1] / 2, expected[1], np.inf):
            partial = dtw_module._vertex_box_cost(A, B_box, limit)
            assert partial <= expected[1] + 1e-12
            if expected[1] > limit + 1e-12:
                assert partial > limit
            else:
                assert partial == pytest.approx(expected[1], rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("A,B,box,forward,reverse,combined", [
    ([0], [3], 3.0, 3.0, 3.0, 3.0),
    ([0], [3, 3, 3], 9.0, 3.0, 9.0, 9.0),
    ([3, 3, 3], [0], 9.0, 9.0, 3.0, 9.0),
    ([0, 0, 0], [1, 1, 1], 3.0, 3.0, 3.0, 3.0),
    ([0, 4, 4, 0], [0, 0, 0, 0], 0.0, 8.0, 0.0, 8.0),
    ([0, 10, 0], [0, -10, 0], 0.0, 10.0, 10.0, 10.0),
    ([0, 0, 1], [0, 1, 1], 0.0, 0.0, 0.0, 0.0),
])
def test_box_bounds_singletons_repetition_overlap_and_no_double_count(
        A, B, box, forward, reverse, combined):
    A, B = curve(A), curve(B)
    A_box, B_box = dtw_module._curve_box(A), dtw_module._curve_box(B)
    assert dtw_module._box_cost(A_box, B_box, max(len(A), len(B))) == box
    assert dtw_module._vertex_box_cost(A, B_box, np.inf) == forward
    assert dtw_module._vertex_box_cost(B, A_box, np.inf) == reverse
    assert combined_bounds(A[None], B[None])[0, 0] == combined
    assert combined <= reference_dtw(A, B)
    assert dtw_distance(A[None], B[None]) == reference_dtw(A, B)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_combined_bottleneck_bound_against_exhaustive_permutations(k):
    rng = np.random.default_rng(2300 + k)
    for _ in range(20):
        dimensions = int(rng.integers(1, 5))
        P = rng.normal(size=(k, int(rng.integers(1, 7)), dimensions))
        Q = rng.normal(size=(k, int(rng.integers(1, 7)), dimensions))
        P += rng.integers(-3, 4, (k, 1, dimensions))
        bounds = combined_bounds(P, Q)
        for i in range(k):
            for j in range(k):
                endpoint = reference_endpoint_lb(P[j:j+1], Q[i:i+1])
                expected = max(endpoint, *reference_box_bounds(Q[i], P[j]))
                assert bounds[i, j] == pytest.approx(expected, rel=1e-13, abs=1e-13)
                assert bounds[i, j] <= reference_dtw(Q[i], P[j]) + 1e-12
        lower = min(max(bounds[i, j] for i, j in enumerate(order))
                    for order in permutations(range(k)))
        assert lower >= dtw_endpoint_lb(P, Q) - 1e-12
        assert lower <= reference_distance(P, Q) + 1e-12


def test_whole_box_and_vertex_box_bounds_prune_before_any_dp():
    Q = np.zeros((1, 6, 1))
    initial = curve([0, 0, 1, 1, 0, 0])[None]
    for rejected in (curve([1, 1, 1, 1, 1, 1])[None],
                     curve([0, 4, 4, 4, 0, 0])[None]):
        assert dtw_endpoint_lb(rejected, Q) <= 2.0
        value, checked = dtw_module._distance_upto(rejected, Q, 2.0)
        assert value == np.inf and checked is False
        X = np.stack([initial, rejected])
        assert dtw_verify(X, Q, np.array([0, 1]), config(), time.perf_counter()) == (
            0, 2.0, 1, False)
        ids, distances = dtw_ground_truth(X, Q[None], np.array([1]))
        assert ids.tolist() == [0] and distances.tolist() == [2.0]


def test_vertex_box_matching_prunes_hall_failure_with_covered_rows_and_columns():
    flat, excursion = curve([0, 0, 0, 0]), curve([0, 10, 10, 0])
    Q = np.array([flat, flat, excursion])
    P = np.array([flat, excursion, excursion])
    assert dtw_endpoint_lb(P, Q) == 0.0
    bounds = combined_bounds(P, Q)
    assert np.all(bounds.min(axis=0) == 0) and np.all(bounds.min(axis=1) == 0)
    assert dtw_module._distance_upto(P, Q, 0.0) == (np.inf, False)
    assert dtw_distance(P, Q) == reference_distance(P, Q) == 20.0
    X = np.stack([Q, P])
    assert dtw_verify(X, Q, np.array([0, 1]), config(), time.perf_counter()) == (
        0, 0.0, 1, False)


@pytest.mark.parametrize("lengths", [(1, 1), (1, 37), (37, 1), (37, 37)])
@pytest.mark.parametrize("shift", [0.0, .1, 1e-12, 1e8])
def test_box_cutoff_ties_and_roundoff_preserve_exact_distances(lengths, shift):
    mp, mq = lengths
    P, Q = np.full((2, mp, 2), shift), np.zeros((2, mq, 2))
    exact = dtw(P[0], Q[0])
    bounds = combined_bounds(P, Q)
    assert np.all(bounds <= exact * (1 + 1e-12))
    value, checked = dtw_module._distance_upto(P, Q, exact)
    assert value == exact and checked is True
    assert dtw_distance(P, Q) == exact
    X = np.stack([P, P, P])
    ids, distances = dtw_ground_truth(X, Q[None], np.array([2]))
    assert ids.tolist() == [0] and distances.tolist() == [exact]


def test_ground_truth_precomputes_boxes_once_per_collection():
    rng = np.random.default_rng(9100)
    X, Qs = rng.normal(size=(5, 3, 4, 2)), rng.normal(size=(2, 3, 3, 2))
    expected = np.array([[reference_distance(P, Q) for P in X] for Q in Qs])
    initial = np.argmax(expected, axis=1)
    with patch("fresh.dtw._collection_boxes", wraps=dtw_module._collection_boxes) as build_boxes:
        ids, distances = dtw_ground_truth.py_func(X, Qs, initial)
    assert build_boxes.call_count == 2
    assert build_boxes.call_args_list[0].args[0] is X
    assert build_boxes.call_args_list[1].args[0] is Qs
    assert ids.tolist() == np.argmin(expected, axis=1).tolist()
    np.testing.assert_allclose(distances, expected.min(axis=1), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("k,dimensions", [(1, 1), (2, 2), (3, 3), (4, 1)])
def test_parallel_ground_truth_is_exhaustive_not_the_initial_parent(k, dimensions):
    rng = np.random.default_rng(900 + k)
    X = rng.normal(size=(10, k, 4, dimensions))
    Qs = rng.normal(size=(4, k, 3, dimensions))
    expected = np.array([[reference_distance(P, Q) for P in X] for Q in Qs])
    initial = np.argmax(expected, axis=1)
    ids, distances = dtw_ground_truth(X, Qs, initial)
    assert ids.dtype == np.int64 and distances.dtype == np.float64
    assert ids.tolist() == np.argmin(expected, axis=1).tolist()
    np.testing.assert_allclose(distances, expected.min(axis=1), rtol=1e-13, atol=1e-13)
    assert all(ids != initial)


def test_ground_truth_strict_improvement_inside_endpoint_bound():
    Q = curve([0, 0, 0])[None]
    X = np.array([[curve([0, 2, 0])], [curve([0, 1, 0])], [curve([0, 3, 0])]])
    ids, values = dtw_ground_truth(X, Q[None], np.array([0]))
    assert ids.tolist() == [1] and values.tolist() == [1.0]
    X = np.array([[curve([1, 1, 1])], [curve([1, 0, 1])], [curve([2, 2, 2])]])
    ids, values = dtw_ground_truth(X, Q[None], np.array([0]))
    assert ids.tolist() == [1] and values.tolist() == [2.0]


def test_ground_truth_ties_at_zero_and_nonzero_cutoffs():
    for shift in (0.0, 1.0, 1e-12, 1e8):
        Q = np.zeros((2, 3, 2))
        X = np.full((3, 2, 3, 2), shift)
        ids, distances = dtw_ground_truth(X, Q[None], np.array([2]))
        assert ids.tolist() == [0]
        assert distances[0] == pytest.approx(reference_distance(X[0], Q), abs=1e-20)


def test_nearest_dtw_parent_differs_from_nearest_frechet_parent():
    Q = curve([0, 0, 0])[None]
    X = np.array([[curve([1, 1, 1])], [curve([0, 2, 0])]])
    assert np.argmin([core.dmatch(P, Q) for P in X]) == 0
    ids, values = dtw_ground_truth(X, Q[None], np.array([0]))
    assert ids.tolist() == [1] and values.tolist() == [2.0]


@pytest.mark.parametrize("chunk", [1, 3, 20])
def test_verification_matches_exhaustive_candidate_scan(chunk):
    rng = np.random.default_rng(773)
    X = rng.normal(size=(12, 3, 4, 2))
    Q = rng.normal(size=(3, 5, 2))
    parents = np.array([8, 2, 11, 0, 4, 5, 1])
    expected = [reference_distance(X[p], Q) for p in parents]
    p, value, checked, timeout = dtw_verify(X, Q, parents, config(chunk), time.perf_counter())
    assert p == parents[np.argmin(expected)]
    assert value == pytest.approx(min(expected), rel=1e-13, abs=1e-13)
    assert 1 <= checked <= len(parents)
    assert timeout is False


def test_verification_scalar_pruning_avoids_far_parents():
    Q = np.zeros((1, 3, 1))
    X = np.array([[curve([0, 0, 0])], [curve([100, 100, 100])]])
    assert dtw_verify(X, Q, np.array([0, 1]), config(), time.perf_counter()) == (0, 0.0, 1, False)


def test_verification_empty_candidates_do_not_claim_a_success():
    X = np.zeros((2, 1, 2, 1))
    assert dtw_verify(X, X[0], np.array([], np.int64), config(), time.perf_counter()) == (
        -1, np.inf, 0, False)


def test_verification_real_elapsed_deadline_is_not_a_candidate_count():
    X = np.zeros((2, 1, 2, 1))
    assert dtw_verify(X, X[0], np.array([0, 1]), config(deadline=.001),
                      time.perf_counter() - 1) == (-1, np.inf, 0, True)


def test_verification_checks_deadline_between_configured_chunks():
    X = np.zeros((3, 1, 2, 1))
    with patch("fresh.dtw.time.perf_counter", side_effect=[0.0, 2.0]):
        result = dtw_verify(X, X[0], np.array([0, 1, 2]), config(2, 1.0), 0.0)
    assert result == (-1, np.inf, 2, True)


def test_verification_checks_deadline_after_last_chunk():
    X = np.zeros((2, 1, 2, 1))
    with patch("fresh.dtw.time.perf_counter", side_effect=[0.0, 2.0]):
        result = dtw_verify(X, X[0], np.array([0, 1]), config(2, 1.0), 0.0)
    assert result == (-1, 0.0, 2, True)


@pytest.mark.parametrize("A,B", [
    (np.zeros((0, 1)), np.zeros((2, 1))),
    (np.zeros((2, 1)), np.zeros((0, 1))),
    (np.zeros((2, 0)), np.zeros((2, 0))),
    (np.zeros((2, 1)), np.zeros((2, 2))),
    (np.array([[np.nan]]), np.zeros((2, 1))),
    (np.zeros((2, 1)), np.array([[np.inf]])),
])
def test_invalid_curves_raise_explicit_value_errors(A, B):
    with pytest.raises(ValueError):
        dtw(A, B)
    with pytest.raises(ValueError):
        dtw_bounded(A, B, 1.0)


@pytest.mark.parametrize("P,Q", [
    (np.zeros((0, 2, 1)), np.zeros((0, 2, 1))),
    (np.zeros((1, 0, 1)), np.zeros((1, 2, 1))),
    (np.zeros((1, 2, 1)), np.zeros((1, 0, 1))),
    (np.zeros((1, 2, 1)), np.zeros((1, 2, 2))),
    (np.zeros((1, 2, 0)), np.zeros((1, 2, 0))),
    (np.zeros((1, 2, 1)), np.zeros((2, 2, 1))),
])
def test_invalid_multicurves_raise_explicit_value_errors(P, Q):
    for function in (dtw_distance, dtw_endpoint_lb):
        with pytest.raises(ValueError):
            function(P, Q)


def test_nan_cutoff_is_explicit_error_but_negative_cutoff_is_infeasible():
    A = curve([0])
    with pytest.raises(ValueError, match="NaN"):
        dtw_bounded(A, A, np.nan)
    assert dtw_bounded(A, A, -1.0) == np.inf


@pytest.mark.parametrize("ids", [
    np.array([-1]), np.array([3]), np.array([.5]), np.array([np.nan]),
    np.array([], np.int64), np.array([0, 1]),
])
def test_ground_truth_rejects_invalid_initial_ids(ids):
    X = np.zeros((3, 1, 2, 1))
    with pytest.raises(ValueError):
        dtw_ground_truth(X, X[:1], ids)


def test_ground_truth_rejects_empty_data_and_accepts_empty_queries():
    X = np.zeros((2, 1, 2, 1))
    with pytest.raises(ValueError):
        dtw_ground_truth(X[:0], X[:1], np.array([0]))
    ids, values = dtw_ground_truth(X, X[:0], np.array([], np.int64))
    assert ids.shape == values.shape == (0,)


@pytest.mark.parametrize("parents", [np.array([-1]), np.array([2]), np.array([.25])])
def test_verification_rejects_invalid_parents(parents):
    X = np.zeros((2, 1, 2, 1))
    with pytest.raises(ValueError):
        dtw_verify(X, X[0], parents, config(), time.perf_counter())


@pytest.mark.parametrize("options", [
    config(0), config(-1), config(deadline=-1.0), config(deadline=np.nan),
])
def test_verification_rejects_invalid_config(options):
    X = np.zeros((2, 1, 2, 1))
    with pytest.raises(ValueError):
        dtw_verify(X, X[0], np.array([0]), options, time.perf_counter())
