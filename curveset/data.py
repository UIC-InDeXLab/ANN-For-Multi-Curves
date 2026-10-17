"""Synthetic workloads for the experiments.

Collections are float64 arrays of shape (n, k, m, d). Every generator takes a numpy Generator for
reproducibility. Distances are in the same units as the search radius r.
"""
import numpy as np


def random_walks(rng, N, m, d, extent, step):
    """N random-walk trajectories: start uniform in [0, extent]^d, Gaussian steps with std `step`."""
    start = rng.uniform(0, extent, size=(N, 1, d))
    steps = rng.normal(0, step, size=(N, m - 1, d))
    return np.concatenate([start, start + np.cumsum(steps, axis=1)], axis=1)


def collection(rng, n, k, m, d, extent, step):
    """n curve sets of k independent random-walk trajectories."""
    return random_walks(rng, n * k, m, d, extent, step).reshape(n, k, m, d)


def jitter(rng, C, rho):
    """Move every vertex by a vector drawn uniformly from the ball of radius rho; the identity traversal then
    gives d_F(C, result) <= rho."""
    shape = C.shape[:-1]
    d = C.shape[-1]
    v = rng.normal(size=shape + (d,))
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    radius = rho * rng.uniform(0, 1, size=shape + (1,)) ** (1.0 / d)
    return C + v * radius


def planted_queries(rng, X, nq, rho):
    """Queries built from random data sets: shuffle the curves (unknown matching) and jitter every vertex by at
    most rho, so the planted set is within d_match <= rho of its query."""
    n, k = X.shape[0], X.shape[1]
    targets = rng.choice(n, size=nq, replace=False)
    Q = np.empty((nq,) + X.shape[1:])
    for q, t in enumerate(targets):
        Q[q] = jitter(rng, X[t, rng.permutation(k)], rho)
    return Q, targets


def add_near_misses(rng, X, targets, radii):
    """Append, for every target, one jittered copy per radius (shuffled), creating curve sets at distance
    roughly radius from the target. Returns the extended collection."""
    extra = []
    k = X.shape[1]
    for t in targets:
        for rad in radii:
            extra.append(jitter(rng, X[t, rng.permutation(k)], rad))
    return np.concatenate([X, np.array(extra)], axis=0) if extra else X


def motif_collection(rng, n, k, m, d, extent, step, n_motifs, rho, planted_per_query, nq):
    """Workload for the per-curve baseline (Q5).

    Every curve of every data set is a jittered copy (radius rho) of one of n_motifs motif trajectories,
    chosen uniformly, so every motif appears in about a k/n_motifs fraction of the sets. Queries are k distinct
    motifs; only the `planted_per_query` sets planted for a query contain all k of its motifs (up to chance).
    Returns (X, Q, motif ids of the queries).
    """
    motifs = random_walks(rng, n_motifs, m, d, extent, step)
    choice = rng.integers(0, n_motifs, size=(n, k))
    X = jitter(rng, motifs[choice], rho)
    Q = np.empty((nq, k, m, d))
    qm = np.empty((nq, k), dtype=np.int64)
    for q in range(nq):
        ids = rng.choice(n_motifs, size=k, replace=False)
        qm[q] = ids
        Q[q] = jitter(rng, motifs[ids], rho)
        for _ in range(planted_per_query):
            slot = rng.integers(0, n)
            X[slot] = jitter(rng, motifs[ids[rng.permutation(k)]], rho)
    return X, Q, qm


def crowd_collection(rng, n, k, m, d, extent, step, n_paths, rho, zipf=0.0):
    """Crowd workload: every object follows one of n_paths popular paths (random walks), chosen i.i.d. with
    popularity proportional to 1/rank^zipf, and is a jittered copy (radius rho per vertex) of its path. With
    n_paths close to k, every path occurs in most curve sets and several objects of a set share a path, like
    pedestrians walking in groups along the few paths of a scene."""
    paths = random_walks(rng, n_paths, m, d, extent, step)
    w = 1.0 / np.arange(1, n_paths + 1) ** zipf
    choice = rng.choice(n_paths, size=(n, k), p=w / w.sum())
    return jitter(rng, paths[choice], rho)


def spread_collection(rng, n, k, m, d, delta, step_frac=0.05):
    """Adversarial placement for a shared grid (Proposition 5.3): curve i of every set is shifted by i*delta/k
    along the first axis, so the cell boundaries that cut different matched pairs are spread over the cell.
    Curves are short random walks (extent much smaller than delta) placed at random multiples of delta."""
    base = rng.integers(0, 1000, size=(n, 1, 1, d)).astype(float) * delta
    offs = np.zeros((1, k, 1, d))
    offs[0, :, 0, 0] = np.arange(k) * delta / k
    walk = random_walks(rng, n * k, m, d, 0.0, step_frac * delta).reshape(n, k, m, d)
    return base + offs + walk


def formation_collection(rng, n, k, m, d, delta, length_frac=0.5, wiggle_frac=0.002):
    """k objects walking side by side in formation: object i is offset by i*delta/k across the direction of
    motion and every object walks the same straight path (plus a small wiggle). Across the motion, the cell
    boundaries that separate different matched pairs are spread evenly over the cell, which is the placement
    that a single shared grid handles worst; along the motion, the pairs are aligned."""
    base = rng.integers(0, 1000, size=(n, 1, 1, d)).astype(float) * delta
    base[..., 1:] += rng.uniform(0, delta, size=(n, 1, 1, d - 1))
    offs = np.zeros((1, k, 1, d))
    offs[0, :, 0, 0] = np.arange(k) * delta / k
    path = np.zeros((1, 1, m, d))
    path[0, 0, :, 1] = np.linspace(0, length_frac * delta, m)
    wiggle = np.cumsum(rng.normal(0, wiggle_frac * delta, size=(n, k, m, d)), axis=2)
    return base + offs + path + wiggle
