"""Families of orders (permutations) stored and probed by the indexes of Section 5.

An order sigma is a row of length k: the key of a curve set P listed in order sigma holds curve P[sigma[i]] in
slot i. Storing sigma and probing tau tests the matching pi = sigma o tau^{-1}, which sends query curve tau[i]
to data curve sigma[i] for every slot i.

Families are int8 arrays of shape (count, k) whose rows are sorted lexicographically, so that consecutive rows
share long prefixes; the query kernels reuse the hash of the shared prefix. Generation runs in Numba.
"""
from math import factorial, prod

import numpy as np
from numba import njit

MAX_K = 16  # rows are packed into 64-bit codes with 4 bits per entry


@njit(cache=True)
def _next_perm(a, lo, hi):
    """Next lexicographic permutation of a[lo:hi] in place (distinct permutations of a multiset); False if the
    segment was the last one, in which case it is left unchanged."""
    i = hi - 2
    while i >= lo and a[i] >= a[i + 1]:
        i -= 1
    if i < lo:
        return False
    j = hi - 1
    while a[j] <= a[i]:
        j -= 1
    t = a[i]
    a[i] = a[j]
    a[j] = t
    lo2, hi2 = i + 1, hi - 1
    while lo2 < hi2:
        t = a[lo2]
        a[lo2] = a[hi2]
        a[hi2] = t
        lo2 += 1
        hi2 -= 1
    return True


@njit(cache=True)
def _gen_all(k, count):
    out = np.empty((count, k), np.int8)
    a = np.arange(k).astype(np.int8)
    for s in range(count):
        out[s, :] = a
        _next_perm(a, 0, k)
    return out


@njit(cache=True)
def _gen_within(k, starts, ends, count):
    """Rows that permute every block of slots within itself, as an odometer over the blocks (lexicographic)."""
    out = np.empty((count, k), np.int8)
    a = np.arange(k).astype(np.int8)
    nb = starts.shape[0]
    for s in range(count):
        out[s, :] = a
        j = nb - 1
        while j >= 0:
            if _next_perm(a, starts[j], ends[j]):
                break
            lo, hi = starts[j], ends[j] - 1  # the block is descending: reset it to ascending
            while lo < hi:
                t = a[lo]
                a[lo] = a[hi]
                a[hi] = t
                lo += 1
                hi -= 1
            j -= 1
    return out


@njit(cache=True)
def _gen_alloc(k, sizes, starts, count):
    """Allocations: rows that are increasing within every block. Each distinct sequence of block labels of the
    query curves gives one row; rows are packed into 64-bit codes and sorted, which sorts them lexicographically."""
    nb = sizes.shape[0]
    lab = np.empty(k, np.int8)
    pos = 0
    for j in range(nb):
        for _ in range(sizes[j]):
            lab[pos] = j
            pos += 1
    codes = np.empty(count, np.uint64)
    row = np.empty(k, np.int8)
    cnt = np.empty(nb, np.int64)
    for s in range(count):
        cnt[:] = 0
        for x in range(k):
            j = lab[x]
            row[starts[j] + cnt[j]] = x
            cnt[j] += 1
        code = np.uint64(0)
        for i in range(k):
            code = (code << np.uint64(4)) | np.uint64(row[i])
        codes[s] = code
        _next_perm(lab, 0, k)
    codes.sort()
    out = np.empty((count, k), np.int8)
    for s in range(count):
        c = codes[s]
        for i in range(k - 1, -1, -1):
            out[s, i] = np.int8(c & np.uint64(15))
            c = c >> np.uint64(4)
    return out


def identity(k):
    return np.arange(k, dtype=np.int8)[None, :]


def all_orders(k):
    assert k <= MAX_K
    return _gen_all(k, factorial(k))


def blocks_of(sizes):
    """Consecutive slot blocks B_1, ..., B_l with the given sizes."""
    out, start = [], 0
    for b in sizes:
        out.append(list(range(start, start + b)))
        start += b
    return out


def _starts_ends(sizes):
    starts = np.cumsum([0] + list(sizes[:-1])).astype(np.int64)
    ends = starts + np.asarray(sizes, dtype=np.int64)
    return starts, ends


def within_block_orders(sizes):
    """U = {sigma : sigma(B_j) = B_j for every j}; |U| = prod b_j!."""
    k = sum(sizes)
    assert k <= MAX_K
    starts, ends = _starts_ends(sizes)
    return _gen_within(k, starts, ends, prod(factorial(b) for b in sizes))


def allocations(sizes):
    """T: for every ordered partition (T_1, ..., T_l) of the query indices with |T_j| = b_j, the order that maps
    the slots of B_j, in increasing order, to the indices of T_j, in increasing order; |T| = k!/prod b_j!."""
    k = sum(sizes)
    assert k <= MAX_K
    starts, _ = _starts_ends(sizes)
    count = factorial(k) // prod(factorial(b) for b in sizes)
    return _gen_alloc(k, np.asarray(sizes, dtype=np.int64), starts, count)


def partitions(k, largest=None):
    largest = k if largest is None else largest
    if k == 0:
        yield ()
        return
    for p in range(min(k, largest), 0, -1):
        for rest in partitions(k - p, p):
            yield (p,) + rest


def blocking_points(k):
    """All (stored, probed, sizes, swapped) choices of block sizes with either role, deduplicated by cost."""
    K, pts = factorial(k), {}
    for sizes in partitions(k):
        a = prod(factorial(s) for s in sizes)
        pts.setdefault((a, K // a), (sizes, False))
    for sizes in partitions(k):
        a = prod(factorial(s) for s in sizes)
        pts.setdefault((K // a, a), (sizes, True))
    return [(a, b, sizes, swapped) for (a, b), (sizes, swapped) in sorted(pts.items())]


def balanced_blocking(k):
    """Most balanced block sizes: minimize max(stored, probed), then the sum; standard orientation first."""
    return min(blocking_points(k), key=lambda t: (max(t[0], t[1]), t[0] + t[1], t[3], t[0]))


def blocking_families(sizes, swapped=False):
    """(stored, probed) families of blocking; swapped stores the allocations and probes U."""
    if swapped:
        return allocations(sizes), within_block_orders(sizes)
    return within_block_orders(sizes), allocations(sizes)


def compose_inverse(sigma, tau):
    """pi = sigma o tau^{-1} as an array: pi[tau[i]] = sigma[i]."""
    pi = np.empty_like(sigma)
    pi[tau] = sigma
    return pi
