# Release validation

Validated on Linux with Python 3.14.8. This is a functional release check, not a
rerun of the large timing experiments.

## Environment

| Package | Version |
|---|---|
| NumPy | 2.4.4 |
| Numba | 0.65.1 |
| llvmlite | 0.47.0 |
| Matplotlib | 3.10.8 |
| Pillow | 12.2.0 |
| ijson | 3.6.0 |
| psutil | 7.2.2 |
| pytest | 9.1.1 |
| pandas | 3.0.6 |
| SciPy | 1.18.1 |

## Checks completed

* Full test suite with two Numba threads and single-thread BLAS:
  **377 passed, 3 skipped, 55 subtests passed**. The skipped checks require the
  original full NBA query logs or the separate public pilot corpus.
* Independent synthetic demo: all four indexes returned the nearest neighbor
  found by exhaustive enumeration of all curve matchings.
* Subprocess benchmark smoke run: common, rare, and repeated collections
  completed. Each used 240 multi-curves, k=3, one index seed, eight tuning queries,
  and twelve test queries. All main methods and the exact-reference pass
  completed.
* Report checksum verification passed. Seventeen tables were regenerated and
  the recorded scaling table retained. The table bodies agree with the
  distributed versions, ignoring generated-file comments and purely presentational
  spacing.
* All eight experimental plot PDFs were regenerated from the released aggregate
  measurements with the paper's fonts.
* The trajectories-only NBA animation renderer was exercised against the actual
  recorded five-player query 327, seed 2301, and completed blocking answer.
* Local video composition was tested using a generated solid-color clip, not
  broadcast footage. The output contained twelve frames at the requested fps.
* Release text and figure metadata were checked for personal author names,
  absolute home-directory paths, and credentials.

## Not rerun here

* The full resource-heavy synthetic, scaling, NBA preparation, budget, and DTW
  studies were not rerun during packaging. Their aggregate measurements are
  explicitly identified as recorded results.
* Native `hnswlib==0.8.0` compilation could not be completed on this machine
  because a C++ compiler was unavailable. Install a C++11 compiler and Python
  headers before installing the optional HNSW requirements. Numerical tests use
  the explicitly labelled test backend; it is never published as HNSW timing.
* Broadcast downloads and broadcast redistribution were not performed.

Test updates for this standalone checkout account for relocated data
directories, the optional explicit benchmark-host guard, and the already removed
exact-scan plot series. The numerical algorithms and benchmark results were
not changed to make tests pass.
