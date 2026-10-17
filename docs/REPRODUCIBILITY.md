# Reproducing the evaluation

Run commands from the repository root in a clean Git checkout. Install the
requirements in the README first. All outputs below are ignored by Git.

## Environment and limits

The reported experiments used an AMD Ryzen Threadripper 7970X, 128 GB RAM,
single-core query timing, and up to 48 construction/ground-truth threads.
The general limits were 24 GB for indexes, 48 GB construction RAM, and 60 seconds
per query. Budget experiments replace the serving memory and query limits.

Use `--core` to pick an allowed CPU and `--threads` for construction. Keep
concurrent timed jobs off that core. The wrapper sets `PIN_CORE`,
`NUMBA_NUM_THREADS`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1` for its child
processes; it never edits tracked configurations. Warm-up and compilation are
outside query timers.

Large experiments can take hours or days. A resource-excluded configuration is
different from a measured timeout; both are recorded. The 100 ms budget study
did not complete k=12 because an untimed Probe-all warm-up exceeded 60 seconds.
Do not treat that case as a measured failure of every method.

The runtime used to validate this code release is recorded in
[VALIDATION.md](../VALIDATION.md). Fresh runs record their own versions, revision,
seed, configuration hash, dataset hash, and completion markers.

## Dataset construction

Synthetic templates are random walks. Database multi-curves choose templates
with replacement; common- and rare-path queries choose distinct templates.
Repeated-path queries sample with replacement from four templates. Each curve
receives independent vertex perturbations. One near multi-curve is planted per
query, but exact search determines the actual nearest neighbor.

The default comparison has n=100,000, k in {4,6,8,10}, m=32, 500 tuning queries,
1,000 test queries, and index seeds 1701–1703. Larger collections preserve the
generator's nested background prefix, query set, and planted positions.

The approximation-quality benchmark is separate: 10,000 initial multi-curves,
200 queries, nine added perturbed copies per query, m=32, d in {2,3},
k=2 through 10, p=0.75, r=1, and eta=0.01. The recorded configuration uses
perturbation radii `[2,4,8,16,32,64,128,256,512]` for those extra copies. Use the
configuration's values, rather than inferring the radii from abbreviated prose.

## Run order

### 1. Main synthetic, scaling, and occurrence studies

```bash
python -m reproducibility.run synthetic --core 0 --threads 8
python -m reproducibility.run scaling --core 0 --threads 8
python -m reproducibility.run selectivity --core 0 --threads 8
```

These invoke `fresh.poc`, `fresh.plot_poc`, and
`fresh.aggregate_replications`; the selectivity command also invokes
`fresh.plot_selectivity`. Raw per-query records and frozen selections remain
under `results/`.

### 2. Accuracy, collision, partition, and first-hit studies

```bash
python -m reproducibility.run collision --core 0 --threads 8
python -m reproducibility.run approximation --core 0 --threads 8
```

The original approximation evaluation used a separate continuation for k=10.
To follow that layout:

```bash
python -m reproducibility.run approximation-k10 --core 0 --threads 8
python -m fresh.report_approximation \
  --parent results/fresh-e1-20261012-full \
  --continuation results/fresh-approximation-k10-20261012 \
  --output results/fresh-approximation-final-20261012
```

The following studies need the completed main synthetic run, including its
exact ground truth and frozen parameter selections:

```bash
python -m reproducibility.run frontier --core 0 --threads 8
python -m reproducibility.run matched-recall --core 0 --threads 8
python -m reproducibility.run first-hit-a --core 0 --threads 8
python -m reproducibility.run first-hit-b --core 0 --threads 8
python -m reproducibility.run first-hit-1m --core 0 --threads 8
python -m reproducibility.run first-hit-selectivity --core 0 --threads 8
```

The first-hit selectivity study also needs the occurrence study.
`frontier` enumerates both orientations of the integer partitions of eight
slots. The runner retains labelled endpoint duplicates in its diagnostic rows;
the paper counts 42 distinct configurations after identifying the two
Store-all and two Probe-all representations.

### 3. Replication, curve length, perturbation, and budgets

```bash
python -m reproducibility.run replication-b --core 0 --threads 8
python -m reproducibility.run replication-c --core 0 --threads 8
python -m reproducibility.run sensitivity-m16 --core 0 --threads 8
python -m reproducibility.run sensitivity-m64 --core 0 --threads 8
python -m reproducibility.run sensitivity-r025 --core 0 --threads 8
python -m reproducibility.run sensitivity-r1 --core 0 --threads 8
python -m reproducibility.run budget --core 0 --threads 8
python -m reproducibility.run budget-256mb-1ms --core 0 --threads 8
python -m reproducibility.run budget-16gb-10ms --core 0 --threads 8
python -m reproducibility.run budget-16gb-100ms --core 0 --threads 8
```

Budgets use the fastest tuning-qualified configuration within the hard limits,
rather than the weighted objective. A tested k qualifies only when every seed
returns an exact nearest neighbor on time for at least 90% of the test queries.

### 4. NBA preparation and evaluation

Install `7z` (7-Zip) and `ijson`. On distributions with apt, the archive utility
is available as `p7zip-full` or `7zip`. The input collection is downloaded by
the preparation command, not bundled with this repository.

```bash
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m fresh.nba_full prepare --config configs/nba-full-20261012.json

NUMBA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m fresh.nba_full benchmark \
  --config configs/nba-full-20261012.json \
  --output results/nba-full-20261012 \
  --confirm-benchmark --pin-core 0 \
  --index-budget-bytes 24000000000 --peak-ram-cap-bytes 48000000000

python -m fresh.summarize_nba_full results/nba-full-20261012
python -m reproducibility.run nba-selectivity --core 0 --threads 8
```

Preparation is resumable and writes hashes and rejection reasons under
`data/nba-full-20261012/`. It uses the pinned upstream revision in the config.
Expect substantial downloads and temporary disk usage; processing every archive
is not a small-demo command. Availability of that third-party archive can affect
the reconstructed corpus. The recorded database has 23,805 plays, with 500 tuning
and 1,000 test queries from disjoint games. The unused remainder of the extracted
clips is not silently added to the database.

The `--confirm-benchmark` flag explicitly allows a resource-heavy timed run.
An optional `--expected-hostname` protects against running on the wrong machine.
The legacy resource-profile name in the original runner is retained for
compatibility; explicit memory overrides above are portable.

### 5. HNSW, DTW, and construction

Install `requirements-hnsw.txt` for the actual graph baseline:

```bash
python -m reproducibility.run hnsw --core 0 --threads 8
python -m reproducibility.run hnsw-ksweep --core 0 --threads 8
python -m reproducibility.run hnsw-scaling --core 0 --threads 8
python -m reproducibility.run hnsw-selectivity --core 0 --threads 8
python -m reproducibility.run dtw --core 0 --threads 8
python -m reproducibility.run build-time --core 0 --threads 8
```

Run the relevant synthetic, occurrence, scaling, and NBA references first.
The wrapper updates the DTW/HNSW NBA hash guards to the freshly prepared
corpus; the guards still validate the actual input files. It does not substitute
Fréchet ground truth for DTW ground truth. DTW is the unconstrained sum of
Euclidean vertex costs along a monotone traversal, without normalization.
Across curves, the objective is still bottleneck assignment.

Construction tests rebuild frozen selected configurations; they require both
the synthetic/scaling results and the completed NBA evaluation.

## Figure and table generation

The self-contained path for the included measurements is:

```bash
python -m reproducibility.verify_reference
python -m reproducibility.render_reference --figures
```

PDF figures use Libertine fonts and `pdflatex` through Matplotlib's PGF backend.
On a TeX Live installation, provide `kpsewhich`, `pdflatex`, the `libertine`
package, and its fonts. They are commonly supplied by `texlive-latex-extra` and
`texlive-fonts-extra` (package names depend on the distribution).

Tables can be regenerated without TeX. Eight figure PDFs and 17 table bodies
can be rebuilt from the included summaries. The recorded scaling table is copied
because its Build column additionally depends on per-seed records.
The snapshot manifest checks the numerical inputs before export.

For newly measured runs, the original `fresh.render_experiment_tables` and
`fresh.plot_experiment_figures` CLIs expect manuscript provenance files
(`generated-results.json` and `scaleup-provenance.json`) in a `--paper` directory
as well as raw study outputs. They are preserved as original publication tools,
not presented as self-contained commands in this standalone release. Individual
study summarizers operate on the new run directories.

## Recorded results versus a new reproduction

The included references preserve the measured values. They are not a portable
cache of indexes or a guarantee that a newly built index will have identical
latency. Array hashes, source-data pins, fixed seeds, warm-up, and separate
tuning/test queries make discrepancies diagnosable.

Changes for this release are limited to relative input/output paths, generic
benchmark authorization, CPU selection, packaging, and the reproduction/demo
entry points. The distance, hashing, order-family, index-construction,
candidate-verification, and parameter-selection kernels are preserved.
