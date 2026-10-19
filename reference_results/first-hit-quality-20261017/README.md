# First-hit approximation-quality runs (October 17-18, 2026)

Compact records behind the first-hit (FH) evaluation: per-query logs, frozen selections, tuning summaries, and completion markers. Ground-truth arrays (`*.npz`), raw tuning-query logs, and the reference stages are omitted because they are large and are regenerated deterministically from the configurations.

| Directory | Contents |
|---|---|
| `first-hit-quality-followup-20261017` | Frontier of all block partitions, lambda selections, matched quality, 2M scale-up, sensitivity, independent datasets B and C, DTW, and the hard-budget sweep k=4..12 (index seed 1701). Replications also include seeds 1702 and 1703. |
| `joint-first-hit-baselines-seed1-20261017` | Probe-all and Store-all first-hit baselines for the same studies (index seed 1701). |
| `submission-fh-quality-20261017` | NBA and DTW studies with three index seeds (NBA seeds 2301-2303, DTW seeds 1701-1703). |
| `nba-fh-accuracy-20261018`, `nba-fh-storage-20261018` | NBA first-hit follow-up (fine-to-coarse cascade) with seeds 2301-2303. |
| `database-near-20261017`, `collection-reference-20261017` | Exact full-database near-neighbor counts and reference distances. |
| `paper-distance-quality-records.json` | Per-seed summaries consumed by `fresh/render_fh_blocking_paper.py`. |

Rebuild the tables and figures from the aggregate records without rerunning any experiment:

```bash
python -m fresh.render_fh_blocking_paper \
    --records reference_results/first-hit-quality-20261017/paper-distance-quality-records.json \
    --paper <paper-checkout> --preview <output-directory> \
    --reference-results reference_results/first-hit-quality-20261017/collection-reference-20261017 \
    --near-results reference_results/first-hit-quality-20261017/database-near-20261017 \
    --frontier-results reference_results/first-hit-quality-20261017/first-hit-quality-followup-20261017
```

The configurations are in `configs/` (`first-hit-quality-followup-20261017.json`, `joint-first-hit-baselines-seed1-20261017.json`, `submission-fh-quality-20261017.json`, `nba-fh-accuracy-20261018.json`, `nba-fh-storage-20261018.json`). Absolute workstation paths inside the JSON metadata were replaced by relative ones, so the SHA-256 values in `completion.json` describe the original, unrewritten metadata files; the per-query CSV logs are unmodified.

The NBA table's first-hit row comes from `nba-fh-accuracy-20261018`; the other tables regenerate identically from the records above. The hard-budget sweep's raw query logs (about 370 MB) are not included, so the renderer skips that step; its per-job `main-results.json` files and selections are included.

Index seeds: the hard-budget sweep, partition frontier, lambda selection, matched quality, scale-up, and sensitivity studies were run with one index seed; additional seeds are pending.
