import csv
import json
import shutil
from pathlib import Path
import unittest

import numpy as np

from fresh.experiment_collision import measure_collision_trials, run, shift_vector


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "tests" / "_artifacts"


def reference_frechet(A, B):
    dist = np.linalg.norm(A[:, None] - B[None, :], axis=2)
    best = np.full(dist.shape, np.inf)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i == 0 and j == 0:
                prev = 0.0
            else:
                prev = min(
                    best[i - 1, j] if i else np.inf,
                    best[i, j - 1] if j else np.inf,
                    best[i - 1, j - 1] if i and j else np.inf,
                )
            best[i, j] = max(prev, dist[i, j])
    return best[-1, -1]


def reference_hash(curve, shift, delta):
    snapped = [tuple(np.floor((vertex - shift) / delta + 0.5).astype(np.int64)) for vertex in curve]
    dedup = [snapped[0]]
    for cell in snapped[1:]:
        if cell != dedup[-1]:
            dedup.append(cell)
    return tuple(dedup)


def reference_collision(P, Q, delta, trials, seed):
    pairs, k, _, d = P.shape
    joint = np.zeros((pairs, trials), dtype=bool)
    slot_counts = np.zeros((pairs, k), dtype=np.int64)
    for pair in range(pairs):
        for trial in range(trials):
            all_hit = True
            for slot in range(k):
                shift = shift_vector(seed, pair, trial, slot, d, delta)
                hit = reference_hash(P[pair, slot], shift, delta) == reference_hash(Q[pair, slot], shift, delta)
                slot_counts[pair, slot] += int(hit)
                all_hit &= hit
            joint[pair, trial] = all_hit
    return joint, slot_counts


class CollisionExperimentTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(exist_ok=True)

    def tearDown(self):
        for name in ("collision-small-output",):
            path = ARTIFACTS / name
            if path.exists():
                shutil.rmtree(path)

    def test_tiny_measurement_matches_reference_hash_and_frequencies(self):
        P = np.array(
            [
                [
                    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                ],
                [
                    [[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]],
                    [[10.0, 1.0], [11.0, 1.0], [12.0, 1.0]],
                ],
            ]
        )
        Q = P.copy()
        Q[0, 0, :, 1] += 0.2
        Q[1, 1, :, 0] += 0.6
        delta = 2.0
        trials = 11
        seed = 12345
        fast_joint, fast_slot = measure_collision_trials(P, Q, delta, trials, seed)
        ref_joint, ref_slot = reference_collision(P, Q, delta, trials, seed)
        np.testing.assert_array_equal(fast_joint, ref_joint)
        np.testing.assert_array_equal(fast_slot, ref_slot)

    def test_reference_frechet_confirms_planted_partner_radius(self):
        curve = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        perturbed = np.array([[0.2, 0.1], [1.0, 0.3], [1.8, -0.2]])
        self.assertLessEqual(reference_frechet(curve, perturbed), 0.4)

    def test_small_fixture_writes_summary_and_csv(self):
        config = ROOT / "tests" / "fixtures" / "collision-small.json"
        output = ARTIFACTS / "collision-small-output"
        summary = run(config, output)
        self.assertEqual(len(summary["cases"]), 2)
        self.assertTrue((output / "summary.json").is_file())
        self.assertTrue((output / "summary.csv").is_file())
        with (output / "summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual({int(row["k"]) for row in rows}, {1, 2})
        loaded = json.loads((output / "summary.json").read_text())
        self.assertIn("code_commit", loaded)


if __name__ == "__main__":
    unittest.main()
