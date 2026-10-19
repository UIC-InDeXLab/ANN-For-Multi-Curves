import unittest

from fresh.fh_frontier_results import point_summary


class FirstHitFrontierTests(unittest.TestCase):
    def test_low_quality_fast_point_does_not_define_practical_speedup(self):
        points = [
            dict(a=1, b=40320, index_mb=5., median_ms=1., a125=.98),
            dict(a=24, b=1680, index_mb=120., median_ms=.1, a125=.97),
            dict(a=5040, b=8, index_mb=5000., median_ms=.001, a125=.6)]
        report = point_summary(points)
        self.assertEqual(report["measured"], 3)
        self.assertEqual(report["qualified"], 2)
        self.assertEqual(report["speedup"], 10)
        self.assertEqual(report["fastest"]["a"], 24)
        self.assertEqual(report["largest_stored"]["a"], 24)
        self.assertAlmostEqual(report["spearman_b_latency"], 1)

    def test_incomplete_quality_frontier_fails_explicitly(self):
        with self.assertRaises(ValueError):
            point_summary([])
        with self.assertRaises(ValueError):
            point_summary([dict(a=1, b=40320, index_mb=5, median_ms=1, a125=.5)])


if __name__ == "__main__":
    unittest.main()
