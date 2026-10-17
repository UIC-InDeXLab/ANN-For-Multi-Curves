import unittest

from fresh.plot_styles import COLORS, LINESTYLES, MARKERS, method_style


class PublicationStyleTests(unittest.TestCase):
    def test_methods_have_distinct_monochrome_encodings_and_small_strokes(self):
        self.assertEqual(len(set(MARKERS.values())),len(MARKERS))
        self.assertEqual(len({repr(value) for value in LINESTYLES.values()}),len(LINESTYLES))
        for method in COLORS:
            style=method_style(method)
            self.assertEqual(style["marker"],MARKERS[method])
            self.assertEqual(style["linestyle"],LINESTYLES[method])
            self.assertLessEqual(style["linewidth"],.7)
            self.assertLessEqual(style["markersize"],3)
            self.assertEqual(style["markeredgecolor"],"black")
            self.assertLessEqual(style["markeredgewidth"],.55)

    def test_removed_baseline_has_no_publication_style(self):
        with self.assertRaises(KeyError):
            method_style("per_curve-rarest")


if __name__=="__main__":
    unittest.main()
