import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("ijson") is None or importlib.util.find_spec("matplotlib") is None:
    raise unittest.SkipTest("Full NBA finalization validates with the local optional dependencies")

from fresh.finalize_nba import finish


class FinalizeNBATests(unittest.TestCase):
    def test_missing_qualifying_blocking_does_not_fabricate_an_example(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/"data";output=root/"output";data.mkdir();output.mkdir()
            (data/"dataset.json").write_text('{"hash":"fixture"}')
            (output/"replicated-summary.json").write_text('{"main":[]}')
            with patch("fresh.finalize_nba.summarize",return_value=dict(main=[])):
                finish(output,data,root/"unused-policy.json")
            marker=json.loads((output/"analysis-complete.json").read_text())
            self.assertIn("not completely evaluated",marker["illustration_status"])
            self.assertFalse((output/"seed-2301").exists())


if __name__=="__main__":
    unittest.main()
