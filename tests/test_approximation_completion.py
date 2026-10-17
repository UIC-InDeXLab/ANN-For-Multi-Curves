import json
from pathlib import Path
import tempfile
import unittest

from fresh.e1 import worker,write_json


class ApproximationCompletionTests(unittest.TestCase):
    def test_quality_only_filter_does_not_run_expensive_verified_baselines(self):
        path=Path(__file__).resolve().parents[1]/"configs"/"e1-pilot-20261012.json"
        config=json.loads(path.read_text())
        config.update(background_n=8,test_queries=4,index_seeds=[17],m=8,warmup_queries=1,
                      methods=["cost-model"],query_modes=["first"])
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write_json(root/"manifest.json",dict(commit="unit",config_sha256="unit"))
            write_json(root/"calibration.json",dict(entries=[1000],seconds_per_probe=[1e-8]))
            folder=root/"case"
            folder.mkdir()
            worker(config,folder,"random",2,2)
            rows=json.loads((folder/"configurations.json").read_text())
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]["method"],"cost-model")
            self.assertEqual(rows[0]["mode"],"first")
            self.assertEqual(rows[0]["status"],"complete")


if __name__=="__main__":
    unittest.main()
