import importlib.util
import unittest

if importlib.util.find_spec("ijson") is None or importlib.util.find_spec("matplotlib") is None:
    raise unittest.SkipTest("NBA example selection validates with the optional local dependencies")

from fresh.nba_motion_examples import choose


class MotionExampleTests(unittest.TestCase):
    def test_static_closest_pair_does_not_replace_a_declared_moving_example(self):
        rows=[dict(id=0,distance=1),dict(id=1,distance=5),dict(id=2,distance=4)]
        queries=[dict(game_id="heldout",mean_path_length_feet=1),
                 dict(game_id="heldout",mean_path_length_feet=12),
                 dict(game_id="heldout",mean_path_length_feet=15)]
        database=[dict(game_id="database",mean_path_length_feet=1),
                  dict(game_id="database",mean_path_length_feet=9),
                  dict(game_id="database",mean_path_length_feet=10)]
        diagnostics=[dict(distinct_game_results=[dict(database_index=i,game_id="database")]) for i in range(3)]
        policy=dict(minimum_query_mean_path_length_feet=8,minimum_result_mean_path_length_feet=8)
        selected,count=choose(rows,diagnostics,queries,database,policy)
        self.assertEqual(selected,2)
        self.assertEqual(count,2)
        self.assertEqual(rows[0]["distance"],1)

    def test_missing_retrieval_is_not_replaced_with_a_ground_truth_neighbor(self):
        with self.assertRaises(ValueError):
            choose([dict(id=-1,distance=None)],[dict(distinct_game_results=[])],
                   [dict(game_id="heldout",mean_path_length_feet=12)],[],
                   dict(minimum_query_mean_path_length_feet=8,minimum_result_mean_path_length_feet=8))


if __name__=="__main__":
    unittest.main()
