import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from fresh.nba import clock_value,extract_clip,optimal_mapping,game_clips,render
from fresh.nba_gallery import quality_metrics


def fixture():
    players=[dict(playerid=i,firstname=f"Player{i}",lastname="Example",jersey=str(i)) for i in range(1,6)]
    event=dict(eventId="25",home=dict(teamid=2,name="Home",abbreviation="H",players=[]),
               visitor=dict(teamid=1,name="Visitor",abbreviation="V",players=players),moments=[])
    for frame in range(101):
        objects=[[-1,-1,20+frame*.01,25,1]]
        for i in range(1,6):
            objects.append([1,i,10+i+frame*.02,5+i*7,0])
        event["moments"].append([1,1_000_000+40*frame,100-frame*.04,10,None,objects])
    shot=dict(PERIOD="1",PCTIMESTRING="1:36",PLAYER1_TEAM_ID="1.0",PLAYER1_ID="1",
              VISITORDESCRIPTION="Example shot",EVENTMSGTYPE="1")
    cfg=dict(clip_duration_seconds=4,clock_alignment_tolerance_seconds=.12,
             maximum_gap_seconds=.12,clock_elapsed_tolerance_seconds=.2,m=32,dataset_seed=12)
    return event,shot,cfg


class NBATests(unittest.TestCase):
    def test_animation_uses_vertices_not_number_of_players(self):
        from PIL import Image
        Q=np.empty((5,32,2))
        for i in range(5):
            Q[i,:,0]=5+5*i+np.linspace(0,6,32)
            Q[i,:,1]=7+7*i+np.linspace(0,2,32)
        ball=np.stack((np.linspace(5,15,32),np.full(32,25.)),axis=1)
        info=dict(matchup="Example teams",date="01.01.2016",quarter=1,clock_start=100,clock_end=96,
                  duration_seconds=4,player_ids=[1,2,3,4,5],player_names=["A","B","C","D","E"],
                  jerseys=["1","2","3","4","5"])
        X=np.array([Q+.2])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            render(root,(Q,ball,info),[dict(database_index=0,rank=1,distance_feet=.3,approximation_ratio=1)],
                   X,np.array([ball]),[info])
            with Image.open(root/"nba-query-and-results.gif") as movie:
                self.assertEqual(movie.n_frames,32)
            self.assertTrue((root/"nba-start-middle-end.png").exists())
    def test_full_cohort_quality_does_not_hide_empty_answers(self):
        rows=[dict(found_exact_nearest=True,results=[dict(approximation_ratio=1.)]),
              dict(found_exact_nearest=False,results=[])]
        stats=quality_metrics(rows)
        self.assertEqual(stats["coverage"],.5)
        self.assertEqual(stats["recall_at_1"],.5)
        self.assertEqual(stats["c_success"]["1.1"],.5)
        self.assertEqual(stats["mean_ratio"],1)

    def test_clock_and_player_identity_order_are_not_event_array_indices(self):
        self.assertEqual(clock_value("8:05"),485)
        event,shot,cfg=fixture()
        result,reason=extract_clip(event,shot,cfg,"0000000001","example.7z",7)
        self.assertIsNone(reason)
        curves,ball,metadata=result
        self.assertEqual(curves.shape,(5,32,2))
        self.assertEqual(metadata["event_id"],25)
        self.assertEqual(metadata["event_array_index"],7)
        self.assertEqual(metadata["duration_seconds"],4)
        json.dumps(metadata)
        self.assertEqual(sorted(metadata["player_ids"]),[1,2,3,4,5])
        for curve,p in zip(curves,metadata["player_ids"]):
            self.assertAlmostEqual(curve[0,0],10+p)
            self.assertAlmostEqual(curve[-1,0],12+p)

    def test_rotation_is_shared_by_every_player_and_the_ball(self):
        event,shot,cfg=fixture()
        original,reason=extract_clip(event,shot,cfg,"0000000001","example.7z",7)
        for moment in event["moments"]:
            for obj in moment[5]:
                obj[2],obj[3]=94-obj[2],50-obj[3]
        rotated,reason=extract_clip(event,shot,cfg,"0000000001","example.7z",7)
        self.assertIsNone(reason)
        self.assertTrue(rotated[2]["orientation_rotated"])
        self.assertTrue(np.allclose(original[0],rotated[0]))
        self.assertTrue(np.allclose(original[1],rotated[1]))

    def test_missing_tracking_and_lineup_changes_are_rejected(self):
        event,shot,cfg=fixture()
        event["moments"]=event["moments"][:30]+event["moments"][40:]
        result,reason=extract_clip(event,shot,cfg,"0000000001","example.7z",7)
        self.assertIsNone(result)
        self.assertEqual(reason,"tracking gap")
        event,shot,cfg=fixture()
        for moment in event["moments"][50:]:
            moment[5][1][1]=10
        result,reason=extract_clip(event,shot,cfg,"0000000001","example.7z",7)
        self.assertIsNone(result)
        self.assertEqual(reason,"lineup/identity change")

    def test_matching_does_not_use_player_identity(self):
        Q=np.zeros((5,4,2))
        for i in range(5):Q[i,:,0]=i*10
        order=[4,2,0,3,1]
        P=Q[order].copy()
        mapping,pairwise,distance=optimal_mapping(Q,P)
        self.assertEqual(distance,0)
        self.assertTrue(np.array_equal(P[mapping],Q))
        self.assertTrue(np.array_equal(pairwise,np.zeros(5)))

    def test_shot_clip_uses_global_timeline_not_tracking_event_id(self):
        event,shot,cfg=fixture()
        event["eventId"]="999"
        cfg["source_raw_root"]="https://example.invalid"
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"raw").mkdir()
            raw=root/"raw"/"0000000001.json"
            raw.write_text(json.dumps(dict(gameid="0000000001",events=[event,event])))
            events=root/"events.csv"
            import csv
            row=dict(shot,EVENTNUM="25")
            with events.open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(row))
                writer.writeheader();writer.writerow(row)
            (root/"raw"/"archive.7z").write_bytes(b"fixture")
            with patch("fresh.nba.prepare_archive",return_value=(raw,events,"0000000001")):
                clips,audit=game_clips(cfg,"archive.7z",root)
            self.assertEqual(len(clips),1)
            self.assertEqual(clips[0][2]["event_id"],25)
            self.assertEqual(clips[0][2]["tracking_event_ids"],[999])
            self.assertIsNone(clips[0][2]["event_array_index"])
            self.assertEqual(audit["duplicate_frames"],101)
            self.assertEqual(audit["unique_frames"],101)

    def test_corrected_shot_time_requires_shooter_proximity_and_ball_flight(self):
        event,shot,cfg=fixture()
        cfg.update(use_fixed_shot_times=True,maximum_shooter_ball_distance_feet=6,
                   maximum_nominal_shot_distance_error_feet=5,
                   minimum_future_ball_height_feet=9.5,maximum_future_ball_basket_distance_feet=7,
                   ball_flight_validation_seconds=3)
        shot["_nominal_distance_feet"]=float(np.linalg.norm(np.array([13,12])-np.array([5.25,25])))
        event["moments"][-1][5][0][2:5]=[13,12,7]
        for frame in range(101,176):
            t=(frame-100)/75
            objects=[[-1,-1,13+(5.25-13)*t,12+13*t,7+6*np.sin(np.pi*t)]]
            objects.extend([1,i,12+i,5+i*7,0] for i in range(1,6))
            event["moments"].append([1,1_000_000+40*frame,100-frame*.04,10,None,objects])
        result,reason=extract_clip(event,shot,cfg,"0000000001","example.7z",7)
        self.assertIsNone(reason)
        self.assertEqual(result[2]["shot_anchor_source"],"validated corrected-shots table")
        shot["_nominal_distance_feet"]=40
        result,reason=extract_clip(event,shot,cfg,"0000000001","example.7z",7)
        self.assertIsNone(result)
        self.assertEqual(reason,"corrected shot anchor disagrees with nominal shot distance")
        shot["_nominal_distance_feet"]=float(np.linalg.norm(np.array([13,12])-np.array([5.25,25])))
        event["moments"][100][5][0][2:4]=[40,40]
        result,reason=extract_clip(event,shot,cfg,"0000000001","example.7z",7)
        self.assertIsNone(result)
        self.assertEqual(reason,"corrected shot anchor fails shooter-ball proximity")


if __name__=="__main__":
    unittest.main()
