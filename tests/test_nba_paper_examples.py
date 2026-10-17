import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
import subprocess
from unittest.mock import patch

if importlib.util.find_spec("matplotlib") is None or importlib.util.find_spec("PIL") is None:
    raise unittest.SkipTest("Publication figure tests use the local graphics environment")

from PIL import Image

from fresh.nba_paper_examples import PUBLICATION_DPI,annotated_frame,broadcast_court_limits,configure_paper_font,player_legend,presentation_panels,render_panels,save_paper_pdf,shared_player_legend,validate_panels


class PaperExampleTests(unittest.TestCase):
    def test_motivation_reversal_swaps_curves_metadata_stills_and_headings(self):
        query=(object(),object(),dict(game_id="Toronto"),"Query",None)
        result=(object(),object(),dict(game_id="Brooklyn"),"Retrieved ANN",8.58)
        videos=[dict(game_id="Toronto"),dict(game_id="Brooklyn")]
        definition=dict(name="motivation",presentation_order="reversed",panels=videos)
        panels,annotations=presentation_panels(definition,[query,result])
        self.assertEqual(panels,[(result[0],result[1],result[2],"Query",None),
                                 (query[0],query[1],query[2],"Retrieved play",8.58)])
        self.assertEqual(annotations,list(reversed(videos)))
        self.assertEqual(query[3],"Query")
        self.assertEqual(videos[0]["game_id"],"Toronto")
        definition["name"]="experiment"
        with self.assertRaisesRegex(ValueError,"Only the two-panel motivation"):
            presentation_panels(definition,[query,result])
        definition["presentation_order"]="measured"
        self.assertEqual(presentation_panels(definition,[query,result]),([query,result],videos))

    @unittest.skipUnless(shutil.which("kpsewhich"),"Paper font tests use the local TeX fonts")
    def test_compact_snapshots_stack_vertically_and_fill_each_game_column(self):
        import matplotlib.pyplot as plt
        import numpy as np
        configure_paper_font()
        info=dict(matchup="DAL at LAC",date="10.29.2015",
                  player_names=["Chris Paul","Paul Pierce","DeAndre Jordan","Blake Griffin","JJ Redick"],
                  jerseys=["3","34","6","32","4"],player_ids=[1,2,3,4,5],orientation_rotated=False)
        curves=np.array([[[10+i,20],[15+i,25]] for i in range(5)])
        frame=dict(file="frame.png",centers=[[100+150*i,300] for i in range(5)])
        panels=[(curves,curves[0],info,"Query",None),
                (curves,curves[0],dict(info,orientation_rotated=True),"Retrieved ANN",7.89)]
        videos=[dict(start=frame,end=frame)]*2

        def check_layout(fig,path):
            fig.canvas.draw()
            courts=[ax for ax in fig.axes if ax.lines]
            self.assertEqual([ax.get_title() for ax in courts],
                             ["Query: DAL at LAC\n10.29.2015","Retrieved ANN: DAL at LAC\n10.29.2015"])
            self.assertFalse(courts[0].xaxis_inverted())
            self.assertTrue(courts[0].yaxis_inverted())
            self.assertTrue(courts[1].xaxis_inverted())
            self.assertFalse(courts[1].yaxis_inverted())
            for ax in courts:
                np.testing.assert_array_equal(ax.lines[4].get_xydata(),curves[0])
            photos=[ax for ax in fig.axes if ax.images]
            self.assertEqual([ax.get_title() for ax in photos],["Start","End","Start","End"])
            for ax in photos:
                self.assertEqual(ax.images[0].get_array().shape,(675,1200,3))
                self.assertEqual(ax.images[0].get_interpolation(),"none")
            for start,end in zip(photos[::2],photos[1::2]):
                first,last=start.get_position(),end.get_position()
                self.assertAlmostEqual(first.x0,last.x0)
                self.assertAlmostEqual(first.width,last.width)
                self.assertGreater(first.y0,last.y1)
                self.assertLess((first.y0-last.y1)*fig.get_figheight(),.25)
                self.assertGreater(first.width,.44)
            self.assertLess(photos[0].get_position().x1,photos[2].get_position().x0)
            renderer=fig.canvas.get_renderer()
            boxes=[ax.get_tightbbox(renderer) for ax in fig.axes]
            self.assertFalse(any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]))
            self.assertEqual(len(fig.axes[-1].texts),5)

        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            Image.new("RGB",(1200,675),"gray").save(root/"frame.png")
            with patch("fresh.nba_paper_examples.save_paper_pdf",side_effect=check_layout) as save, \
                    patch.object(plt.Figure,"savefig") as preview:
                render_panels(root/"motivation",panels,videos,root,compact=True)
                save.assert_called_once()
                self.assertEqual(preview.call_args.kwargs["dpi"],PUBLICATION_DPI)

    @unittest.skipUnless(all(shutil.which(tool) for tool in ("kpsewhich","pdflatex","pdfimages")),
                         "Snapshot resolution tests use the paper's TeX and PDF tools")
    def test_rendered_pdf_preserves_all_four_native_resolution_snapshots(self):
        import numpy as np
        configure_paper_font()
        info=dict(matchup="DAL at LAC",date="10.29.2015",
                  player_names=["Chris Paul","Paul Pierce","DeAndre Jordan","Blake Griffin","JJ Redick"],
                  jerseys=["3","34","6","32","4"],player_ids=[1,2,3,4,5],orientation_rotated=False)
        curves=np.array([[[10+i,20],[15+i,25]] for i in range(5)])
        frame=dict(file="frame.png",centers=[[100+150*i,300] for i in range(5)])
        panels=[(curves,curves[0],info,"Query",None),
                (curves,curves[0],info,"Retrieved ANN",7.89)]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            Image.new("RGB",(1200,675),"gray").save(root/"frame.png")
            render_panels(root/"example",panels,[dict(start=frame,end=frame)]*2,root,compact=True)
            listing=subprocess.run(["pdfimages","-list",str(root/"example.pdf")],
                                   check=True,capture_output=True,text=True).stdout
            images=[line.split() for line in listing.splitlines()
                    if len(line.split())>4 and line.split()[2]=="image"]
            self.assertEqual(len(images),4)
            self.assertTrue(all(row[3:5]==["1200","675"] for row in images),listing)
            with Image.open(root/"example.png") as preview:
                self.assertGreater(preview.width,2700)
                self.assertGreater(preview.height,3500)

    def test_shooter_moves_in_the_same_screen_direction_as_footage(self):
        import matplotlib.pyplot as plt
        from fresh.nba import court
        fig,axes=plt.subplots(1,2)
        try:
            for ax,rotated,ends in zip(axes,[False,True],
                    [((4.66,9.95),(21.4,44.87)),((3.69,11.54),(14.09,47.58))]):
                court(ax,broadcast_court_limits(dict(orientation_rotated=rotated)))
                start,end=ax.transData.transform(ends)
                if rotated:
                    self.assertLess(start[1],end[1])
                    self.assertGreater(start[0],end[0])
                else:
                    self.assertGreater(start[1],end[1])
                    self.assertLess(start[0],end[0])
        finally:
            plt.close(fig)

    def test_broadcast_orientation_cannot_be_guessed_from_missing_metadata(self):
        with self.assertRaises(KeyError):
            broadcast_court_limits({})
        with self.assertRaisesRegex(ValueError,"attack-direction flag"):
            broadcast_court_limits(dict(orientation_rotated="false"))

    @unittest.skipUnless(shutil.which("kpsewhich"),"Paper font tests use the local TeX fonts")
    def test_compact_motivation_shares_one_two_by_three_legend(self):
        import matplotlib.pyplot as plt
        configure_paper_font()
        query=dict(player_names=["Chris Paul","Paul Pierce","DeAndre Jordan","Blake Griffin","JJ Redick"],
                   jerseys=["3","34","6","32","4"],player_ids=[1,2,3,4,5])
        result=dict(query,player_names=["Chris Paul","Luc Mbah a Moute","DeAndre Jordan","Blake Griffin","JJ Redick"],
                    jerseys=["3","12","6","32","4"],player_ids=[1,6,3,4,5])
        fig,ax=plt.subplots(figsize=(4.7,1.1))
        try:
            shared_player_legend(ax,query,result)
            self.assertEqual(len(ax.texts),5)
            self.assertEqual(len({t.get_position()[0] for t in ax.texts}),3)
            self.assertEqual(len({t.get_position()[1] for t in ax.texts}),2)
            self.assertEqual(ax.texts[1].get_text(),"2: Paul Pierce #34\nLuc Mbah a Moute #12")
            fig.canvas.draw()
            boxes=[t.get_window_extent(fig.canvas.get_renderer()) for t in ax.texts]
            self.assertFalse(any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]))
        finally:
            plt.close(fig)

    @unittest.skipUnless(all(shutil.which(tool) for tool in ("kpsewhich","pdflatex","pdffonts")),
                         "Native-font export uses the paper's existing TeX tools")
    def test_paper_pdf_embeds_native_fonts_and_restores_jersey_labels(self):
        import matplotlib.pyplot as plt
        configure_paper_font()
        fig,ax=plt.subplots(figsize=(4.7,1))
        try:
            ax.axis("off");label=ax.text(.02,.5,"Chris Paul #3",fontsize=9)
            with tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)/"legend.pdf"
                save_paper_pdf(fig,path)
                fonts=subprocess.run(["pdffonts",str(path)],check=True,capture_output=True,text=True).stdout
                self.assertIn("LinLibertineT",fonts)
                self.assertIn("Type 1",fonts)
                self.assertNotIn("Type 3",fonts)
                self.assertEqual(label.get_text(),"Chris Paul #3")
        finally:
            plt.close(fig)

    @unittest.skipUnless(shutil.which("kpsewhich"),"Paper font tests use the local TeX fonts")
    def test_player_names_use_two_rows_and_three_columns(self):
        import matplotlib.pyplot as plt
        configure_paper_font()
        fig,ax=plt.subplots(figsize=(4.7,1.1))
        try:
            player_legend(ax,dict(player_names=["Chris Paul","Luc Mbah a Moute",
                "DeAndre Jordan","Blake Griffin","JJ Redick"],jerseys=["3","12","6","32","4"]))
            self.assertEqual(len(ax.texts),5)
            self.assertEqual(len({t.get_position()[0] for t in ax.texts}),3)
            self.assertEqual(len({t.get_position()[1] for t in ax.texts}),2)
            self.assertTrue(all(t.get_fontsize()==9 for t in ax.texts))
            self.assertTrue(all("Libertine" in t.get_fontfamily()[0] for t in ax.texts))
            self.assertEqual(ax.texts[1].get_text(),"2: Luc #12\nMbah a Moute")
            self.assertEqual([t.get_text().split(":")[0] for t in ax.texts],["1","2","3","4","5"])
            fig.canvas.draw()
            boxes=[t.get_window_extent(fig.canvas.get_renderer()) for t in ax.texts]
            self.assertFalse(any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]))
        finally:
            plt.close(fig)

    def fixture(self):
        query=dict(game_id="query",event_id=1)
        database=[dict(game_id="first",event_id=2),dict(game_id="second",event_id=3)]
        matches=[dict(candidate_rank=1,database_index=0,distance_feet=6),
                 dict(candidate_rank=2,database_index=1,distance_feet=8)]
        frame=dict(file="frame.png",offset_seconds=0,
                   centers=[[100,100],[200,200],[300,300],[400,400],[500,500]])
        panels=[dict(info,video_url=f"https://videos.nba.com/nba/pbp/media/{info['game_id']}/{info['event_id']}/clip.mp4",
                     video_duration_seconds=9,start=copy.deepcopy(frame),
                     end=dict(copy.deepcopy(frame),offset_seconds=4))
                for info in [query]+database[:1]]
        return dict(result_ranks=[1],panels=panels),dict(timeout=False,id=0,distance=6),dict(
            distinct_game_results=matches),query,database

    def test_only_actual_single_return_is_illustrated(self):
        example,row,diagnostics,query,database=self.fixture()
        matches,metadata=validate_panels(example,row,diagnostics,query,database)
        self.assertEqual([r["candidate_rank"] for r in matches],[1])
        self.assertEqual(len({m["game_id"] for m in metadata}),2)
        diagnostics["distinct_game_results"][0]["candidate_rank"]=2
        with self.assertRaisesRegex(ValueError,"lower-ranked"):
            validate_panels(example,row,diagnostics,query,database)

    def test_ranked_list_is_not_a_single_answer_sample(self):
        example,row,diagnostics,query,database=self.fixture()
        example["result_ranks"]=[1,2]
        with self.assertRaisesRegex(ValueError,"one actual single-answer"):
            validate_panels(example,row,diagnostics,query,database)

    def test_video_game_and_timed_result_must_match(self):
        example,row,diagnostics,query,database=self.fixture()
        row["id"]=1
        with self.assertRaisesRegex(ValueError,"timed single return"):
            validate_panels(example,row,diagnostics,query,database)
        row["id"]=0
        example["panels"][0]["event_id"]=99
        with self.assertRaisesRegex(ValueError,"event does not match"):
            validate_panels(example,row,diagnostics,query,database)

    def test_off_screen_players_cannot_get_invented_positions(self):
        example,row,diagnostics,query,database=self.fixture()
        frame=example["panels"][1]["end"]
        frame["centers"][0]=None
        with self.assertRaisesRegex(ValueError,"stated explicitly"):
            validate_panels(example,row,diagnostics,query,database)
        frame["off_screen"]=[1]
        validate_panels(example,row,diagnostics,query,database)

    def test_video_window_must_have_four_seconds_and_valid_offsets(self):
        example,row,diagnostics,query,database=self.fixture()
        example["panels"][0]["end"]["offset_seconds"]=3
        with self.assertRaisesRegex(ValueError,"four-second"):
            validate_panels(example,row,diagnostics,query,database)
        example["panels"][0]["end"]["offset_seconds"]=10
        with self.assertRaisesRegex(ValueError,"outside"):
            validate_panels(example,row,diagnostics,query,database)

    @unittest.skipUnless(shutil.which("kpsewhich"),"Paper font tests use the local TeX fonts")
    def test_annotations_preserve_image_size_and_off_screen_notice(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"frame.png"
            Image.new("RGB",(1200,675),"gray").save(path)
            frame=dict(centers=[None,[200,200],[300,300],[400,400],[500,500]],
                       off_screen=[1])
            result=annotated_frame(path,frame)
            self.assertEqual(result.size,(1200,675))
            self.assertNotEqual(result.getpixel((15,650)),(128,128,128))
            Image.new("RGB",(640,360)).save(path)
            with self.assertRaisesRegex(ValueError,"1200 by 675"):
                annotated_frame(path,frame)


if __name__=="__main__":
    unittest.main()
