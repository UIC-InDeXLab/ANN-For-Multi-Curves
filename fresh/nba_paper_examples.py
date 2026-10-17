"""Render genuine NBA examples using the paper's TeX fonts and approved broadcast stills."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.text import Text
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from fresh.e1 import write_json
from fresh.nba import court,optimal_mapping
from fresh.nba_full import digest


COLORS=["#007f5f","#277da1","#b5179e","#e76f51","#7a6c15"]
PUBLICATION_DPI=600


@lru_cache(maxsize=1)
def paper_font_paths():
    paths=[]
    for filename in ("LinLibertine_R.otf","LinLibertine_RB.otf"):
        path=Path(subprocess.run(["kpsewhich",filename],check=True,capture_output=True,text=True).stdout.strip())
        if not path.is_file():
            raise FileNotFoundError(f"The paper's Libertine font is unavailable: {filename}")
        paths.append(path)
    return paths


def configure_paper_font():
    regular,bold=paper_font_paths()
    fontManager.addfont(str(regular));fontManager.addfont(str(bold))
    plt.rcParams["font.family"]=FontProperties(fname=str(regular)).get_name()
    plt.rcParams.update({"pgf.texsystem":"pdflatex","pgf.rcfonts":False,
                         "pgf.preamble":r"\usepackage[T1]{fontenc}\usepackage{libertine}"})


def validate_panels(example,row,diagnostic,query_metadata,database_metadata):
    if row["timeout"] or row["id"]<0:
        raise ValueError("A paper example requires a completed, nonempty retrieval")
    ranks=example["result_ranks"]
    if ranks!=[1]:
        raise ValueError("Every paper sample must show one actual single-answer retrieval")
    results=diagnostic["distinct_game_results"][:len(ranks)]
    if [r["candidate_rank"] for r in results]!=ranks:
        raise ValueError("Do not substitute or relabel a lower-ranked candidate")
    if results[0]["database_index"]!=row["id"]:
        raise ValueError("Illustrated ANN must equal the timed single return")
    if abs(results[0]["distance_feet"]-row["distance"])>1e-8:
        raise ValueError("Illustrated ANN distance differs from the measurement")
    metadata=[query_metadata]+[database_metadata[r["database_index"]] for r in results]
    if len({m["game_id"] for m in metadata})!=len(metadata):
        raise ValueError("Every illustrated panel must come from a different game")
    panels=example["panels"]
    if len(panels)!=len(metadata):
        raise ValueError("Missing query or retrieved-game video panel")
    for panel,info in zip(panels,metadata):
        if (panel["game_id"],panel["event_id"])!=(info["game_id"],info["event_id"]):
            raise ValueError("Video panel event does not match the tracked clip")
        if f"/{info['game_id']}/{info['event_id']}/" not in panel["video_url"]:
            raise ValueError("Official video URL does not match the illustrated event")
        for boundary in ("start","end"):
            frame=panel[boundary]
            if not 0<=frame["offset_seconds"]<panel["video_duration_seconds"]:
                raise ValueError("Frame offset lies outside the verified video")
            if len(frame["centers"])!=5:
                raise ValueError("Every player needs an identified position or an explicit off-screen entry")
            hidden=frame.get("off_screen",[])
            if [i+1 for i,p in enumerate(frame["centers"]) if p is None]!=hidden:
                raise ValueError("Off-screen players must be stated explicitly")
            if any(p is not None and (len(p)!=2 or not 0<=p[0]<1200 or not 0<=p[1]<675)
                   for p in frame["centers"]):
                raise ValueError("Manual image annotations must lie within the captured frame")
        if abs(panel["end"]["offset_seconds"]-panel["start"]["offset_seconds"]-4)>1e-8:
            raise ValueError("Start and end stills must bracket a four-second video interval")
    return results,metadata


def annotated_frame(path,frame):
    with Image.open(path) as source:
        image=source.convert("RGB")
    if image.size!=(1200,675):
        raise ValueError("Annotations use the verified 1200 by 675 browser captures")
    draw=ImageDraw.Draw(image)
    font_path=str(paper_font_paths()[1])
    font=ImageFont.truetype(font_path,72)
    for i,center in enumerate(frame["centers"]):
        if center is None:
            continue
        x,y=center
        color=COLORS[i]
        draw.ellipse((x-25,y-25,x+25,y+25),outline="white",width=14)
        draw.ellipse((x-25,y-25,x+25,y+25),outline=color,width=8)
        dx,dy=frame.get("label_offsets",{}).get(str(i+1),[42,-43])
        bx,by=min(1158,max(42,x+dx)),min(633,max(42,y+dy))
        draw.line((x,y,bx,by),fill="white",width=7)
        draw.line((x,y,bx,by),fill=color,width=4)
        draw.ellipse((bx-39,by-39,bx+39,by+39),fill=color,outline="white",width=5)
        draw.text((bx,by+1),str(i+1),fill="white",font=font,anchor="mm")
    if frame.get("off_screen"):
        text="Off-screen: "+", ".join(str(i) for i in frame["off_screen"])
        small=ImageFont.truetype(font_path,32)
        draw.rectangle((8,630,335,672),fill="white")
        draw.text((15,634),text,fill="black",font=small)
    return image


def player_legend(ax,info):
    ax.axis("off")
    for i,(name,jersey) in enumerate(zip(info["player_names"],info["jerseys"])):
        first,_,surname=name.partition(" ")
        ax.text(.02+.33*(i%3),.98-.55*(i//3),
                f"{i+1}: {first} #{jersey}\n{surname}",color=COLORS[i],
                fontsize=9,va="top",transform=ax.transAxes)


def shared_player_legend(ax,query,result):
    ax.axis("off")
    for i,name in enumerate(query["player_names"]):
        label=f"{i+1}: {name} #{query['jerseys'][i]}"
        if query["player_ids"][i]!=result["player_ids"][i]:
            label+=f"\n{result['player_names'][i]} #{result['jerseys'][i]}"
        ax.text(.02+.33*(i%3),.98-.55*(i//3),label,color=COLORS[i],
                fontsize=9,va="top",transform=ax.transAxes)


def save_paper_pdf(fig,path,pad_inches=.1):
    # Native TeX fonts avoid Type 3 embedding; PNG labels keep their original text.
    labels=[(text,text.get_text()) for text in fig.findobj(match=Text)]
    try:
        for text,value in labels:
            text.set_text(value.replace("#",r"\#"))
        fig.savefig(path,bbox_inches="tight",pad_inches=pad_inches,backend="pgf",
                    dpi=PUBLICATION_DPI)
    finally:
        for text,value in labels:
            text.set_text(value)


def broadcast_court_limits(info):
    rotated=info["orientation_rotated"]
    if not isinstance(rotated,bool):
        raise ValueError("Broadcast orientation requires the original attack-direction flag")
    # Undo canonical attack rotation visually; SportVU y increases toward the camera.
    return ([49,-2],[-2,52]) if rotated else ([-2,49],[52,-2])


def presentation_panels(example,panels):
    order=example.get("presentation_order","measured")
    if order=="measured":
        return panels,example["panels"]
    if order!="reversed" or example["name"]!="motivation" or len(panels)!=2:
        raise ValueError("Only the two-panel motivation may reverse its illustrative presentation")
    query,result=panels
    return [(result[0],result[1],result[2],"Query",None),
            (query[0],query[1],query[2],"Retrieved play",result[4])],list(reversed(example["panels"]))


def render_panels(destination,panels,annotations,frames_root,compact=False):
    count=len(panels)
    if compact and count!=2:
        raise ValueError("The one-column motivation layout requires two game panels")
    fig=plt.figure(figsize=(4.7,6.2) if compact else (4.7*count,7.4))
    grid=fig.add_gridspec(3,count,height_ratios=[2.35,2.8,.55] if compact else [4.5,1.15,.6],
                         hspace=.1 if compact else .2,wspace=.08 if compact else .16)
    for col,((curves,ball,info,heading,distance),video) in enumerate(zip(panels,annotations)):
        ax=fig.add_subplot(grid[0,col])
        court(ax,broadcast_court_limits(info))
        ax.set_xlabel("")
        if compact:
            ax.axis("off")
        else:
            ax.text(.98,.02,"x (ft)",ha="right",va="bottom",fontsize=11,transform=ax.transAxes)
            ax.set_ylabel("Court y (ft)" if col==0 else "",fontsize=13)
            ax.tick_params(labelsize=11)
        for i,curve in enumerate(curves):
            ax.plot(curve[:,0],curve[:,1],color=COLORS[i],linewidth=2.4)
            ax.scatter(*curve[0],color=COLORS[i],marker="o",s=55)
            ax.scatter(*curve[-1],color=COLORS[i],marker="X",s=65)
            ax.annotate(str(i+1),curve[0],xytext=(5,4),textcoords="offset points",
                        color=COLORS[i],fontsize=9 if compact else 14,fontweight="bold")
        ax.plot(ball[:,0],ball[:,1],"--",color="#ff9f1c",linewidth=1.6)
        details="" if distance is None else f"; distance {distance:.2f} ft"
        title=f"{heading}: {info['matchup']}\n{info['date']}"
        if not compact:
            title+=f", Q{info['quarter']}{details}"
        ax.set_title(title,fontsize=9 if compact else 14)
        pictures=(grid[1,col].subgridspec(2,1,hspace=.16) if compact
                  else grid[1,col].subgridspec(1,2,wspace=.025))
        for j,boundary in enumerate(("start","end")):
            photo=fig.add_subplot(pictures[j,0] if compact else pictures[0,j])
            frame=video[boundary]
            image=annotated_frame(frames_root/frame["file"],frame)
            # Preserve source pixels in the PDF instead of resampling to axes DPI.
            photo.imshow(image,interpolation="none")
            title=boundary.title() if compact else f"{boundary.title()} ({frame['scoreboard_clock']})"
            photo.set_title(title,fontsize=8 if compact else 12,pad=3)
            photo.axis("off")
        if not compact:
            player_legend(fig.add_subplot(grid[2,col]),info)
    if compact:
        shared_player_legend(fig.add_subplot(grid[2,:]),panels[0][2],panels[1][2])
    if compact:
        fig.subplots_adjust(left=.01,right=.99,top=.94,bottom=.005)
    else:
        fig.subplots_adjust(left=.045,right=.995,top=.90,bottom=.025)
    save_paper_pdf(fig,destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".png"),dpi=PUBLICATION_DPI,bbox_inches="tight")
    plt.close(fig)


def main(args):
    configure_paper_font()
    definition=json.loads(args.annotations.read_text())
    if (definition["method"]!="blocking" or definition["index_seed"]!=2301 or
            {e["name"] for e in definition["examples"]}!={"motivation","experiment","experiment2"}):
        raise ValueError("The three paper examples use the declared frozen blocking seed")
    if definition["permission"]["status"]!="approved by author":
        raise ValueError("Broadcast still reuse must be approved before rendering")
    root=args.results.resolve();data=args.data.resolve()
    manifest=json.loads((root/"manifest.json").read_text())
    if not (root/"complete.json").exists() or manifest["dataset_sha256"]!=digest(data/"dataset.json"):
        raise ValueError("Examples require the completed, unchanged NBA corpus")
    seed=root/f"seed-{definition['index_seed']}"
    cid=json.loads((seed/"frozen-selections.json").read_text())["main"][definition["method"]]["config_id"]
    test=seed/(cid+"-test.json");diagnostics_file=seed/(cid+"-candidate-diagnostics.json")
    marker=json.loads((seed/"complete.json").read_text())["artifacts"]
    for path in (test,diagnostics_file):
        if digest(path)!=marker[path.name]:
            raise ValueError("A frozen retrieval artifact changed after completion")
    records=json.loads(test.read_text())
    if records["status"]!="complete":
        raise ValueError("An incomplete retrieval run cannot supply paper figures")
    diagnostics=json.loads(diagnostics_file.read_text())
    qmeta=json.loads((data/"test-metadata.json").read_text())
    dmeta=json.loads((data/"database-metadata.json").read_text())
    with np.load(data/"test.npz",allow_pickle=False) as arrays:
        Q,qballs=arrays["curves"],arrays["balls"]
    with np.load(data/"database.npz",allow_pickle=False) as arrays:
        X,balls=arrays["curves"],arrays["balls"]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    previous=None
    if args.only is not None:
        previous=json.loads((output/"provenance.json").read_text())
    examples={}
    games=set()
    for definition_example in definition["examples"]:
        name=definition_example["name"]
        q=definition_example["query"]
        row=records["queries"][q]
        if row["query"]!=q:
            raise ValueError("Figure query index does not match the stored evaluation")
        matches,metadata=validate_panels(definition_example,row,diagnostics[q],qmeta[q],dmeta)
        if games.intersection(info["game_id"] for info in metadata):
            raise ValueError("Motivation and experiment examples must use distinct games")
        games.update(info["game_id"] for info in metadata)
        panels=[(Q[q],qballs[q],metadata[0],"Query",None)]
        verified=[]
        for match,info in zip(matches,metadata[1:]):
            p=match["database_index"]
            order,pairwise,distance=optimal_mapping(Q[q],X[p])
            if abs(distance-match["distance_feet"])>1e-8:
                raise ValueError("Recovered matching disagrees with the actual candidate distance")
            reordered=dict(info,player_names=[info["player_names"][i] for i in order],
                           jerseys=[info["jerseys"][i] for i in order],
                           player_ids=[info["player_ids"][i] for i in order])
            heading="Retrieved ANN"
            panels.append((X[p,order],balls[p],reordered,heading,distance))
            verified.append(dict(match,query_to_result_player_ids=reordered["player_ids"],
                                 pairwise_frechet_feet=pairwise.tolist()))
        destination=output/name
        display_panels,display_videos=presentation_panels(definition_example,panels)
        if args.only is None or args.only==name:
            render_panels(destination,display_panels,display_videos,args.frames.resolve(),
                          compact=True)
        elif not destination.with_suffix(".pdf").is_file():
            raise FileNotFoundError("Unselected figure must already exist: "+str(destination))
        elif digest(destination.with_suffix(".pdf"))!=previous["examples"][name]["figure_sha256"]:
            raise ValueError("Unselected figure changed after its verified rendering")
        examples[name]=dict(query=q,query_metadata=metadata[0],results=verified,
                            retrieved_metadata=metadata[1:],optimal_feet=row["optimal_feet"],
                            file=destination.with_suffix(".pdf").name,
                            layout="two game panels with vertically stacked start/end stills and shared legend",
                            presentation_order=definition_example.get("presentation_order","measured"),
                            displayed_game_ids=[panel[2]["game_id"] for panel in display_panels],
                            reversed_retrieval_evaluated=False,
                            figure_sha256=digest(destination.with_suffix(".pdf")),
                            video_panels=definition_example["panels"])
    frames={frame["file"]:digest(args.frames/frame["file"]) for example in definition["examples"]
            for panel in example["panels"] for frame in (panel["start"],panel["end"])}
    write_json(output/"provenance.json",dict(
        examples=examples,method=definition["method"],index_seed=definition["index_seed"],
        dataset_sha256=digest(data/"dataset.json"),benchmark_manifest_sha256=digest(root/"manifest.json"),
        test_artifact=test.name,test_artifact_sha256=digest(test),
        diagnostics_artifact=diagnostics_file.name,diagnostics_artifact_sha256=digest(diagnostics_file),
        annotation_sha256=digest(args.annotations),frame_sha256=frames,
        permission=definition["permission"],alignment=definition["alignment"],
        typography=dict(font_family=plt.rcParams["font.family"][0],legend_size_points=9,
                        legend_rows=2,legend_columns=3),
        rendering=dict(screenshot_pixels=[1200,675],pdf_sampling="native-resolution",
                       preview_dpi=PUBLICATION_DPI,marker_font_pixels=72,
                       court_orientation="broadcast-aligned axes; canonical retrieval coordinates unchanged"),
        selection="Author-requested illustrations from frozen actual outputs; not used to tune or score the benchmark",
        changed_test_outcomes=False,retuned=False))
    print(f"Verified {len(examples)} query--ANN samples, {len(games)} distinct games and {len(frames)} approved broadcast stills.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--frames",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--only",choices=["motivation","experiment","experiment2"],
                        help="Update one figure while preserving the other generated PDFs")
    main(parser.parse_args())
