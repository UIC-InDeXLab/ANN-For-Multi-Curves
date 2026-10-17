"""SportVU five-player retrieval with original court diagrams and exact provenance."""
import argparse
import csv
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

import ijson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc,Circle,Rectangle
from matplotlib.animation import FuncAnimation,PillowWriter
import numpy as np

from curveset import core
from curveset.tune import tables_for_recall
from fresh.e1 import write_json
from fresh.poc import Joint
from fresh.regimes import family_first_hits,timed_queries,warm
from fresh.tuple_quality import exact_distance


def download(url,path):
    if path.exists():
        if path.stat().st_size==0:
            raise ValueError(f"Empty cached download: {path}")
        return
    temporary=path.with_suffix(path.suffix+".part")
    with urllib.request.urlopen(url,timeout=60) as response,temporary.open("wb") as handle:
        while chunk:=response.read(1<<20):
            handle.write(chunk)
    temporary.replace(path)


def prepare_archive(config,name,root):
    if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}\.[A-Z]{3}\.at\.[A-Z]{3}\.7z",name):
        raise ValueError(f"Unexpected archive name: {name}")
    archive=root/"raw"/name
    download(config["source_raw_root"]+"/data/"+name,archive)
    listing=subprocess.check_output(["7z","l","-slt",str(archive)],text=True)
    names=re.findall(r"^Path = (.+)$",listing,flags=re.M)[1:]
    if len(names)!=1 or not re.fullmatch(r"\d{10}\.json",names[0]):
        raise ValueError(f"Unsafe or unexpected archive members: {names}")
    sizes=[int(x) for x in re.findall(r"^Size = (\d+)$",listing,flags=re.M)]
    if not sizes or max(sizes)>250_000_000:
        raise ValueError("Tracking JSON exceeds the small-pilot extraction limit")
    game_id=Path(names[0]).stem
    raw=root/"raw"/names[0]
    if not raw.exists():
        subprocess.run(["7z","x","-y",f"-o{root/'raw'}",str(archive)],check=True,capture_output=True)
    events=root/"events"/f"{game_id}.csv"
    download(config["source_raw_root"]+f"/data/events/{game_id}.csv",events)
    return raw,events,game_id


def clock_value(text):
    pieces=text.split(":")
    if len(pieces)!=2:
        raise ValueError(f"Invalid play clock: {text}")
    return int(pieces[0])*60+float(pieces[1])


def shot_rows(path):
    with path.open() as handle:
        return {str(int(float(row["EVENTNUM"]))):row for row in csv.DictReader(handle)
                if row["EVENTMSGTYPE"] in ("1","2") and row.get("PLAYER1_TEAM_ID")}


@lru_cache(maxsize=1)
def corrected_shots(path):
    result={}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            clock=float(row["SHOT_TIME"]) if row["SHOT_TIME"] else float("nan")
            quarter=float(row["QUARTER"]) if row["QUARTER"] else float("nan")
            if np.isfinite(clock) and np.isfinite(quarter):
                key=(str(int(float(row["GAME_ID"]))).zfill(10),str(int(float(row["GAME_EVENT_ID"]))))
                distance=float(row["SHOT_DISTANCE"])
                result[key]=dict(clock=clock,quarter=int(quarter),action=row["ACTION_TYPE"],
                                 nominal_distance_feet=distance)
    return result


def extract_clip(event,shot,config,game_id,archive,index_in_file):
    period=int(float(shot["PERIOD"]))
    official_anchor=clock_value(shot["PCTIMESTRING"])
    anchor=shot.get("_tracking_shot_clock",official_anchor)
    team_id=int(float(shot["PLAYER1_TEAM_ID"]))
    shooter=int(float(shot["PLAYER1_ID"]))
    frames={}
    for moment in event["moments"]:
        if len(moment)<6 or int(moment[0])!=period or moment[2] is None:
            continue
        if abs(float(moment[2])-anchor)>config["clip_duration_seconds"]+2:
            continue
        timestamp=float(moment[1])/1000
        players={int(player[1]):np.asarray(player[2:4],float) for player in moment[5]
                 if int(player[0])==team_id and int(player[1])>0}
        ball=[row for row in moment[5] if int(row[0])==-1]
        if len(players)!=5 or len(ball)!=1:
            continue
        frames[timestamp]=(float(moment[2]),players,np.asarray(ball[0][2:4],float),float(ball[0][4]))
    if not frames:
        return None,"no complete five-player frames"
    anchor_time=min(frames,key=lambda t:(abs(frames[t][0]-anchor),t))
    if abs(frames[anchor_time][0]-anchor)>config["clock_alignment_tolerance_seconds"]:
        return None,"event clock not covered"
    start=anchor_time-config["clip_duration_seconds"]
    times=np.array(sorted(t for t in frames if start-.04<=t<=anchor_time+.0001))
    if len(times)<2 or times[0]>start+.04:
        return None,"insufficient four-second coverage"
    if np.diff(times).max()>config["maximum_gap_seconds"]:
        return None,"tracking gap"
    if abs(frames[times[0]][0]-frames[times[-1]][0]-config["clip_duration_seconds"])>config["clock_elapsed_tolerance_seconds"]:
        return None,"stopped or discontinuous game clock"
    player_ids=sorted(frames[times[0]][1])
    if any(sorted(frames[t][1])!=player_ids for t in times):
        return None,"lineup/identity change"
    if shooter not in player_ids:
        return None,"shooter absent from lineup"
    if config.get("use_fixed_shot_times"):
        shooter_xy=frames[anchor_time][1][shooter]
        if np.linalg.norm(shooter_xy-frames[anchor_time][2])>config["maximum_shooter_ball_distance_feet"]:
            return None,"corrected shot anchor fails shooter-ball proximity"
        basket=np.array([88.75 if shooter_xy[0]>47 else 5.25,25.])
        expected=shot.get("_nominal_distance_feet")
        if expected is None or not np.isfinite(expected):
            return None,"no nominal shot-distance validation"
        if abs(np.linalg.norm(shooter_xy-basket)-expected)>config["maximum_nominal_shot_distance_error_feet"]:
            return None,"corrected shot anchor disagrees with nominal shot distance"
        future=[frame for t,frame in frames.items() if anchor_time<t<=anchor_time+config["ball_flight_validation_seconds"]]
        if not future or max(frame[3] for frame in future)<config["minimum_future_ball_height_feet"]:
            return None,"corrected shot anchor lacks elevated ball flight"
        if min(np.linalg.norm(frame[2]-basket) for frame in future)>config["maximum_future_ball_basket_distance_feet"]:
            return None,"corrected shot anchor ball does not approach inferred basket"
    positions=np.stack([[frames[t][1][p] for p in player_ids] for t in times])
    ball=np.stack([frames[t][2] for t in times])
    if not np.isfinite(positions).all() or not np.isfinite(ball).all():
        return None,"nonfinite coordinates"
    if positions[:,:,0].min() < -10 or positions[:,:,0].max()>104 or positions[:,:,1].min() < -10 or positions[:,:,1].max()>60:
        return None,"coordinate range anomaly"
    shooter_x=positions[-1,player_ids.index(shooter),0]
    if abs(shooter_x-47)<8:
        return None,"ambiguous attack direction"
    toward_right=bool(shooter_x>47)
    if ((positions[-1,:,0]>47)==toward_right).sum()<4:
        return None,"not a stable attacking half-court"
    sample=np.linspace(max(start,times[0]),anchor_time,config["m"])
    curves=np.stack([np.stack([np.interp(sample,times,positions[:,p,d]) for d in range(2)],axis=1)
                     for p in range(5)])
    ball_curve=np.stack([np.interp(sample,times,ball[:,d]) for d in range(2)],axis=1)
    if toward_right:
        curves=np.array([94.,50.])-curves
        ball_curve=np.array([94.,50.])-ball_curve
    roster={int(row["playerid"]):row for side in ("home","visitor") for row in event[side]["players"]}
    seed=int(hashlib.sha256(f"{game_id}:{event['eventId']}".encode()).hexdigest()[:12],16)+config["dataset_seed"]
    order=np.random.default_rng(seed).permutation(5)
    ids=[player_ids[i] for i in order]
    names=[roster[p]["firstname"]+" "+roster[p]["lastname"] if p in roster else str(p) for p in ids]
    description=shot.get("VISITORDESCRIPTION") or shot.get("HOMEDESCRIPTION") or ""
    metadata=dict(game_id=game_id,archive=archive,event_id=int(event["eventId"]),
                  event_array_index=index_in_file,quarter=period,offense_team_id=team_id,
                  offense_team_name=next(event[side]["name"] for side in ("home","visitor") if int(event[side]["teamid"])==team_id),
                  matchup=event["visitor"]["abbreviation"]+" at "+event["home"]["abbreviation"],
                  date=".".join(archive.split(".")[:3]),player_ids=ids,player_names=names,
                  jerseys=[roster[p].get("jersey","?") if p in roster else "?" for p in ids],
                  shot_description=description,made=shot["EVENTMSGTYPE"]=="1",
                  clock_start=float(frames[times[0]][0]),clock_end=float(frames[times[-1]][0]),
                  official_event_clock=shot["PCTIMESTRING"],timestamp_start=float(sample[0]),
                  corrected_shot_clock_seconds=float(anchor),shot_anchor_shift_seconds=float(anchor-official_anchor),
                  shot_anchor_source="validated corrected-shots table" if config.get("use_fixed_shot_times") else "play-by-play clock",
                  nominal_shot_distance_feet=shot.get("_nominal_distance_feet"),
                  timestamp_end=float(sample[-1]),duration_seconds=float(sample[-1]-sample[0]),
                  orientation_rotated=toward_right,observed_frames=len(times),
                  mean_path_length_feet=float(np.linalg.norm(np.diff(curves,axis=1),axis=2).sum(axis=1).mean()),
                  nba_event_url=f"https://www.nba.com/stats/events?GameEventID={event['eventId']}&GameID={game_id}&Season=2015-16&flag=1")
    return (curves[order],ball_curve,metadata),None


def game_clips(config,name,root):
    raw,events,game_id=prepare_archive(config,name,root)
    with raw.open("rb") as handle:
        actual_id=next(ijson.items(handle,"gameid",use_float=True))
    if str(actual_id).zfill(10)!=game_id:
        raise ValueError(f"Tracking JSON game ID disagrees with archive member: {raw}")
    shots=shot_rows(events)
    if config.get("use_fixed_shot_times"):
        (root/"shots").mkdir(exist_ok=True)
        fixed_path=root/"shots"/"shots_fixed.csv"
        download(config["shot_times_url"],fixed_path)
        fixed=corrected_shots(fixed_path)
    else:
        fixed={}
    clips=[]
    excluded={}
    windows=set()
    frames={};origins={};conflicts=set();template=None
    duplicates=0
    with raw.open("rb") as handle:
        for index,event in enumerate(ijson.items(handle,"events.item",use_float=True)):
            if template is None:
                template=dict(home=event["home"],visitor=event["visitor"])
            for moment in event["moments"]:
                if len(moment)<6 or moment[1] is None or moment[2] is None:
                    continue
                key=(int(moment[0]),int(moment[1]))
                if key in conflicts:
                    continue
                signature=(float(moment[2]),tuple(sorted(tuple(row) for row in moment[5])))
                if key in frames:
                    duplicates+=1
                    old=frames[key]
                    previous=(float(old[2]),tuple(sorted(tuple(row) for row in old[5])))
                    if signature!=previous:
                        conflicts.add(key)
                        del frames[key];del origins[key]
                        continue
                    origins[key].add((str(event["eventId"]),index))
                else:
                    frames[key]=moment
                    origins[key]={(str(event["eventId"]),index)}
    if template is None:
        raise ValueError(f"No tracking events in {raw}")
    buckets={}
    for key,moment in sorted(frames.items()):
        buckets.setdefault((key[0],int(float(moment[2]))),[]).append(moment)
    for event_id,shot in shots.items():
        period=int(float(shot["PERIOD"]))
        anchor=clock_value(shot["PCTIMESTRING"])
        if config.get("use_fixed_shot_times"):
            corrected=fixed.get((game_id,event_id))
            if corrected is None:
                excluded["no corrected shot anchor"]=excluded.get("no corrected shot anchor",0)+1
                continue
            if corrected["quarter"]!=period or abs(corrected["clock"]-anchor)>config["maximum_shot_anchor_shift_seconds"]:
                excluded["inconsistent corrected shot anchor"]=excluded.get("inconsistent corrected shot anchor",0)+1
                continue
            anchor=corrected["clock"]
            shot=dict(shot,_tracking_shot_clock=anchor,_nominal_distance_feet=corrected["nominal_distance_feet"])
        moments=[]
        for second in range(int(anchor)-4,int(anchor+config["clip_duration_seconds"])+3):
            moments.extend(buckets.get((period,second),[]))
        event=dict(template,eventId=event_id,moments=sorted(moments,key=lambda m:m[1]))
        result,reason=extract_clip(event,shot,config,game_id,name,None)
        if result is None:
            excluded[reason]=excluded.get(reason,0)+1
        else:
            metadata=result[2]
            contributing=set()
            for moment in moments:
                if metadata["timestamp_start"]<=moment[1]/1000<=metadata["timestamp_end"]:
                    contributing.update(origins[(period,int(moment[1]))])
            metadata["tracking_event_ids"]=sorted({int(event) for event,index in contributing})
            metadata["tracking_event_array_indices"]=sorted({index for event,index in contributing})
            metadata["alignment_method"]="deduplicated game timeline by quarter/clock; not tracking-event-ID equality"
            key=(metadata["quarter"],metadata["offense_team_id"],
                 round(metadata["timestamp_start"],3),round(metadata["timestamp_end"],3))
            if key in windows:
                excluded["duplicate shot window"]=excluded.get("duplicate shot window",0)+1
                continue
            windows.add(key)
            clips.append(result)
    digest=hashlib.sha256((root/"raw"/name).read_bytes()).hexdigest()
    print(f"NBA {name}: {len(clips)} usable shot clips; exclusions {excluded}",flush=True)
    return clips,dict(archive=name,game_id=game_id,archive_sha256=digest,usable_clips=len(clips),
                      unique_frames=len(frames),duplicate_frames=duplicates,conflicting_timestamps=len(conflicts),
                      excluded=excluded,source=config["source_raw_root"]+"/data/"+name)


def optimal_mapping(Q,P):
    D=np.array([[core.frechet(q,p) for p in P] for q in Q])
    threshold=core.bottleneck(D)
    k=len(Q)
    row_of_col=np.empty(k,np.int64);col_of_row=np.empty(k,np.int64)
    matched=core._has_perfect_matching(D,threshold,row_of_col,col_of_row,np.empty(k,np.int64),
                                      np.empty(k,bool),np.empty(k,np.int64))
    if not matched:
        raise AssertionError("Optimal distance has no recovered matching")
    return col_of_row.copy(),D[np.arange(k),col_of_row],float(threshold)


def court(ax,limits):
    ax.add_patch(Rectangle((0,0),94,50,fill=False,color="#555555",linewidth=1))
    ax.plot([47,47],[0,50],color="#bbbbbb",linewidth=.8)
    ax.plot([0,19],[17,17],color="#777777",linewidth=.8)
    ax.plot([0,19],[33,33],color="#777777",linewidth=.8)
    ax.plot([19,19],[17,33],color="#777777",linewidth=.8)
    ax.add_patch(Arc((19,25),12,12,theta1=-90,theta2=90,color="#777777",linewidth=.8))
    ax.add_patch(Arc((5.25,25),47.5,47.5,theta1=-68,theta2=68,color="#bbbbbb",linewidth=.8))
    ax.add_patch(Circle((5.25,25),.75,fill=False,color="#777777"))
    ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.set_aspect("equal")
    ax.set_xlabel("Court x (feet)");ax.set_ylabel("Court y (feet)")
    ax.spines[["top","right"]].set_visible(False)


def clock_text(seconds):
    return f"{int(seconds)//60:02d}:{seconds%60:04.1f}"


def render(root,query,display,X,balls,metadata):
    Q,ball,info=query
    vertices=Q.shape[1]
    if vertices!=ball.shape[0] or vertices<2:
        raise ValueError("Player and ball curves need matching frame counts")
    colors=["#007f5f","#277da1","#b5179e","#e76f51","#7a6c15"]
    panels=[(Q,ball,info,"Query",None)]
    for match in display:
        p=match["database_index"]
        order,pairwise,value=optimal_mapping(Q,X[p])
        match.update(query_to_result_player_ids=[metadata[p]["player_ids"][i] for i in order],
                     query_to_result_player_names=[metadata[p]["player_names"][i] for i in order],
                     query_to_result_jerseys=[metadata[p]["jerseys"][i] for i in order],
                     pairwise_frechet_feet=pairwise.tolist(),recovered_matching_distance_feet=value)
        panels.append((X[p,order],balls[p],metadata[p],f"Retrieved rank {match['rank']}",match))
    all_points=np.concatenate([np.concatenate((c.reshape(-1,2),b)) for c,b,meta,label,match in panels])
    limits=([min(-2,float(all_points[:,0].min())-2),max(49,float(all_points[:,0].max())+2)],
            [min(-2,float(all_points[:,1].min())-2),max(52,float(all_points[:,1].max())+2)])
    fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),8))
    for ax,(curves,ball_curve,meta,label,match) in zip(np.atleast_1d(axes),panels):
        court(ax,limits)
        for i,C in enumerate(curves):
            ax.plot(C[:,0],C[:,1],color=colors[i],linewidth=2,label=f"Matched player {i+1}")
            ax.scatter(*C[0],s=55,color=colors[i],marker="o",zorder=3)
            ax.scatter(*C[-1],s=60,color=colors[i],marker="X",zorder=3)
            point=max(1,len(C)//2)
            ax.annotate("",xy=C[point],xytext=C[point-1],arrowprops=dict(arrowstyle="->",color=colors[i],lw=2))
        ax.plot(ball_curve[:,0],ball_curve[:,1],"--",color="#ff9f1c",linewidth=1.3,label="Ball (context)")
        factor=("infinity" if match is not None and match["approximation_ratio"]=="infinity"
                else f"{match['approximation_ratio']:.2f}" if match is not None else "")
        subtitle="" if match is None else f"\nD = {match['distance_feet']:.2f} ft; / optimum = {factor}"
        ax.set_title(f"{label}: {meta['matchup']}\n{meta['date']}, Q{meta['quarter']} "
                     f"{clock_text(meta['clock_start'])}–{clock_text(meta['clock_end'])}"+subtitle,fontsize=10)
        names=meta["player_names"] if match is None else match["query_to_result_player_names"]
        jerseys=meta["jerseys"] if match is None else match["query_to_result_jerseys"]
        for i,(name,jersey) in enumerate(zip(names,jerseys)):
            ax.text(.02,-.16-.055*i,f"{i+1}: {name} #{jersey}",transform=ax.transAxes,color=colors[i],fontsize=8)
    fig.suptitle("Real NBA offensive trajectories: circles = start, crosses = end; colors follow recovered correspondence",
                 fontsize=11,y=.98)
    fig.tight_layout(rect=(0,.1,1,.9))
    fig.savefig(root/"nba-query-and-results.png",dpi=180,bbox_inches="tight")
    fig.savefig(root/"nba-query-and-results.pdf",bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(3,len(panels),figsize=(5*len(panels),15),squeeze=False)
    for row,frame in enumerate((0,vertices//2,vertices-1)):
        for col,(curves,ball_curve,meta,label,match) in enumerate(panels):
            ax=axes[row,col];court(ax,limits)
            for i,C in enumerate(curves):
                ax.plot(C[:frame+1,0],C[:frame+1,1],color=colors[i],alpha=.6)
                ax.scatter(*C[frame],s=80,color=colors[i],zorder=3)
                ax.annotate(str(i+1),C[frame],xytext=(4,4),textcoords="offset points",fontsize=9)
            ax.scatter(*ball_curve[frame],s=40,color="#ff9f1c",marker="D",zorder=4)
            ax.set_title(f"{label}: t = {frame/(vertices-1)*info['duration_seconds']:.1f}s",fontsize=10)
    fig.tight_layout();fig.savefig(root/"nba-start-middle-end.png",dpi=140,bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),5))
    objects=[]
    for ax,(curves,ball_curve,meta,label,match) in zip(np.atleast_1d(axes),panels):
        court(ax,limits);ax.set_title(label+"\n"+meta["matchup"])
        for i,C in enumerate(curves):
            line,=ax.plot([],[],color=colors[i],linewidth=2)
            dot,=ax.plot([],[],"o",color=colors[i],markersize=7)
            objects.append((line,dot,C))
        line,=ax.plot([],[],"--",color="#ff9f1c")
        dot,=ax.plot([],[],"o",color="#ff9f1c",markersize=5)
        objects.append((line,dot,ball_curve))
    def update(frame):
        for line,dot,C in objects:
            line.set_data(C[:frame+1,0],C[:frame+1,1]);dot.set_data([C[frame,0]],[C[frame,1]])
        return [artist for line,dot,C in objects for artist in (line,dot)]
    fig.tight_layout()
    fps=vertices/info["duration_seconds"]
    FuncAnimation(fig,update,frames=vertices,interval=1000/fps,blit=False).save(
        root/"nba-query-and-results.gif",writer=PillowWriter(fps=fps))
    plt.close(fig)


def main(args):
    path=args.config.resolve();config=json.loads(path.read_text())
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    data_root=Path(config["data_directory"]).resolve()
    (data_root/"raw").mkdir(parents=True,exist_ok=True);(data_root/"events").mkdir(parents=True,exist_ok=True)
    write_json(root/"config.json",config)
    manifest=dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),started=time.time(),
                  python=sys.version,numpy=np.__version__,ijson=ijson.__version__,
                  scope="local qualitative pilot; no broadcast footage downloaded; ball excluded from metric")
    write_json(root/"manifest.json",manifest)
    database=[];sources=[]
    database_game_ids=set()
    for archive in config["database_archives"]:
        clips,source=game_clips(config,archive,data_root)
        if source["game_id"] in database_game_ids:
            raise ValueError("Duplicate database game; do not inflate the real-data sample")
        database_game_ids.add(source["game_id"]);database.extend(clips);sources.append(source)
    tunes,source=game_clips(config,config["tuning_archive"],data_root)
    if source["game_id"] in database_game_ids:
        raise ValueError("Tuning game leaks into the database")
    tuning_game_id=source["game_id"];sources.append(source)
    queries,source=game_clips(config,config["query_archive"],data_root)
    if source["game_id"] in database_game_ids or source["game_id"]==tuning_game_id:
        raise ValueError("Query game must be held out of database and tuning")
    sources.append(source)
    write_json(root/"data-audit.json",sources)
    if len(database)<20 or len(tunes)<10:
        raise RuntimeError("Too few valid clips for a meaningful pilot")
    X=np.stack([c[0] for c in database]);balls=np.stack([c[1] for c in database]);metadata=[c[2] for c in database]
    tune_ids=np.random.default_rng(config["dataset_seed"]).choice(len(tunes),min(len(tunes),config["tuning_query_cap"]),replace=False)
    Qt=np.stack([tunes[i][0] for i in tune_ids])
    distances=np.array([[exact_distance(P,Q) for P in X] for Q in Qt])
    nearest=distances.argmin(axis=1);optimum=distances.min(axis=1)
    NN=X[nearest];grid=[];best=None
    for delta in config["deltas_feet"]:
        shifts=np.random.default_rng(config["index_seed"]).uniform(0,delta,(config["Lmax"],5,2))
        for sizes in config["blocking_families"]:
            first=family_first_hits(Qt,NN,shifts,delta,np.asarray(sizes),False)
            L=tables_for_recall(first,config["target_recall"],config["Lmax"])
            if L is None:
                grid.append(dict(delta=delta,blocks=sizes,status="tuning target not reached within Lmax"))
                continue
            index=Joint(X,delta,L,config["index_seed"],sizes,False)
            query=lambda Q:index.query(X,Q,config)
            warm(query,Qt[:config["warmup_queries"]])
            rows,complete=timed_queries(query,Qt,optimum,config)
            recall=sum(r["exact_nn"] for r in rows)/len(rows)
            qualified=complete and recall>=config["target_recall"] and index.index_bytes<=config["index_budget_bytes"]
            median=float(np.median([r["seconds"] for r in rows]))
            cost=index.index_bytes/config["cost_memory_scale_bytes"]+median/config["cost_latency_scale_seconds"]
            choice=dict(delta=delta,blocks=sizes,L=L,tuning_recall=recall,index_bytes=index.index_bytes,
                        normalized_cost=cost,status="qualified" if qualified else "not qualified")
            grid.append(choice)
            if qualified and (best is None or cost<best["normalized_cost"]):
                best=choice
            del index
    write_json(root/"tuning.json",grid)
    if best is None:
        raise RuntimeError("No blocking configuration qualified; retain this failure rather than fabricate matches")
    write_json(root/"frozen-selection.json",best)
    index=Joint(X,best["delta"],best["L"],config["index_seed"],best["blocks"],False)
    wanted={int(i) for i in config["query_events"]}
    selected=[c for c in queries if c[2]["event_id"] in wanted]
    missing=sorted(wanted-{c[2]["event_id"] for c in selected})
    if config["primary_display_event"] not in {c[2]["event_id"] for c in selected}:
        raise RuntimeError("Predeclared primary query has no valid clip")
    outcomes=[];display=None;primary=None
    for Q,ball,info in selected:
        answer=index.query(X,Q,config)
        if answer["timeout"]:
            raise RuntimeError("Qualitative query timed out")
        candidates=index.base.cand[:answer["candidates"]].copy()
        ranked=sorted((float(exact_distance(X[p],Q)),int(p)) for p in candidates)
        exact=np.array([exact_distance(P,Q) for P in X])
        optimal=float(exact.min())
        results=[dict(rank=rank+1,database_index=p,distance_feet=value,
                      approximation_ratio=value/optimal if optimal>0 else 1 if value==0 else "infinity",
                      clip=metadata[p]) for rank,(value,p) in enumerate(ranked[:config["top_results_logged"]])]
        if ranked and abs(answer["distance"]-ranked[0][0])>1e-8:
            raise AssertionError("Candidate reranking disagrees with the blocking query")
        outcomes.append(dict(query=info,candidates=len(candidates),exact_optimal_distance_feet=optimal,
                             exact_best_database_index=int(exact.argmin()),results=results,
                             found_exact_nearest=bool(ranked and abs(ranked[0][0]-optimal)<=1e-8)))
        if info["event_id"]==config["primary_display_event"]:
            primary=(Q,ball,info)
            if results:
                display=[results[0]]
                other=None
                for rank,(value,p) in enumerate(ranked[1:],start=2):
                    if metadata[p]["game_id"]!=results[0]["clip"]["game_id"]:
                        other=dict(rank=rank,database_index=p,distance_feet=value,
                                   approximation_ratio=value/optimal if optimal>0 else 1 if value==0 else "infinity",
                                   clip=metadata[p])
                        break
                if other is not None:display.append(other)
                elif len(results)>1:display.append(results[1])
    if not display:
        raise RuntimeError("Primary query returned no candidates; no 'similar' images will be fabricated")
    render(root,primary,display,X,balls,metadata)
    report=dict(manifest=manifest,selected_index=best,database_clips=len(database),
                database_games=len({m["game_id"] for m in metadata}),database_input_games=len(config["database_archives"]),
                tuning_games=1,tuning_queries=len(Qt),query_game=config["query_archive"],
                predeclared_query_events=config["query_events"],unusable_predeclared_events=missing,
                outcomes=outcomes,displayed_results=display,
                correspondence_note="Player identities are ignored by retrieval; matching recovered by exact bottleneck assignment.",
                ball_note="Orange ball is context only, not part of the retrieval metric.",
                timing_note="Local qualitative demonstration, not a performance comparison.",
                semantic_note="Geometric similarities are observed; no basketball-expert tactical relevance labels are claimed.")
    write_json(root/"results.json",report)
    np.savez(root/"clips.npz",database=X,balls=balls,primary_query=primary[0],primary_ball=primary[1])
    write_json(root/"clip-metadata.json",metadata)
    write_json(root/"completion.json",dict(completed=time.time()))
    print(json.dumps(dict(database_clips=len(database),selected_index=best,
                         primary_query=primary[2],top_results=display),indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();main(args)
