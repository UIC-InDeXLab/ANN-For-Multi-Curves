"""Paper-sized version of the actual selected NBA result, without tiny roster footers."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fresh.nba import court,clock_text,game_clips,optimal_mapping


def main(root):
    report=json.loads((root/"results.json").read_text())
    source=Path(report["source_results"])
    config=json.loads((source/"config.json").read_text())
    metadata=json.loads((source/"clip-metadata.json").read_text())
    with np.load(source/"clips.npz") as saved:
        X=saved["database"];balls=saved["balls"]
    queries,_=game_clips(config,config["query_archive"],Path(config["data_directory"]).resolve())
    selected=next(c for c in queries if c[2]["event_id"]==report["selected_query"]["event_id"])
    Q,ball,info=selected
    panels=[(Q,ball,info,"Query",None)]
    for row in report["displayed_results"]:
        p=row["database_index"];order,_,_=optimal_mapping(Q,X[p])
        panels.append((X[p,order],balls[p],metadata[p],f"Rank {row['rank']}",row))
    points=np.concatenate([np.concatenate((P.reshape(-1,2),b)) for P,b,meta,title,match in panels])
    limits=([min(-2,float(points[:,0].min())-2),max(49,float(points[:,0].max())+2)],
            [min(-2,float(points[:,1].min())-2),max(52,float(points[:,1].max())+2)])
    colors=["#007f5f","#277da1","#b5179e","#e76f51","#7a6c15"]
    fig,axes=plt.subplots(1,len(panels),figsize=(12,5.3))
    for ax,(P,b,meta,title,match) in zip(np.atleast_1d(axes),panels):
        court(ax,limits)
        for i,C in enumerate(P):
            ax.plot(C[:,0],C[:,1],color=colors[i],lw=2)
            ax.scatter(*C[0],color=colors[i],s=45,zorder=3)
            ax.scatter(*C[-1],color=colors[i],s=55,marker="X",zorder=3)
            j=len(C)//2
            ax.annotate("",xy=C[j],xytext=C[j-1],arrowprops=dict(arrowstyle="->",color=colors[i],lw=2))
        ax.plot(b[:,0],b[:,1],"--",color="#ff9f1c",lw=1.5)
        score="" if match is None else f"\nD = {match['distance_feet']:.2f} ft"
        ax.set_title(f"{title}: {meta['matchup']}\nQ{meta['quarter']}, "
                     f"{clock_text(meta['clock_start'])}–{clock_text(meta['clock_end'])}"+score,fontsize=12)
        ax.tick_params(labelsize=11)
        ax.set_xlabel("Court x (feet)",fontsize=12)
        ax.set_ylabel("Court y (feet)" if ax is axes[0] else "",fontsize=12)
    fig.tight_layout()
    fig.savefig(root/"nba-paper-figure.pdf",bbox_inches="tight")
    fig.savefig(root/"nba-paper-figure.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("results",type=Path)
    main(parser.parse_args().results.resolve())
