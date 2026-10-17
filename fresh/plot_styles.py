"""Consistent method encodings for color and monochrome publication plots."""
COLORS={"blocking":"#007f5f","probe_all":"#277da1","store_all":"#D55E00",
        "per_curve-intersection":"#b5179e","brute-force":"#444444",
        "hnsw_curve":"#c9a000","hnsw_sketch":"#8a4b2a"}
MARKERS={"blocking":"o","probe_all":"^","store_all":"s",
         "per_curve-intersection":"D","brute-force":"x","hnsw_curve":"v","hnsw_sketch":"P"}
LINESTYLES={"blocking":"-","probe_all":"--","store_all":"-.",
            "per_curve-intersection":":","brute-force":(0,(4,2,1,2,1,2)),
            "hnsw_curve":(0,(5,1.5)),"hnsw_sketch":(0,(1,1.2))}
LINE_WIDTH=.7
MARKER_SIZE=3


def method_style(method):
    return dict(color=COLORS[method],marker=MARKERS[method],
                linestyle=LINESTYLES[method],linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,markeredgewidth=.55,markeredgecolor="black")
