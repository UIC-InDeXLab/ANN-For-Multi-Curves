"""Curve sets from MOTChallenge ground truth (MOT17, MOT20).

For every sequence, pedestrian tracks (class 1, considered) are cut into sliding time windows. A track that
covers a window is resampled to m positions, and every pedestrian present in a window forms a curve set with
its k-1 nearest pedestrians (by mean distance over the window), a group of objects moving together.

Coordinates:
  2d - foot point (box bottom center) in pixels, rescaled to an image height of 1080.
  3d - pinhole lifting with an assumed person height H and vertical field of view: depth Z = f H / h for box
       height h (smoothed), then X = (u - cx) Z / f and Y = (v - cy) Z / f, in meters.
"""
import configparser
from pathlib import Path

import numpy as np

PERSON_HEIGHT = 1.7   # meters
VFOV_DEG = 60.0       # assumed vertical field of view


def read_seqinfo(seq_dir):
    cp = configparser.ConfigParser()
    cp.read(Path(seq_dir) / "seqinfo.ini")
    s = cp["Sequence"]
    return dict(name=s["name"], fps=float(s["frameRate"]), length=int(s["seqLength"]),
                width=int(s["imWidth"]), height=int(s["imHeight"]))


def read_tracks(seq_dir):
    """{track id: array (frames, 4) of [frame, u_foot, v_foot, box_height]} for considered pedestrians."""
    gt = np.loadtxt(Path(seq_dir) / "gt" / "gt.txt", delimiter=",")
    gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    tracks = {}
    for tid in np.unique(gt[:, 1]):
        rows = gt[gt[:, 1] == tid]
        rows = rows[np.argsort(rows[:, 0])]
        u = rows[:, 2] + rows[:, 4] / 2
        v = rows[:, 3] + rows[:, 5]
        tracks[int(tid)] = np.stack([rows[:, 0], u, v, rows[:, 5]], axis=1)
    return tracks


def _smooth(x, w):
    if w <= 1 or x.size < w:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")[: x.size]


def lift(track, info, dims):
    """Per-frame coordinates of one track: (frames,) and (frames, dims)."""
    f, u, v, h = track.T
    if dims == 2:
        s = 1080.0 / info["height"]
        return f, np.stack([u * s, v * s], axis=1)
    focal = (info["height"] / 2) / np.tan(np.radians(VFOV_DEG / 2))
    z = focal * PERSON_HEIGHT / np.maximum(_smooth(h, 9), 1.0)
    x = (u - info["width"] / 2) * z / focal
    y = (v - info["height"] / 2) * z / focal
    return f, np.stack([x, y, z], axis=1)


def curve_sets(seq_dir, k, m=32, window_s=4.0, stride_s=1.0, dims=2, max_gap=10):
    """Curve sets of size k from one sequence. Returns (X (n, k, m, dims), meta list of (seq, start, ids))."""
    info = read_seqinfo(seq_dir)
    tracks = read_tracks(seq_dir)
    coords = {}
    for tid, tr in tracks.items():
        f, P = lift(tr, info, dims)
        if f.size >= 2 and np.all(np.diff(f) <= max_gap):  # gaps are interpolated; long gaps are dropped
            coords[tid] = (f, P)
    win = window_s * info["fps"]
    step = stride_s * info["fps"]
    X, meta = [], []
    start = 1.0
    while start + win <= info["length"]:
        times = np.linspace(start, start + win, m)
        present = {}
        for tid, (f, P) in coords.items():
            if f[0] <= times[0] and f[-1] >= times[-1]:
                present[tid] = np.stack([np.interp(times, f, P[:, a]) for a in range(dims)], axis=1)
        ids = list(present)
        if len(ids) >= k:
            C = np.stack([present[i] for i in ids])            # (A, m, dims)
            centers = C.mean(axis=1)
            D = np.linalg.norm(centers[:, None] - centers[None], axis=2)
            seen = set()
            for a in range(len(ids)):
                group = tuple(sorted(np.argsort(D[a])[:k].tolist()))
                if group in seen:
                    continue
                seen.add(group)
                X.append(C[list(group)])
                meta.append((info["name"], int(start), tuple(ids[g] for g in group)))
        start += step
    if not X:
        return np.empty((0, k, m, dims)), []
    return np.stack(X), meta


def sequences(root, benchmark):
    """Training sequence directories of MOT17 (one detector variant, ground truth is shared) or MOT20."""
    root = Path(root)
    if benchmark == "MOT17":
        return sorted(p for p in (root / "MOT17" / "train").glob("MOT17-*-FRCNN"))
    return sorted((root / "MOT20" / "MOT20Labels" / "train").glob("MOT20-*"))


def load(root, benchmark, k, m=32, dims=2, **kw):
    Xs, metas = [], []
    for seq in sequences(root, benchmark):
        X, meta = curve_sets(seq, k, m=m, dims=dims, **kw)
        if len(meta):
            Xs.append(X)
            metas.extend(meta)
    return np.concatenate(Xs), metas
