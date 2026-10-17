# Basketball animation and optional local video

The [README animation](../assets/basketball.gif) is a generated court diagram.
It contains no broadcast pixels or audio. The query is a Dallas-at-Clippers
possession and the returned play is a Clippers-at-Lakers possession. Five colors
link the offensive trajectories according to the recovered correspondence.
The ball is contextual and is not part of the distance.

The released GIF is an existing illustration from the five-player benchmark.
It is selected to show movement and correspondence, not a representative sample
of retrieval quality. Test accuracy is measured on all held-out queries.

## Regenerate a trajectory animation

First prepare the full NBA corpus and complete the benchmark as described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Then:

```bash
python -m reproducibility.animate \
  --data data/nba-full-20261012 \
  --results results/nba-full-20261012 \
  --seed 2301 --query 327 --output outputs/basketball
```

The renderer reads an actual completed blocking answer, recovers its
curve-to-curve matching, and writes a GIF, a court PDF/PNG, and start/middle/end
panels. It refuses empty or timed-out results. The recorded example is at
distance 7.89 feet. Query numbers apply to the pinned corpus and split; they may
change if upstream archives become unavailable.

## Add a locally authorized broadcast

Obtain any needed permission yourself and save the relevant clip locally.
Official NBA event pages use this form:

```text
https://www.nba.com/stats/events?GameEventID=<event>&GameID=<game>&Season=2015-16&flag=1
```

Exact game/event IDs are recorded in the generated
`test-metadata.json` and `database-metadata.json`. This repository does not
download or redistribute broadcasts, extract protected streams, or bypass access
controls.

Install FFmpeg, then align the beginning of a locally available clip with the
chosen four-second tracking window:

```bash
python -m reproducibility.video_overlay \
  --video /path/to/authorized-clip.mp4 \
  --curves outputs/basketball/nba-query-and-results.gif \
  --start 0 --duration 4 --fps 12 \
  --output outputs/basketball-with-video.gif
```

Use `assets/basketball.gif` for `--curves` to compose the bundled animation
instead. The tool places the video beside the court animation, uses the
specified time offset, and refuses to overwrite an existing file.
The output remains ignored by Git. Do not publish it unless the permission
covers that redistribution.

Broadcast editing and tracking clocks need not agree exactly. A side-by-side
display is not a calibrated projection of player positions onto broadcast
pixels, nor does it assert exact frame synchronization or tactical equivalence.
