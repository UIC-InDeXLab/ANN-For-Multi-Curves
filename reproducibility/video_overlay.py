"""Combine a local, authorized game clip with the trajectories-only GIF."""
import argparse
from pathlib import Path
import shutil
import subprocess


def command(video, curves, output, start=0.0, duration=4.0, fps=12):
    if start < 0 or duration <= 0 or not 1 <= fps <= 60:
        raise ValueError("Start must be nonnegative, duration positive, and fps between 1 and 60")
    for path in (video, curves):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if Path(output).resolve() in (Path(video).resolve(), Path(curves).resolve()):
        raise ValueError("Output cannot replace either input")
    graph = (
        f"[0:v]setpts=PTS-STARTPTS,fps={fps},scale=-2:400,setsar=1[v];"
        f"[1:v]setpts=PTS-STARTPTS,fps={fps},scale=-2:400,setsar=1[c];"
        "[v][c]hstack=inputs=2:shortest=1[joined];"
        "[joined]split[a][b];[a]palettegen[p];[b][p]paletteuse[out]"
    )
    return ["ffmpeg", "-n", "-ss", str(start), "-i", str(video),
            "-i", str(curves), "-filter_complex_threads", "1",
            "-filter_complex", graph, "-map", "[out]", "-t", str(duration), str(output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--curves", type=Path, default=Path("assets/basketball.gif"))
    parser.add_argument("--output", type=Path, default=Path("outputs/basketball-with-video.gif"))
    parser.add_argument("--start", type=float, default=0, help="Start offset in the local video, in seconds")
    parser.add_argument("--duration", type=float, default=4)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install ffmpeg to generate a video overlay")
    cmd = command(args.video, args.curves, args.output, args.start, args.duration, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
