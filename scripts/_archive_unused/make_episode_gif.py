import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def find_episode_file(data_dir: Path, episode_name: str):
    candidates = list(data_dir.rglob(episode_name))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Episode file '{episode_name}' not found under {data_dir}"
        )
    if len(candidates) > 1:
        print("Warning: multiple matches found, using the first one:")
        for c in candidates:
            print("  ", c)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing raw episodes, e.g. data/raw")
    parser.add_argument("--episode-file", type=str, required=True,
                        help="Episode filename, e.g. 1781972395073_00000.npz")
    parser.add_argument("--out-gif", type=str, required=True,
                        help="Output gif path")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip", type=int, default=1,
                        help="Keep one frame every --skip frames")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    episode_path = find_episode_file(data_dir, args.episode_file)

    data = np.load(episode_path)
    if "obs" not in data:
        raise KeyError(f"'obs' not found in {episode_path}. Keys: {list(data.keys())}")

    obs = data["obs"]   # expected shape [T, H, W, C]
    if obs.ndim != 4:
        raise ValueError(f"Expected obs with shape [T,H,W,C], got {obs.shape}")

    frames = obs

    if args.skip > 1:
        frames = frames[::args.skip]

    if args.max_frames is not None:
        frames = frames[:args.max_frames]

    frames = np.asarray(frames)

    # Convert to uint8 if needed
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frames = frames.clip(0, 255).astype(np.uint8)

    out_gif = Path(args.out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(out_gif, list(frames), fps=args.fps)

    print(f"Episode file: {episode_path}")
    print(f"Num frames written: {len(frames)}")
    print(f"Saved gif to: {out_gif}")


if __name__ == "__main__":
    main()