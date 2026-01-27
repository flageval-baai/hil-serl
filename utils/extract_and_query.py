#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "pyarrow is required to read parquet files. Install it with:\n"
        "  pip install pyarrow\n"
    ) from exc


def load_info(info_path: Path) -> dict:
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_episode_paths(root: Path, info: dict, episode_index: int, video_key: str) -> tuple[Path, Path]:
    chunk_size = int(info.get("chunks_size", 10000))
    episode_chunk = episode_index // chunk_size
    data_path = info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    video_path = info["video_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    return root / data_path, root / video_path


def extract_frames(video_path: Path, output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory not empty: {output_dir}")

    pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-start_number",
        "0",
        pattern,
    ]
    subprocess.run(cmd, check=True)


def pick_gripper_index(names: list[str], gripper_name: str | None) -> int:
    if gripper_name:
        if gripper_name not in names:
            raise ValueError(f"gripper name not found in state names: {gripper_name}")
        return names.index(gripper_name)
    for i, name in enumerate(names):
        if "gripper" in name:
            return i
    raise ValueError("No gripper dimension found. Pass --gripper-name explicitly.")


def format_state(names: list[str], values: np.ndarray) -> str:
    lines = []
    for name, val in zip(names, values):
        lines.append(f"{name}: {float(val):.6f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract video frames and query state/gripper info by frame index."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root with meta/, data/, videos/.")
    parser.add_argument("--episode-index", type=int, required=True, help="Episode index to inspect.")
    parser.add_argument(
        "--video-key",
        default="observation.images.image_front",
        help="Video key folder under videos/chunk-XXX/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save extracted frames (default: <dataset-root>/frames/<episode>/<video-key>).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output-dir if it exists.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip frame extraction.")
    parser.add_argument("--gripper-name", default=None, help="Exact gripper name in observation.state.")
    parser.add_argument("--gripper-threshold", type=float, default=0.7, help="Threshold for gripper value.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")

    info = load_info(info_path)
    #episode_path, video_path = resolve_episode_paths(
        #ataset_root, info, args.episode_index, args.video_key
    #)
    #if not episode_path.exists():
        #raise FileNotFoundError(f"Missing episode parquet: {episode_path}")
    #if not video_path.exists():
        #raise FileNotFoundError(f"Missing video: {video_path}")

    #if args.output_dir:
        #output_dir = Path(args.output_dir)
    #else:
        #safe_key = args.video_key.replace("/", "_")
        #output_dir = dataset_root / "frames" / f"episode_{args.episode_index:06d}" / safe_key

    #if not args.skip_extract:
        #extract_frames(video_path, output_dir, args.overwrite)
        #print(f"Frames saved to: {output_dir}")

    table = pq.read_table("/home/baai/DoRobot/dataset/20260122/user/P3_Put all blocks that have the same color as the cylinder into the basket_1024/P3_Put all blocks that have the same color as the cylinder into the basket_1024_99505/data/chunk-000/episode_000000.parquet")
    df = table.to_pandas()
    if "observation.state" not in df.columns:
        raise ValueError("Column 'observation.state' not found in parquet.")

    state_names = info["features"]["observation.state"]["names"]
    gripper_idx = pick_gripper_index(state_names, args.gripper_name)

    if "frame_index" in df.columns:
        frame_indices = df["frame_index"].tolist()
        frame_to_row = {int(f): i for i, f in enumerate(frame_indices)}
        use_frame_index = True
    else:
        frame_to_row = None
        use_frame_index = False

    while True:
        raw = input("Enter frame index (or 'q' to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        if not raw:
            continue
        try:
            frame_idx = int(raw)
        except ValueError:
            print("Please enter an integer frame index.")
            continue

        if use_frame_index:
            if frame_idx not in frame_to_row:
                print(f"Frame {frame_idx} not found in frame_index column.")
                continue
            row_idx = frame_to_row[frame_idx]
        else:
            if frame_idx < 0 or frame_idx >= len(df):
                print(f"Frame {frame_idx} out of range: 0..{len(df) - 1}")
                continue
            row_idx = frame_idx

        state = np.asarray(df["observation.state"].iloc[row_idx], dtype=np.float32)
        print(f"Frame {frame_idx} -> row {row_idx}")
        print(format_state(state_names, state))

        gripper_values = np.asarray(
            [np.asarray(s, dtype=np.float32)[gripper_idx] for s in df["observation.state"]],
            dtype=np.float32,
        )
        next_idx = None
        for i in range(row_idx, len(gripper_values)):
            if gripper_values[i] < args.gripper_threshold:
                next_idx = i
                break

        if next_idx is None:
            print(
                f"No gripper value < {args.gripper_threshold} found after row {row_idx}."
            )
        else:
            if use_frame_index:
                next_frame = int(frame_indices[next_idx])
            else:
                next_frame = next_idx
            next_state = np.asarray(
                df["observation.state"].iloc[next_idx], dtype=np.float32
            )
            print(
                f"Next gripper < {args.gripper_threshold} at frame {next_frame} (row {next_idx}): "
                f"{float(next_state[gripper_idx]):.6f}"
            )


if __name__ == "__main__":
    main()
