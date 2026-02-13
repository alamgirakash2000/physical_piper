#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required for conversion. Install with: pip install pandas pyarrow"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_POLICY_ROOT = REPO_ROOT / "external_diffusion_policy"
if str(DIFFUSION_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))


def _require_runtime_dependencies() -> None:
    missing_modules = []
    for module_name in ("zarr",):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_modules.append(module_name)

    if missing_modules:
        raise SystemExit(
            "Missing required modules for conversion: "
            f"{', '.join(missing_modules)}. "
            "Install into your runtime environment with:\n"
            "  pip install \"zarr<3\" \"numcodecs<0.16\""
        )

    import zarr  # imported after availability check

    version = getattr(zarr, "__version__", "unknown")
    major_str = str(version).split(".", maxsplit=1)[0]
    if major_str.isdigit() and int(major_str) >= 3:
        raise SystemExit(
            f"Incompatible zarr version detected: {version}. "
            "external_diffusion_policy currently requires zarr<3. "
            "Please run:\n"
            "  pip install --force-reinstall \"zarr<3\" \"numcodecs<0.16\""
        )


_require_runtime_dependencies()
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def find_latest_lerobot_dataset(base_path: Path) -> Path | None:
    if not base_path.exists():
        return None
    dataset_dirs = [p for p in base_path.iterdir() if p.is_dir()]
    if not dataset_dirs:
        return None
    dataset_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for dataset_dir in dataset_dirs:
        if list(dataset_dir.glob("data/chunk-*/episode_*.parquet")):
            return dataset_dir
    return None


def _read_video_frames(
    video_path: Path, expected_frames: int, image_width: int, image_height: int
) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[1] != image_width or frame.shape[0] != image_height:
                frame = cv2.resize(
                    frame,
                    (image_width, image_height),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    if len(frames) < expected_frames:
        last = frames[-1]
        for _ in range(expected_frames - len(frames)):
            frames.append(last.copy())
    elif len(frames) > expected_frames:
        frames = frames[:expected_frames]

    return np.stack(frames, axis=0).astype(np.uint8)


def _find_video_path(lerobot_dir: Path, camera_key: str, episode_idx: int) -> Path:
    pattern = f"videos/chunk-*/{camera_key}/episode_{episode_idx:06d}.mp4"
    matches = sorted(lerobot_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find video for episode {episode_idx} under '{camera_key}'. "
            f"Expected pattern: {pattern}"
        )
    return matches[0]


def convert_dataset(
    lerobot_dataset: Path,
    output_dir: Path,
    camera0_key: str,
    camera1_key: str,
    image_width: int,
    image_height: int,
    action_type: str,
) -> None:
    parquet_files = sorted(lerobot_dataset.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet episodes found in {lerobot_dataset}")

    output_dir.mkdir(parents=True, exist_ok=True)
    replay_buffer_path = output_dir / "replay_buffer.zarr"
    replay_buffer = ReplayBuffer.create_from_path(str(replay_buffer_path), mode="w")

    for episode_num, parquet_path in enumerate(parquet_files):
        episode_idx = int(parquet_path.stem.split("_")[1])
        print(
            f"[{episode_num + 1}/{len(parquet_files)}] converting episode {episode_idx:06d}"
        )

        df = pd.read_parquet(parquet_path)
        num_steps = len(df)
        if num_steps <= 0:
            print(f"  skipping empty episode: {parquet_path.name}")
            continue

        states = np.asarray(df["observation.state"].tolist(), dtype=np.float32)
        if action_type == "delta":
            actions = np.asarray(df["action"].tolist(), dtype=np.float32)
        elif action_type == "absolute":
            actions = states.copy()
        elif action_type == "absolute_next":
            actions = np.concatenate([states[1:], states[-1:]], axis=0)
        else:
            raise ValueError(f"Unsupported action_type: {action_type}")
        timestamps = np.asarray(df["timestamp"].to_numpy(), dtype=np.float64)
        step_idx = np.arange(num_steps, dtype=np.int64)
        episode_index = np.full((num_steps,), episode_idx, dtype=np.int64)

        camera0_video = _find_video_path(lerobot_dataset, camera0_key, episode_idx)
        camera1_video = _find_video_path(lerobot_dataset, camera1_key, episode_idx)
        camera0_frames = _read_video_frames(
            camera0_video, num_steps, image_width=image_width, image_height=image_height
        )
        camera1_frames = _read_video_frames(
            camera1_video, num_steps, image_width=image_width, image_height=image_height
        )

        episode = {
            "timestamp": timestamps,
            "step_idx": step_idx,
            "episode_index": episode_index,
            "robot_state": states,
            "action": actions,
            "camera_0": camera0_frames,
            "camera_1": camera1_frames,
        }
        chunks = {
            "timestamp": (256,),
            "step_idx": (256,),
            "episode_index": (256,),
            "robot_state": (256, states.shape[1]),
            "action": (256, actions.shape[1]),
            "camera_0": (1, image_height, image_width, 3),
            "camera_1": (1, image_height, image_width, 3),
        }
        compressors = {"camera_0": "disk", "camera_1": "disk"}
        replay_buffer.add_episode(episode, chunks=chunks, compressors=compressors)

    conversion_info = {
        "source_lerobot_dataset": str(lerobot_dataset.resolve()),
        "camera0_key": camera0_key,
        "camera1_key": camera1_key,
        "image_width": image_width,
        "image_height": image_height,
        "action_type": action_type,
        "num_episodes": replay_buffer.n_episodes,
        "num_steps": int(replay_buffer.n_steps),
    }
    (output_dir / "conversion_info.json").write_text(
        json.dumps(conversion_info, indent=2)
    )
    print(f"Finished conversion. Replay buffer saved to: {replay_buffer_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset to replay_buffer.zarr format for Diffusion Policy."
    )
    parser.add_argument(
        "--lerobot-dataset",
        type=Path,
        help="Path to LeRobot dataset directory. If omitted, latest dataset under --datasets-root is used.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets"),
        help="Base datasets directory used when --lerobot-dataset is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dp_policy/data/piper_lerobot_128"),
        help="Output directory where replay_buffer.zarr will be written.",
    )
    parser.add_argument(
        "--camera0-key",
        type=str,
        default="observation.images.global",
        help="LeRobot video key mapped to camera_0.",
    )
    parser.add_argument(
        "--camera1-key",
        type=str,
        default="observation.images.wrist",
        help="LeRobot video key mapped to camera_1.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=128,
        help="Converted frame width.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=128,
        help="Converted frame height.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output directory before conversion.",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default="delta",
        choices=("delta", "absolute", "absolute_next"),
        help=(
            "Action representation in replay buffer. "
            "'delta' uses recorded action deltas. "
            "'absolute' uses observation.state at same timestep as target. "
            "'absolute_next' uses next-timestep observation.state as target."
        ),
    )
    args = parser.parse_args()

    lerobot_dataset = args.lerobot_dataset
    if lerobot_dataset is None:
        lerobot_dataset = find_latest_lerobot_dataset(args.datasets_root)
        if lerobot_dataset is None:
            raise FileNotFoundError(
                f"No dataset found under {args.datasets_root}. "
                "Pass --lerobot-dataset explicitly."
            )
    lerobot_dataset = lerobot_dataset.expanduser().resolve()
    if not lerobot_dataset.exists():
        raise FileNotFoundError(f"LeRobot dataset does not exist: {lerobot_dataset}")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to rebuild."
            )
        shutil.rmtree(output_dir)

    convert_dataset(
        lerobot_dataset=lerobot_dataset,
        output_dir=output_dir,
        camera0_key=args.camera0_key,
        camera1_key=args.camera1_key,
        image_width=args.image_width,
        image_height=args.image_height,
        action_type=args.action_type,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
