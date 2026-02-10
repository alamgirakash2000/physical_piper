#!/usr/bin/env python3
"""Repair LeRobot-style dataset videos for stable training ingestion.

This script validates and optionally rewrites per-episode videos so they:
1. Use a fixed target FPS (usually the dataset FPS from meta/info.json).
2. Keep frame counts aligned with episode lengths.
3. Have consistent frame shape/type for decoder stability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import pandas as pd


@dataclass
class VideoCheck:
    path: Path
    episode_idx: int
    input_fps: float
    input_frames: int
    expected_frames: int | None
    width: int
    height: int
    needs_repair: bool
    reason: str


def _episode_from_name(path: Path) -> int:
    # episode_000123.mp4 -> 123
    stem = path.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(stem.split("_", 1)[1])


def _load_expected_lengths(dataset_path: Path) -> Dict[int, int]:
    lengths: Dict[int, int] = {}
    episodes_jsonl = dataset_path / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        for line in episodes_jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj["episode_index"])
            lengths[idx] = int(obj["length"])
        return lengths

    # Fallback: infer from parquet row counts
    parquet_root = dataset_path / "data"
    for parquet in sorted(parquet_root.rglob("episode_*.parquet")):
        idx = _episode_from_name(parquet)
        lengths[idx] = len(pd.read_parquet(parquet))
    return lengths


def _detect_target_fps(dataset_path: Path, cli_fps: float | None) -> float:
    if cli_fps is not None:
        return float(cli_fps)
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if "fps" in info:
            return float(info["fps"])
    return 30.0


def _normalize_frame(frame, target_size):
    # Ensure BGR uint8 contiguous frame for VideoWriter.
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"unsupported frame shape {frame.shape}")

    if frame.dtype != "uint8":
        frame = frame.clip(0, 255).astype("uint8")

    if (frame.shape[1], frame.shape[0]) != target_size:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    if not frame.flags["C_CONTIGUOUS"]:
        frame = frame.copy(order="C")

    return frame


def _inspect_video(
    video_path: Path, expected_lengths: Dict[int, int], target_fps: float, fps_tol: float
) -> VideoCheck:
    ep_idx = _episode_from_name(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    input_fps = float(cap.get(cv2.CAP_PROP_FPS))
    input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    expected = expected_lengths.get(ep_idx)

    reasons = []
    if expected is not None and input_frames != expected:
        reasons.append(f"frame_count {input_frames} != expected {expected}")
    if abs(input_fps - target_fps) > fps_tol:
        reasons.append(f"fps {input_fps:.3f} != target {target_fps:.3f}")
    if width <= 0 or height <= 0:
        reasons.append("invalid dimensions")

    needs = len(reasons) > 0
    reason = "; ".join(reasons) if reasons else "ok"
    return VideoCheck(
        path=video_path,
        episode_idx=ep_idx,
        input_fps=input_fps,
        input_frames=input_frames,
        expected_frames=expected,
        width=width,
        height=height,
        needs_repair=needs,
        reason=reason,
    )


def _repair_video(
    check: VideoCheck, target_fps: float, codec: str, keep_backup: bool
) -> tuple[int, float]:
    path = check.path
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open for repair: {path}")

    out_size = (check.width, check.height)
    if out_size[0] <= 0 or out_size[1] <= 0:
        cap.release()
        raise RuntimeError(f"Invalid size for {path}: {out_size}")

    # Keep a valid video extension so OpenCV can select the right muxer.
    tmp_path = path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(tmp_path), fourcc, float(target_fps), out_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Failed to open writer for {tmp_path} (codec={codec}, fps={target_fps}, size={out_size})"
        )

    wrote = 0
    last_valid = None
    limit = check.expected_frames if check.expected_frames is not None else None

    try:
        while True:
            if limit is not None and wrote >= limit:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = _normalize_frame(frame, out_size)
            writer.write(frame)
            last_valid = frame
            wrote += 1

        # If short read happened, pad with last frame to keep alignment.
        if limit is not None and wrote < limit and last_valid is not None:
            for _ in range(limit - wrote):
                writer.write(last_valid)
                wrote += 1
    finally:
        cap.release()
        writer.release()

    out_cap = cv2.VideoCapture(str(tmp_path))
    if not out_cap.isOpened():
        raise RuntimeError(f"Failed to open repaired output: {tmp_path}")
    out_count = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = float(out_cap.get(cv2.CAP_PROP_FPS))
    out_cap.release()

    backup_path = path.with_suffix(path.suffix + ".bak")
    if keep_backup:
        if backup_path.exists():
            backup_path.unlink()
        path.rename(backup_path)
    else:
        path.unlink()
    tmp_path.rename(path)
    if not keep_backup and backup_path.exists():
        backup_path.unlink()

    return out_count, out_fps


def _update_info_json(dataset_path: Path, target_fps: float, codec: str):
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return
    info = json.loads(info_path.read_text())
    info["fps"] = int(round(target_fps))

    # Keep feature metadata coherent when present.
    features = info.get("features", {})
    codec_name = "mpeg4" if codec.lower() == "mp4v" else codec
    for key, value in features.items():
        if not isinstance(value, dict):
            continue
        if not key.startswith("observation.images."):
            continue
        video_info = value.get("video_info")
        if not isinstance(video_info, dict):
            continue
        video_info["video.fps"] = int(round(target_fps))
        video_info["video.codec"] = codec_name

    info_path.write_text(json.dumps(info, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--fps-tol", type=float, default=0.05)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-backup", action="store_true")
    args = parser.parse_args()

    dataset_path: Path = args.dataset_path
    videos_root = dataset_path / "videos"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not videos_root.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_root}")

    expected_lengths = _load_expected_lengths(dataset_path)
    target_fps = _detect_target_fps(dataset_path, args.target_fps)

    videos = sorted(videos_root.rglob("episode_*.mp4"))
    if not videos:
        print("No videos found.")
        return

    checks: list[VideoCheck] = []
    for path in videos:
        check = _inspect_video(path, expected_lengths, target_fps, args.fps_tol)
        if args.force:
            check.needs_repair = True
            check.reason = f"force repair (was: {check.reason})"
        checks.append(check)

    needs = [c for c in checks if c.needs_repair]

    print(f"Dataset: {dataset_path}")
    print(f"Target FPS: {target_fps:.3f}")
    print(f"Videos scanned: {len(checks)}")
    print(f"Videos needing repair: {len(needs)}")

    for c in needs[:50]:
        print(
            f"- {c.path}: {c.reason} "
            f"(in_fps={c.input_fps:.3f}, in_frames={c.input_frames}, expected={c.expected_frames})"
        )
    if len(needs) > 50:
        print(f"... {len(needs) - 50} more")

    if args.dry_run:
        print("Dry run only; no files modified.")
        return

    repaired = 0
    for c in needs:
        out_count, out_fps = _repair_video(
            c, target_fps=target_fps, codec=args.codec, keep_backup=args.keep_backup
        )
        repaired += 1
        print(
            f"Repaired {c.path.name}: out_frames={out_count}, out_fps={out_fps:.3f}, "
            f"expected={c.expected_frames}"
        )

    _update_info_json(dataset_path, target_fps=target_fps, codec=args.codec)
    print(f"Completed. Repaired {repaired}/{len(needs)} videos.")


if __name__ == "__main__":
    main()
