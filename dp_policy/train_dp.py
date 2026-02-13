#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _find_latest_lerobot_dataset(base_path: Path) -> Path | None:
    if not base_path.exists():
        return None
    dataset_dirs = [p for p in base_path.iterdir() if p.is_dir()]
    dataset_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for dataset_dir in dataset_dirs:
        if list(dataset_dir.glob("data/chunk-*/episode_*.parquet")):
            return dataset_dir
    return None


def _check_runtime_dependencies() -> None:
    required_modules = ("zarr", "numba", "hydra", "threadpoolctl")
    missing_modules = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_modules.append(module_name)

    if missing_modules:
        raise RuntimeError(
            "Missing required modules for Diffusion Policy training: "
            f"{', '.join(missing_modules)}. "
            "Install them in your environment with:\n"
            "  pip install hydra-core threadpoolctl numba "
            "\"zarr<3\" \"numcodecs<0.16\""
        )

    import zarr  # imported after availability check

    version = getattr(zarr, "__version__", "unknown")
    major_str = str(version).split(".", maxsplit=1)[0]
    if major_str.isdigit() and int(major_str) >= 3:
        raise RuntimeError(
            f"Incompatible zarr version detected: {version}. "
            "external_diffusion_policy expects zarr<3. "
            "Please run:\n"
            "  pip install --force-reinstall \"zarr<3\" \"numcodecs<0.16\""
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy on Piper LeRobot demonstrations."
    )
    parser.add_argument(
        "--lerobot-dataset",
        type=Path,
        help="Path to LeRobot dataset root. If omitted, latest dataset under --datasets-root is used.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets"),
        help="Dataset root used when --lerobot-dataset is omitted.",
    )
    parser.add_argument(
        "--dp-dataset",
        type=Path,
        default=Path("dp_policy/data/piper_lerobot_128"),
        help="Path where converted replay_buffer.zarr is/will be stored.",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip LeRobot->DP conversion and directly run training.",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Force conversion even if replay_buffer.zarr already exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device override, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Override training.num_epochs from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override dataloader and val_dataloader batch size.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="train_piper_diffusion_unet_image_workspace.yaml",
        help="Hydra config name in dp_policy/config.",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default="delta",
        choices=("delta", "absolute", "absolute_next"),
        help=(
            "Action representation used when converting LeRobot dataset. "
            "'delta' trains delta-action policy; "
            "'absolute' trains absolute-target policy; "
            "'absolute_next' trains next-state-target policy."
        ),
    )
    args, hydra_overrides = parser.parse_known_args()
    _check_runtime_dependencies()

    env = os.environ.copy()
    pythonpath_parts = [
        str(REPO_ROOT / "external_diffusion_policy"),
        str(REPO_ROOT),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    dp_dataset_path = args.dp_dataset.expanduser()
    replay_zarr_path = dp_dataset_path / "replay_buffer.zarr"

    if not args.skip_conversion:
        lerobot_dataset = args.lerobot_dataset
        if lerobot_dataset is None:
            lerobot_dataset = _find_latest_lerobot_dataset(args.datasets_root)
            if lerobot_dataset is None:
                raise FileNotFoundError(
                    f"No LeRobot dataset found under {args.datasets_root}."
                )

        needs_conversion = args.force_conversion or (not replay_zarr_path.exists())
        if needs_conversion:
            convert_cmd = [
                sys.executable,
                str(REPO_ROOT / "dp_policy" / "convert_lerobot_to_dp.py"),
                "--lerobot-dataset",
                str(lerobot_dataset),
                "--output-dir",
                str(dp_dataset_path),
                "--overwrite",
                "--action-type",
                args.action_type,
            ]
            _run(convert_cmd, env=env)
        else:
            print(f"Using existing converted dataset: {replay_zarr_path}")

    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "external_diffusion_policy" / "train.py"),
        "--config-dir",
        str(REPO_ROOT / "dp_policy" / "config"),
        "--config-name",
        args.config_name,
        f"task.dataset_path={str(dp_dataset_path.resolve())}",
        f"training.device={args.device}",
    ]
    if args.num_epochs is not None:
        train_cmd.append(f"training.num_epochs={args.num_epochs}")
    if args.batch_size is not None:
        train_cmd.append(f"dataloader.batch_size={args.batch_size}")
        train_cmd.append(f"val_dataloader.batch_size={args.batch_size}")
    train_cmd.extend(hydra_overrides)
    _run(train_cmd, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
