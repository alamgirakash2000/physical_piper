#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_POLICY_ROOT = REPO_ROOT / "external_diffusion_policy"
for p in (str(DIFFUSION_POLICY_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402
from dataset_collector.core.camera_manager import CameraManager  # noqa: E402
from dataset_collector.core.robot_controller import RobotController  # noqa: E402
from dataset_collector.utils.config import load_config  # noqa: E402


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _load_policy(checkpoint: Path, device: torch.device, num_inference_steps: int | None = None):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.eval().to(device)

    if hasattr(policy, "num_inference_steps"):
        default_steps = int(policy.num_inference_steps)
        if num_inference_steps is not None and num_inference_steps > 0:
            policy.num_inference_steps = int(num_inference_steps)
            print(
                "Overriding diffusion inference steps: "
                f"{default_steps} -> {int(policy.num_inference_steps)}"
            )
        else:
            print(f"Using checkpoint diffusion inference steps: {default_steps}")
    return policy, cfg


def _setup_cameras(global_name: str, wrist_name: str) -> CameraManager:
    app_cfg = load_config()
    cam_cfg_map = {cam.name: cam for cam in app_cfg.cameras}
    missing = [name for name in (global_name, wrist_name) if name not in cam_cfg_map]
    if missing:
        raise ValueError(
            f"Missing cameras in config.json: {missing}. "
            f"Available cameras: {sorted(cam_cfg_map.keys())}"
        )

    cam_manager = CameraManager()
    for name in (global_name, wrist_name):
        cam = cam_cfg_map[name]
        cam_manager.add_camera(
            name=cam.name,
            device_index=cam.device_index,
            width=cam.width,
            height=cam.height,
            fps=cam.fps,
        )

    open_result = cam_manager.open_all()
    failed = [name for name, ok in open_result.items() if not ok]
    if failed:
        raise RuntimeError(f"Could not open cameras: {failed}")
    return cam_manager


def _resolve_replay_buffer_path(dataset_path: str | Path) -> Path | None:
    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    if path.is_dir():
        zarr_path = path / "replay_buffer.zarr"
        if zarr_path.exists():
            return zarr_path
    if path.name == "replay_buffer.zarr" and path.exists():
        return path
    return None


def _load_start_pose_from_dataset(cfg, state_key: str):
    dataset_path = cfg.task.get("dataset_path", None)
    if dataset_path is None:
        print("No task.dataset_path in checkpoint config; cannot load start pose.")
        return None

    replay_zarr_path = _resolve_replay_buffer_path(dataset_path)
    if replay_zarr_path is None:
        print(
            f"Could not find replay_buffer.zarr from task.dataset_path={dataset_path}; "
            "cannot load start pose from dataset."
        )
        return None

    try:
        replay_buffer = ReplayBuffer.create_from_path(str(replay_zarr_path), mode="r")
        if replay_buffer.n_episodes <= 0:
            print(f"Replay buffer has no episodes: {replay_zarr_path}")
            return None

        first_episode = replay_buffer.get_episode(0, copy=False)
        if state_key not in first_episode:
            print(
                f"State key '{state_key}' not found in replay buffer. "
                f"Available keys: {sorted(first_episode.keys())}"
            )
            return None

        start_state = np.asarray(first_episode[state_key][0], dtype=np.float32).reshape(-1)
        if start_state.size < 7:
            print(
                f"Expected >=7D state for start pose, got shape {start_state.shape}. "
                "Skipping auto start pose."
            )
            return None

        target_joint = start_state[:6].astype(np.float32)
        target_gripper = float(np.clip(start_state[6], 0.0, 100.0))
        return target_joint, target_gripper
    except Exception as exc:
        print(f"Failed to load start pose from replay buffer: {exc}")
        return None


def _infer_action_space_from_dataset(cfg) -> str:
    dataset_path = cfg.task.get("dataset_path", None)
    if dataset_path is None:
        return "delta"
    replay_zarr_path = _resolve_replay_buffer_path(dataset_path)
    if replay_zarr_path is None:
        return "delta"

    conversion_info_path = replay_zarr_path.parent / "conversion_info.json"
    if not conversion_info_path.exists():
        return "delta"
    try:
        payload = json.loads(conversion_info_path.read_text())
        action_type = str(payload.get("action_type", "delta")).strip().lower()
        if action_type in ("delta", "absolute", "absolute_next"):
            return action_type
    except Exception:
        pass
    return "delta"


def _read_dataset_action_type(cfg) -> str | None:
    dataset_path = cfg.task.get("dataset_path", None)
    if dataset_path is None:
        return None
    replay_zarr_path = _resolve_replay_buffer_path(dataset_path)
    if replay_zarr_path is None:
        return None
    conversion_info_path = replay_zarr_path.parent / "conversion_info.json"
    if not conversion_info_path.exists():
        return None
    try:
        payload = json.loads(conversion_info_path.read_text())
        action_type = str(payload.get("action_type", "")).strip().lower()
        return action_type or None
    except Exception:
        return None


def _move_robot_to_start_pose(
    robot: RobotController,
    target_joint: np.ndarray,
    target_gripper: float,
    speed: int,
    max_step_deg: float,
    step_dt: float,
) -> None:
    state = robot.get_state()
    current_joint = np.asarray(state.joint_positions, dtype=np.float32)
    current_gripper = float(state.gripper_position)

    max_step_deg = max(float(max_step_deg), 1e-3)
    max_joint_error = float(np.max(np.abs(target_joint - current_joint)))
    n_steps = max(1, int(np.ceil(max_joint_error / max_step_deg)))

    print(
        "Moving robot to learned start pose: "
        f"max_joint_error={max_joint_error:.2f}deg, steps={n_steps}, speed={speed}"
    )
    for i in range(1, n_steps + 1):
        alpha = i / float(n_steps)
        cmd_joint = current_joint + (target_joint - current_joint) * alpha
        cmd_gripper = current_gripper + (target_gripper - current_gripper) * alpha
        robot.set_joint_positions(cmd_joint.tolist(), speed=speed)
        robot.set_gripper(float(np.clip(cmd_gripper, 0.0, 100.0)))
        time.sleep(step_dt)

    time.sleep(0.3)


def _build_feed_preview(
    global_img: np.ndarray,
    wrist_img: np.ndarray,
    left_name: str,
    right_name: str,
    status_lines: list[str],
    scale: float,
) -> np.ndarray:
    if cv2 is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    left = global_img.copy()
    right = wrist_img.copy()
    target_h = max(left.shape[0], right.shape[0])

    def _resize_to_h(img: np.ndarray, out_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == out_h:
            return img
        out_w = max(1, int(round(w * (out_h / float(h)))))
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

    left = _resize_to_h(left, target_h)
    right = _resize_to_h(right, target_h)

    cv2.putText(left, left_name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(
        right, right_name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )
    canvas = np.hstack([left, right])

    if status_lines:
        line_h = 24
        pad = 8
        banner_h = pad * 2 + line_h * len(status_lines)
        banner = np.zeros((banner_h, canvas.shape[1], 3), dtype=np.uint8)
        for i, line in enumerate(status_lines):
            y = pad + (i + 1) * line_h - 6
            cv2.putText(
                banner, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2
            )
        canvas = np.vstack([banner, canvas])

    if scale > 0 and abs(scale - 1.0) > 1e-6:
        out_w = max(1, int(round(canvas.shape[1] * scale)))
        out_h = max(1, int(round(canvas.shape[0] * scale)))
        canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return canvas


def _start_web_feed_server(host: str, port: int):
    state = {
        "jpeg": None,
        "lock": threading.Lock(),
        "running": True,
    }

    class FeedHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>DP Policy Feed</title>"
                    "<style>body{background:#111;color:#eee;font-family:monospace;margin:0}"
                    "h3{margin:12px}img{display:block;max-width:100vw;height:auto}</style>"
                    "</head><body><h3>DP Policy Feed</h3><img src='/stream.mjpg'/></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while state["running"]:
                    with state["lock"]:
                        frame = state["jpeg"]
                    if frame is None:
                        time.sleep(0.03)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                pass

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer((host, int(port)), FeedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


def _update_web_feed_frame(feed_state, frame_bgr: np.ndarray) -> None:
    if cv2 is None or feed_state is None:
        return
    try:
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with feed_state["lock"]:
            feed_state["jpeg"] = encoded.tobytes()
    except Exception:
        return


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a trained Diffusion Policy checkpoint on Piper robot in closed loop."
    )
    parser.add_argument("--checkpoint", "-c", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=-1,
        help=(
            "Diffusion denoising steps at runtime. "
            "Use -1 to keep checkpoint default (recommended)."
        ),
    )
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--global-camera-name", type=str, default="global")
    parser.add_argument("--wrist-camera-name", type=str, default="wrist")
    parser.add_argument("--robot-speed", type=int, default=40)
    parser.add_argument("--max-joint-delta-deg", type=float, default=4.0)
    parser.add_argument("--max-gripper-delta", type=float, default=8.0)
    parser.add_argument(
        "--action-space",
        type=str,
        default="auto",
        choices=("auto", "delta", "absolute", "absolute_next"),
        help=(
            "Interpretation of policy action output. "
            "'auto' reads conversion_info.json action_type when available."
        ),
    )
    parser.add_argument(
        "--action-index",
        type=int,
        default=-1,
        help=(
            "Index into predicted action chunk to execute. "
            "Use -1 for auto: 0 for delta mode, middle index for absolute/absolute_next."
        ),
    )
    parser.add_argument(
        "--use-action-chunk",
        action="store_true",
        help=(
            "Execute a short chunk of predicted actions per inference instead of "
            "re-inferring every control step."
        ),
    )
    parser.add_argument(
        "--chunk-start-index",
        type=int,
        default=-1,
        help=(
            "Start index in predicted action chunk when --use-action-chunk is enabled. "
            "Use -1 for auto: middle index for absolute/absolute_next, 0 for delta."
        ),
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=4,
        help="Number of predicted actions to execute per inference when chunk mode is enabled.",
    )
    parser.add_argument(
        "--auto-start-pose",
        action="store_true",
        default=True,
        help="Move to first demo pose from task.dataset_path before policy rollout.",
    )
    parser.add_argument(
        "--no-auto-start-pose",
        dest="auto_start_pose",
        action="store_false",
        help="Disable moving to learned start pose before rollout.",
    )
    parser.add_argument(
        "--start-pose-speed",
        type=int,
        default=40,
        help="Robot speed used for moving to learned start pose.",
    )
    parser.add_argument(
        "--start-pose-max-step-deg",
        type=float,
        default=2.0,
        help="Maximum joint interpolation step (deg) while moving to start pose.",
    )
    parser.add_argument(
        "--start-pose-step-dt",
        type=float,
        default=0.03,
        help="Seconds between start-pose interpolation commands.",
    )
    parser.add_argument(
        "--joint-delta-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to predicted joint deltas before clipping.",
    )
    parser.add_argument(
        "--gripper-delta-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to predicted gripper deltas before clipping.",
    )
    parser.add_argument(
        "--joint-deadband-deg",
        type=float,
        default=0.0,
        help="Ignore absolute joint delta commands smaller than this threshold (deg).",
    )
    parser.add_argument(
        "--gripper-deadband",
        type=float,
        default=0.0,
        help="Ignore absolute gripper delta commands smaller than this threshold.",
    )
    parser.add_argument(
        "--max-target-offset-deg",
        type=float,
        default=12.0,
        help=(
            "Limit commanded target joint offset from measured joints in deg. "
            "Use <=0 to disable."
        ),
    )
    parser.add_argument(
        "--integrate-target",
        action="store_true",
        default=True,
        help=(
            "Accumulate policy deltas on previously commanded target (recommended for "
            "small-action policies)."
        ),
    )
    parser.add_argument(
        "--no-integrate-target",
        dest="integrate_target",
        action="store_false",
        help="Disable target integration and apply deltas on current measured state.",
    )
    parser.add_argument(
        "--show-feed",
        action="store_true",
        help="Show live global+wrist camera preview while policy is running.",
    )
    parser.add_argument(
        "--feed-scale",
        type=float,
        default=1.0,
        help="Display scale for preview window (e.g., 0.75, 1.0, 1.5).",
    )
    parser.add_argument(
        "--show-feed-web",
        action="store_true",
        help="Serve live camera preview over HTTP MJPEG (works without OpenCV GUI).",
    )
    parser.add_argument(
        "--feed-web-host",
        type=str,
        default="127.0.0.1",
        help="Host for MJPEG web preview server.",
    )
    parser.add_argument(
        "--feed-web-port",
        type=int,
        default=8765,
        help="Port for MJPEG web preview server.",
    )
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.device)
    print(f"Loading policy from {checkpoint}")
    requested_inference_steps = None if args.num_inference_steps < 0 else args.num_inference_steps
    policy, cfg = _load_policy(
        checkpoint,
        device=device,
        num_inference_steps=requested_inference_steps,
    )
    if args.action_space == "auto":
        action_space = _infer_action_space_from_dataset(cfg)
    else:
        action_space = args.action_space
    print(f"Using action space: {action_space}")
    dataset_action_type = _read_dataset_action_type(cfg)
    if dataset_action_type is not None:
        print(f"Dataset action_type from conversion_info.json: {dataset_action_type}")
    if action_space == "absolute" and dataset_action_type == "absolute":
        print(
            "WARNING: This checkpoint was trained with same-step absolute targets "
            "(action[t]=state[t]). This frequently causes weak/no-progress motion on robot. "
            "Retrain with --action-type absolute_next for robust movement."
        )
    auto_action_index = (
        int(cfg.n_action_steps) // 2 if action_space in ("absolute", "absolute_next") else 0
    )
    selected_action_index = (
        auto_action_index if args.action_index < 0 else int(max(0, args.action_index))
    )
    print(f"Using action index: {selected_action_index} (auto={args.action_index < 0})")

    obs_shape_meta = cfg.task.shape_meta["obs"]
    rgb_keys = [k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "rgb"]
    lowdim_keys = [k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "low_dim"]
    if len(rgb_keys) != 2:
        raise ValueError(f"Expected 2 RGB obs keys in checkpoint config, got: {rgb_keys}")
    if not lowdim_keys:
        raise ValueError("Checkpoint config has no low_dim observation key.")
    state_key = lowdim_keys[0]
    n_obs_steps = int(cfg.n_obs_steps)

    cam_manager = _setup_cameras(
        global_name=args.global_camera_name, wrist_name=args.wrist_camera_name
    )
    cam_manager.start_capture(target_fps=max(20.0, args.frequency * 2.0))

    show_feed = bool(args.show_feed)
    show_feed_web = bool(args.show_feed_web)
    feed_window_name = "DP Policy Feed"
    web_feed_server = None
    web_feed_state = None
    if show_feed:
        if cv2 is None:
            print("OpenCV GUI not available; disabling --show-feed.")
            show_feed = False
        else:
            try:
                cv2.namedWindow(feed_window_name, cv2.WINDOW_NORMAL)
            except Exception as exc:
                print(f"Could not open preview window ({exc}); disabling --show-feed.")
                show_feed = False
                if cv2 is not None:
                    show_feed_web = True
                    print("Falling back to web preview (--show-feed-web).")

    if show_feed_web:
        if cv2 is None:
            print("OpenCV is unavailable; cannot encode web preview frames.")
            show_feed_web = False
        else:
            try:
                web_feed_server, web_feed_state = _start_web_feed_server(
                    host=args.feed_web_host,
                    port=args.feed_web_port,
                )
                print(
                    "Web preview available at "
                    f"http://{args.feed_web_host}:{args.feed_web_port}/"
                )
            except Exception as exc:
                print(f"Failed to start web preview server ({exc}); disabling --show-feed-web.")
                show_feed_web = False

    robot = None
    simulated_state = np.zeros((7,), dtype=np.float32)
    if not args.dry_run:
        robot = RobotController()
        print("Connecting to robot...")
        if not robot.connect():
            cam_manager.stop_capture()
            cam_manager.close_all()
            raise RuntimeError("Failed to connect to robot.")
        print("Robot connected.")

        if args.auto_start_pose:
            start_pose = _load_start_pose_from_dataset(cfg=cfg, state_key=state_key)
            if start_pose is not None:
                start_joint, start_gripper = start_pose
                _move_robot_to_start_pose(
                    robot=robot,
                    target_joint=start_joint,
                    target_gripper=start_gripper,
                    speed=args.start_pose_speed,
                    max_step_deg=args.start_pose_max_step_deg,
                    step_dt=args.start_pose_step_dt,
                )
            else:
                print("Falling back to go_home() before policy rollout.")
                robot.go_home(speed=args.start_pose_speed)
    else:
        print("Running in dry-run mode (no robot commands).")

    camera0_hist = deque(maxlen=n_obs_steps)
    camera1_hist = deque(maxlen=n_obs_steps)
    state_hist = deque(maxlen=n_obs_steps)

    print(
        f"Warming up cameras/state buffers for {args.warmup_seconds:.1f}s "
        f"(need {n_obs_steps} obs steps)..."
    )
    warmup_end = time.time() + args.warmup_seconds
    while time.time() < warmup_end:
        frames = cam_manager.get_latest_frames()
        global_frame = frames.get(args.global_camera_name)
        wrist_frame = frames.get(args.wrist_camera_name)
        if global_frame is not None and wrist_frame is not None:
            camera0_hist.append(global_frame.frame.copy())
            camera1_hist.append(wrist_frame.frame.copy())

            if robot is not None:
                state = robot.get_state()
                robot_state = np.concatenate(
                    [state.joint_positions, [state.gripper_position]]
                ).astype(np.float32)
            else:
                robot_state = simulated_state.copy()
            state_hist.append(robot_state)

            if (show_feed or show_feed_web) and cv2 is not None:
                preview = _build_feed_preview(
                    global_img=global_frame.frame,
                    wrist_img=wrist_frame.frame,
                    left_name=args.global_camera_name,
                    right_name=args.wrist_camera_name,
                    status_lines=[
                        "Warmup...",
                        f"obs_steps={n_obs_steps}",
                    ],
                    scale=float(args.feed_scale),
                )
                if show_feed:
                    try:
                        cv2.imshow(feed_window_name, preview)
                        cv2.waitKey(1)
                    except Exception as exc:
                        print(f"Preview error ({exc}); disabling --show-feed.")
                        show_feed = False
                if show_feed_web:
                    _update_web_feed_frame(web_feed_state, preview)
        time.sleep(0.01)

    input(
        "Policy is ready. Press ENTER to start closed-loop control (Ctrl+C to stop)..."
    )

    dt = 1.0 / args.frequency
    last_log_time = 0.0
    step_count = 0
    inference_count = 0
    prev_measured_state = None
    commanded_joint = None
    commanded_gripper = None
    pending_actions: deque[np.ndarray] = deque()
    try:
        while True:
            cycle_start = time.perf_counter()
            frames = cam_manager.get_latest_frames()
            global_frame = frames.get(args.global_camera_name)
            wrist_frame = frames.get(args.wrist_camera_name)
            if global_frame is None or wrist_frame is None:
                time.sleep(0.01)
                continue

            if robot is not None:
                state = robot.get_state()
                current_state = np.concatenate(
                    [state.joint_positions, [state.gripper_position]]
                ).astype(np.float32)
            else:
                current_state = simulated_state.copy()

            if commanded_joint is None:
                commanded_joint = current_state[:6].copy()
                commanded_gripper = float(current_state[6])

            camera0_hist.append(global_frame.frame.copy())
            camera1_hist.append(wrist_frame.frame.copy())
            state_hist.append(current_state)
            if (
                len(camera0_hist) < n_obs_steps
                or len(camera1_hist) < n_obs_steps
                or len(state_hist) < n_obs_steps
            ):
                continue

            env_obs = {
                rgb_keys[0]: np.stack(camera0_hist, axis=0),
                rgb_keys[1]: np.stack(camera1_hist, axis=0),
                state_key: np.stack(state_hist, axis=0).astype(np.float32),
            }
            obs_dict_np = get_real_obs_dict(env_obs=env_obs, shape_meta=cfg.task.shape_meta)
            obs_dict = dict_apply(
                obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
            )

            if pending_actions:
                action = pending_actions.popleft()
            else:
                with torch.no_grad():
                    result = policy.predict_action(obs_dict)
                    action_seq = result["action"][0].detach().cpu().numpy().astype(np.float32)
                inference_count += 1

                if args.use_action_chunk:
                    if action_seq.shape[0] <= 0:
                        continue
                    auto_chunk_start = (
                        action_seq.shape[0] // 2
                        if action_space in ("absolute", "absolute_next")
                        else 0
                    )
                    chunk_start_raw = (
                        auto_chunk_start if args.chunk_start_index < 0 else args.chunk_start_index
                    )
                    start_idx = int(
                        np.clip(chunk_start_raw, 0, max(0, action_seq.shape[0] - 1))
                    )
                    chunk_len = max(1, int(args.chunk_length))
                    end_idx = min(action_seq.shape[0], start_idx + chunk_len)
                    for a in action_seq[start_idx:end_idx]:
                        pending_actions.append(a.copy())
                    action = pending_actions.popleft()
                else:
                    action_idx = int(np.clip(selected_action_index, 0, action_seq.shape[0] - 1))
                    action = action_seq[action_idx]

            if action_space == "delta":
                scaled_joint_delta = action[:6] * float(args.joint_delta_scale)
                scaled_gripper_delta = action[6] * 100.0 * float(args.gripper_delta_scale)
                joint_delta = np.clip(
                    scaled_joint_delta, -args.max_joint_delta_deg, args.max_joint_delta_deg
                )
                gripper_delta = float(
                    np.clip(
                        scaled_gripper_delta,
                        -args.max_gripper_delta,
                        args.max_gripper_delta,
                    )
                )

                if args.joint_deadband_deg > 0:
                    joint_delta[np.abs(joint_delta) < args.joint_deadband_deg] = 0.0
                if args.gripper_deadband > 0 and abs(gripper_delta) < args.gripper_deadband:
                    gripper_delta = 0.0

                if args.integrate_target:
                    base_joint = commanded_joint
                    base_gripper = float(commanded_gripper)
                else:
                    base_joint = current_state[:6]
                    base_gripper = float(current_state[6])

                target_joint = base_joint + joint_delta
                if args.max_target_offset_deg > 0:
                    offset_from_measured = np.clip(
                        target_joint - current_state[:6],
                        -args.max_target_offset_deg,
                        args.max_target_offset_deg,
                    )
                    target_joint = current_state[:6] + offset_from_measured
                target_gripper = float(np.clip(base_gripper + gripper_delta, 0.0, 100.0))
            else:
                target_joint = action[:6].copy()
                target_gripper = float(np.clip(action[6], 0.0, 100.0))

                if args.max_target_offset_deg > 0:
                    offset_from_measured = np.clip(
                        target_joint - current_state[:6],
                        -args.max_target_offset_deg,
                        args.max_target_offset_deg,
                    )
                    target_joint = current_state[:6] + offset_from_measured

                joint_delta = target_joint - current_state[:6]
                gripper_delta = float(target_gripper - current_state[6])
                if args.joint_deadband_deg > 0:
                    joint_delta[np.abs(joint_delta) < args.joint_deadband_deg] = 0.0
                    target_joint = current_state[:6] + joint_delta
                if args.gripper_deadband > 0 and abs(gripper_delta) < args.gripper_deadband:
                    gripper_delta = 0.0
                    target_gripper = float(current_state[6])

            if robot is not None:
                if np.max(np.abs(target_joint - commanded_joint)) > 1e-6:
                    robot.set_joint_positions(target_joint.tolist(), speed=args.robot_speed)
                if abs(target_gripper - float(commanded_gripper)) > 1e-6:
                    robot.set_gripper(target_gripper)
            else:
                simulated_state[:6] = target_joint
                simulated_state[6] = target_gripper

            commanded_joint = target_joint.copy()
            commanded_gripper = target_gripper

            step_count += 1
            now = time.time()
            if now - last_log_time > 1.0:
                measured_delta_max = 0.0
                if prev_measured_state is not None:
                    measured_delta_max = float(
                        np.max(np.abs(current_state[:6] - prev_measured_state[:6]))
                    )
                tracking_error_max = float(
                    np.max(np.abs(commanded_joint - current_state[:6]))
                )
                print(
                    f"step={step_count} "
                    f"inference_count={inference_count} "
                    f"pending_actions={len(pending_actions)} "
                    f"joint_delta_max={np.max(np.abs(joint_delta)):.3f}deg "
                    f"measured_joint_delta_max={measured_delta_max:.3f}deg "
                    f"tracking_error_max={tracking_error_max:.3f}deg "
                    f"gripper_delta={gripper_delta:.3f}"
                )
                last_log_time = now

            if (show_feed or show_feed_web) and cv2 is not None:
                preview = _build_feed_preview(
                    global_img=global_frame.frame,
                    wrist_img=wrist_frame.frame,
                    left_name=args.global_camera_name,
                    right_name=args.wrist_camera_name,
                    status_lines=[
                        f"step={step_count} action_space={action_space}",
                        f"joint_delta_max={np.max(np.abs(joint_delta)):.3f} deg",
                        f"tracking_error_max={np.max(np.abs(commanded_joint - current_state[:6])):.3f} deg",
                        "Press 'q' or ESC to stop (GUI)",
                    ],
                    scale=float(args.feed_scale),
                )
                if show_feed:
                    try:
                        cv2.imshow(feed_window_name, preview)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            print("Preview requested stop.")
                            break
                    except Exception as exc:
                        print(f"Preview error ({exc}); disabling --show-feed.")
                        show_feed = False
                if show_feed_web:
                    _update_web_feed_frame(web_feed_state, preview)
            prev_measured_state = current_state.copy()

            elapsed = time.perf_counter() - cycle_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopping policy control.")
    finally:
        cam_manager.stop_capture()
        cam_manager.close_all()
        if show_feed and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if web_feed_state is not None:
            web_feed_state["running"] = False
        if web_feed_server is not None:
            try:
                web_feed_server.shutdown()
                web_feed_server.server_close()
            except Exception:
                pass
        if robot is not None:
            robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
