# Piper Diffusion Policy (Minimal)

## You already started training

If this is running, do not stop it:

```bash
python dp_policy/train_dp.py \
  --lerobot-dataset datasets/pick_the_white_cup_and_place_it_on_the_red_cup \
  --dp-dataset dp_policy/data/piper_lerobot_128 \
  --device cuda:0 \
  --num-epochs 200
```

Next step after training finishes:

```bash
python dp_policy/run_dp_policy.py \
  --checkpoint "$(ls -td dp_policy/outputs/*/*/checkpoints/latest.ckpt | head -n 1)" \
  --device cuda:0 \
  --frequency 10
```

To view live `global` + `wrist` camera feeds while running:

```bash
python dp_policy/run_dp_policy.py \
  --checkpoint "$(ls -td dp_policy/outputs/*/*/checkpoints/latest.ckpt | head -n 1)" \
  --device cuda:0 \
  --frequency 30 \
  --show-feed
```

If OpenCV GUI is unavailable (headless build), use browser preview:

```bash
python dp_policy/run_dp_policy.py \
  --checkpoint "$(ls -td dp_policy/outputs/*/*/checkpoints/latest.ckpt | head -n 1)" \
  --device cuda:0 \
  --frequency 30 \
  --show-feed-web \
  --feed-web-host 127.0.0.1 \
  --feed-web-port 8765
```

Then open `http://127.0.0.1:8765/` in your browser.

`run_dp_policy.py` now automatically moves the robot to the learned start pose
(first pose of episode 0 in `task.dataset_path`) before closed-loop rollout.
Disable this only if you want manual placement:

```bash
python dp_policy/run_dp_policy.py \
  --checkpoint "$(ls -td dp_policy/outputs/*/*/checkpoints/latest.ckpt | head -n 1)" \
  --device cuda:0 \
  --frequency 10 \
  --no-auto-start-pose
```

If motion is jerky with very small updates, try this tuned command:

```bash
python dp_policy/run_dp_policy.py \
  --checkpoint "$(ls -td dp_policy/outputs/*/*/checkpoints/latest.ckpt | head -n 1)" \
  --device cuda:0 \
  --frequency 30 \
  --robot-speed 100 \
  --joint-delta-scale 3.0 \
  --joint-deadband-deg 0.15 \
  --gripper-deadband 0.02 \
  --max-joint-delta-deg 8
```

If you want behavior closer to `replay_demo.py`, train with next-state absolute targets:

```bash
python dp_policy/train_dp.py \
  --lerobot-dataset datasets/pick_the_white_cup_and_place_it_on_the_red_cup \
  --dp-dataset dp_policy/data/piper_lerobot_128_abs_next \
  --action-type absolute_next \
  --device cuda:0 \
  --num-epochs 200
```

Then run using that checkpoint and dataset/action mode:

```bash
python dp_policy/run_dp_policy.py \
  --checkpoint "<checkpoint_from_abs_next_run>/checkpoints/latest.ckpt" \
  --device cuda:0 \
  --frequency 30 \
  --robot-speed 100 \
  --action-space absolute_next \
  --use-action-chunk \
  --chunk-length 4 \
  --max-target-offset-deg 25
```

If movement is still weak, keep full diffusion quality and preview in browser:

```bash
python dp_policy/run_dp_policy.py \
  --checkpoint "<checkpoint_from_abs_next_run>/checkpoints/latest.ckpt" \
  --device cuda:0 \
  --frequency 30 \
  --robot-speed 100 \
  --action-space absolute_next \
  --use-action-chunk \
  --chunk-length 4 \
  --num-inference-steps 32 \
  --show-feed-web \
  --feed-web-host 127.0.0.1 \
  --feed-web-port 8765 \
  --max-target-offset-deg 25
```

## Only required commands (from scratch)

```bash
conda activate aeropiper
```

```bash
python dp_policy/train_dp.py \
  --lerobot-dataset datasets/pick_the_white_cup_and_place_it_on_the_red_cup \
  --dp-dataset dp_policy/data/piper_lerobot_128 \
  --device cuda:0 \
  --num-epochs 200
```

`train_dp.py` already converts the LeRobot dataset automatically, so you do not need to run a separate conversion command.

If imports are missing, install once:

```bash
pip install -r dp_policy/requirements.txt
pip install -e external_diffusion_policy
```
