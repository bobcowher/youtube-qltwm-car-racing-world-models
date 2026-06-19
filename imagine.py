"""
Visualize the agent playing in pure imagination.

Loads saved checkpoints, advances the real env to a fixed race-start
position, then hands off entirely to the world model. Each step:
  encode → q_model (via decoded frame) → dynamics (latent) → decode → display

Usage:
    python imagine.py [--steps N] [--warmup N] [--scale N] [--out FILE]

    --steps   imagination steps to run  (default: 500)
    --warmup  real env steps before handoff, gets past zoom-in (default: 50)
    --scale   display upscale factor    (default: 4)
    --out     video output path         (default: imagination.mp4)
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import gymnasium as gym
from agent import Agent


CHECKPOINT_DIR = "checkpoints"


def run_imagination(steps: int = 500, warmup: int = 50, scale: int = 4, out: str = "imagination.mp4", fps: int = 50):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    agent = Agent(env=env, max_buffer_size=1000)
    # agent.load() only restores q_model — load world_model explicitly
    agent.load()
    agent.world_model.load_the_model("world_model", device=device)
    agent.world_model.eval()
    agent.q_model.eval()

    n_actions = agent.world_model.n_actions

    # Advance real env to race start, past the camera zoom-in
    obs, _ = env.reset()
    for _ in range(warmup):
        obs, _, term, trunc, _ = env.step(3)  # action 3 = gas
        if term or trunc:
            obs, _ = env.reset()

    # Encode the fixed starting frame
    start_obs = agent.process_observation(obs)
    obs_norm = start_obs.unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        current_embed = agent.world_model.encode(obs_norm)  # (1, embed_dim)

    # Video writer
    frame_size = (96 * scale, 96 * scale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, fps, frame_size)

    window = "Imagination"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, *frame_size)

    print(f"Running {steps} imagination steps from race start (warmup={warmup}).")
    print("Press 'q' to quit early.")

    total_reward = 0.0

    with torch.no_grad():
        for step in range(steps):
            # Decode current embed to pixels — (1, C, H, W) in [0, 1]
            frame_tensor = agent.world_model.decode(current_embed)
            frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = np.clip(frame_np, 0.0, 1.0)
            frame_uint8 = (frame_np * 255).astype(np.uint8)

            # Scale up for display
            display = cv2.resize(frame_uint8, frame_size, interpolation=cv2.INTER_NEAREST)
            display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

            # Overlay step and cumulative reward
            cv2.putText(display_bgr, f"step {step+1:4d}  reward {total_reward:7.1f}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(display_bgr)
            cv2.imshow(window, display_bgr)
            if cv2.waitKey(max(1, 1000 // fps)) & 0xFF == ord("q"):
                print("Quit early.")
                break

            action_idx = agent.q_model(current_embed).argmax(dim=1)  # (1,)
            action_onehot = F.one_hot(action_idx, num_classes=n_actions).float()

            next_embed, reward, done = agent.world_model.imagine_step(current_embed, action_onehot)

            total_reward += reward.item()

            current_embed = next_embed

            if done.item() > 0.5:
                print(f"World model predicted done at step {step+1}.")
                break

    writer.release()
    cv2.destroyAllWindows()
    env.close()

    print(f"Done. Total imagined reward: {total_reward:.1f}")
    print(f"Video saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int,   default=500,               help="imagination steps")
    parser.add_argument("--warmup", type=int,   default=50,                help="real env warmup steps")
    parser.add_argument("--scale",  type=int,   default=4,                 help="display upscale factor")
    parser.add_argument("--out",    type=str,   default="imagination.mp4", help="video output path")
    parser.add_argument("--fps",    type=int,   default=50,                help="playback FPS (default: 50, matches CarRacing)")
    args = parser.parse_args()

    run_imagination(steps=args.steps, warmup=args.warmup, scale=args.scale, out=args.out, fps=args.fps)
