import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import gymnasium as gym
except ImportError:
    import gym

from world_model.api import LatentState
from world_model.load_legacy import load_legacy_world_model
from dreamer.evo_controller import EvoController


def features_from_state(state):
    return torch.cat([state.z, state.h[-1]], dim=-1)


def extract_next_state(out):
    for attr in ["next_state", "state", "latent_state"]:
        if hasattr(out, attr):
            return getattr(out, attr)
    if isinstance(out, LatentState):
        return out
    raise TypeError(f"Cannot extract next state from object of type {type(out)}")


def encode_obs(world_model, obs, device):
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)

    if x.max() > 1.0:
        x = x / 255.0

    if x.ndim == 3:
        x = x.permute(2, 0, 1).unsqueeze(0)

    if x.shape[-2:] != (64, 64):
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

    with torch.no_grad():
        if hasattr(world_model, "encode_obs"):
            z = world_model.encode_obs(x)
        elif hasattr(world_model, "vae"):
            out = world_model.vae.encode(x)
            if isinstance(out, tuple):
                z = out[0]
            else:
                z = out
        else:
            raise AttributeError("World model has neither encode_obs nor vae.encode")

    if isinstance(z, tuple):
        z = z[0]

    return z.float()


def make_initial_state(world_model, z, device):
    if hasattr(world_model, "initial_state"):
        try:
            s = world_model.initial_state(batch_size=z.shape[0], device=device)
        except TypeError:
            try:
                s = world_model.initial_state(z.shape[0])
            except TypeError:
                s = world_model.initial_state()

        return LatentState(z=z, h=s.h.to(device), c=s.c.to(device), extra={})

    h = torch.zeros(1, z.shape[0], 256, device=device)
    c = torch.zeros(1, z.shape[0], 256, device=device)
    return LatentState(z=z, h=h, c=c, extra={})


def reset_env(env, seed):
    out = env.reset(seed=seed)

    if isinstance(out, tuple):
        obs, info = out
    else:
        obs, info = out, {}

    return obs, info


def step_env(env, action):
    out = env.step(action)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out

    return obs, reward, done, info


def obs_to_uint8(obs):
    obs = np.asarray(obs)

    if obs.dtype == np.uint8:
        return obs

    if obs.max() <= 1.0:
        obs = obs * 255.0

    return np.clip(obs, 0, 255).astype(np.uint8)

def get_frame(env, obs):
    try:
        frame = env.render()
        if frame is not None:
            return frame
    except Exception:
        pass

    return obs


def normalize_frame(frame):
    frame = np.asarray(frame)

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)

    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    return frame

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--controller-json", type=str, required=True)

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, required=True)

    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )
    world_model.eval()

    controller = EvoController.from_json(args.controller_json).to(device)
    controller.eval()

    try:
        env = gym.make(args.env_id, continuous=True, render_mode="rgb_array")
    except Exception:
        env = gym.make("CarRacing-v2", continuous=True)

    episode_returns = []

    for ep in range(args.episodes):
        obs, _ = reset_env(env, seed=args.seed + ep)

        z = encode_obs(world_model, obs, device=device)
        state = make_initial_state(world_model, z, device=device)

        zs = [z.squeeze(0).detach().cpu().numpy()]
        h_next = []
        c_next = []
        actions = []
        rewards = []
        obs_frames = [obs_to_uint8(obs)]
        render_frames = [normalize_frame(get_frame(env, obs))]

        total_reward = 0.0

        for t in range(args.max_steps):
            with torch.no_grad():
                feat = features_from_state(state)
                action = controller(feat)[0].detach().cpu().numpy().astype(np.float32)

            next_obs, reward, done, info = step_env(env, action)
            obs_frames.append(obs_to_uint8(next_obs))
            render_frames.append(normalize_frame(get_frame(env, next_obs)))

            with torch.no_grad():
                out = world_model.imagine_step(state, torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0))
                next_state_model = extract_next_state(out)

                z_next_real = encode_obs(world_model, next_obs, device=device)

                next_state = LatentState(
                    z=z_next_real,
                    h=next_state_model.h.detach(),
                    c=next_state_model.c.detach(),
                    extra={},
                )

            actions.append(action)
            rewards.append(float(reward))
            h_next.append(next_state.h.squeeze(1).detach().cpu().numpy())
            c_next.append(next_state.c.squeeze(1).detach().cpu().numpy())
            zs.append(z_next_real.squeeze(0).detach().cpu().numpy())

            total_reward += float(reward)
            state = next_state

            if done:
                break

        path = out_dir / f"evo_controller_episode_{ep:05d}.npz"

        np.savez_compressed(
            path,
            obs=np.asarray(obs_frames, dtype=np.uint8),
            render_obs=np.asarray(render_frames, dtype=np.uint8),
            z=np.asarray(zs, dtype=np.float32),
            h_next=np.asarray(h_next, dtype=np.float32),
            c_next=np.asarray(c_next, dtype=np.float32),
            action=np.asarray(actions, dtype=np.float32),
            reward=np.asarray(rewards, dtype=np.float32),
            episode_return=np.asarray(total_reward, dtype=np.float32),
        )

        episode_returns.append(total_reward)

        print(
            f"episode {ep:04d} | steps={len(rewards)} | return={total_reward:.2f} | saved={path}",
            flush=True,
        )

    env.close()

    print("\nReturn summary:")
    print("episodes:", len(episode_returns))
    print("mean:", float(np.mean(episode_returns)))
    print("std:", float(np.std(episode_returns)))
    print("min:", float(np.min(episode_returns)))
    print("max:", float(np.max(episode_returns)))


if __name__ == "__main__":
    main()
