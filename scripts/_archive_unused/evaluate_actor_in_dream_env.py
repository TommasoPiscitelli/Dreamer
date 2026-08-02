import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from world_model.api import LatentState
from world_model.load_legacy import load_legacy_world_model
from dreamer.actor_critic import Actor


def get_key(data, candidates):
    for k in candidates:
        if k in data:
            return k
    raise KeyError(f"None of these keys found: {candidates}. Available keys: {list(data.keys())}")


def preprocess_frames(obs_np, device):
    frames = []

    for obs in obs_np:
        if obs.ndim == 3 and obs.shape[0] == 3:
            obs = np.transpose(obs, (1, 2, 0))

        if obs.dtype != np.uint8:
            x = obs.astype(np.float32)
            if x.max() <= 1.0:
                x = x * 255.0
            obs = np.clip(x, 0, 255).astype(np.uint8)

        img = Image.fromarray(obs).resize((64, 64), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        frames.append(arr)

    frames = np.stack(frames, axis=0)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)
    return tensor


def get_vae(world_model):
    for name in ["vae", "_vae", "v_model", "V"]:
        if hasattr(world_model, name):
            return getattr(world_model, name)
    raise AttributeError("Could not find VAE inside world_model.")


@torch.no_grad()
def decode_z(world_model, z):
    vae = get_vae(world_model)

    if hasattr(vae, "decode"):
        out = vae.decode(z)
    elif hasattr(vae, "decoder"):
        out = vae.decoder(z)
    else:
        raise AttributeError("VAE has neither decode() nor decoder().")

    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.min().item() < -0.05 or out.max().item() > 1.05:
        out = torch.sigmoid(out)

    return out.clamp(0.0, 1.0)


@torch.no_grad()
def tensor_to_uint8(x):
    x = x.detach().cpu().clamp(0, 1)
    x = x.permute(0, 2, 3, 1).numpy()
    return (x * 255).astype(np.uint8)


def load_actor(actor_ckpt, device, feature_dim=288):
    actor = Actor(feature_dim=feature_dim).to(device)

    ckpt = torch.load(actor_ckpt, map_location=device)

    if isinstance(ckpt, dict):
        if "actor_state_dict" in ckpt:
            state_dict = ckpt["actor_state_dict"]
        elif "actor" in ckpt:
            state_dict = ckpt["actor"]
        elif "actor_state" in ckpt:
            state_dict = ckpt["actor_state"]
        else:
            state_dict = ckpt
    else:
        raise RuntimeError("Unexpected actor checkpoint format.")

    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episode-file", type=str, required=True)
    parser.add_argument("--actor-ckpt", type=str, required=True)

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="logs/dream_env_actor_eval")
    parser.add_argument("--fps", type=int, default=12)

    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("episode:", args.episode_file)

    data = np.load(args.episode_file)
    obs_key = get_key(data, ["obs", "observations", "frames"])
    action_key = get_key(data, ["action", "actions", "acts"])

    obs = data[obs_key]
    actions = data[action_key].astype(np.float32)

    if args.start >= len(obs) - 1:
        raise ValueError("start is too large for this episode.")

    obs_prefix = obs[: args.start + 1]
    action_prefix = actions[: args.start]

    real_frames = preprocess_frames(obs_prefix, device)

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )
    world_model.eval()

    actor = load_actor(args.actor_ckpt, device=device, feature_dim=288)

    z_real = world_model.encode_obs(real_frames)

    # Warm-up of h,c using a real trajectory prefix and real actions.
    state = world_model.initial_state(batch_size=1, device=device)

    for t in range(args.start):
        action_t = torch.from_numpy(action_prefix[t]).float().to(device).unsqueeze(0)

        state_in = LatentState(
            z=z_real[t : t + 1],
            h=state.h,
            c=state.c,
        )

        imagined = world_model.imagine_step(state_in, action_t)

        # Teacher forcing on z, but recurrent state h,c comes from the transition model.
        state = LatentState(
            z=z_real[t + 1 : t + 2],
            h=imagined.next_state.h,
            c=imagined.next_state.c,
        )

    # Start pure dream rollout from a real latent state.
    state = LatentState(
        z=z_real[args.start : args.start + 1],
        h=state.h,
        c=state.c,
    )

    zs = [state.z]
    rewards = []
    actions_actor = []

    for t in range(args.horizon):
        action_t, entropy_t = actor.sample(state.features, deterministic=True)

        imagined = world_model.imagine_step(state, action_t)

        rewards.append(float(imagined.reward.squeeze().cpu()))
        actions_actor.append(action_t.squeeze(0).cpu().numpy())

        state = imagined.next_state
        zs.append(state.z)

    z_rollout = torch.cat(zs, dim=0)
    decoded = decode_z(world_model, z_rollout)
    decoded_u8 = tensor_to_uint8(decoded)

    gif_path = out_dir / f"dream_actor_start_{args.start}_horizon_{args.horizon}.gif"
    imageio.mimsave(gif_path, list(decoded_u8), fps=args.fps)

    actions_actor = np.stack(actions_actor, axis=0)
    rewards = np.asarray(rewards, dtype=np.float32)

    df = pd.DataFrame({
        "t": np.arange(args.horizon),
        "predicted_reward": rewards,
        "cumulative_predicted_reward": np.cumsum(rewards),
        "steer": actions_actor[:, 0],
        "gas": actions_actor[:, 1],
        "brake": actions_actor[:, 2],
    })

    csv_path = out_dir / f"dream_actor_start_{args.start}_horizon_{args.horizon}.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(df["t"], df["predicted_reward"], label="Predicted reward")
    plt.plot(df["t"], df["cumulative_predicted_reward"], label="Cumulative predicted reward")
    plt.xlabel("Dream time step")
    plt.ylabel("Reward")
    plt.title("Actor rollout in dream environment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    reward_plot = out_dir / f"dream_actor_rewards_start_{args.start}_horizon_{args.horizon}.png"
    plt.savefig(reward_plot, dpi=200)

    plt.figure(figsize=(9, 5))
    plt.plot(df["t"], df["steer"], label="steer")
    plt.plot(df["t"], df["gas"], label="gas")
    plt.plot(df["t"], df["brake"], label="brake")
    plt.xlabel("Dream time step")
    plt.ylabel("Action")
    plt.title("Actor actions in dream environment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    action_plot = out_dir / f"dream_actor_actions_start_{args.start}_horizon_{args.horizon}.png"
    plt.savefig(action_plot, dpi=200)

    selected = [0, 1, 2, 5, 10, 15, 20, 30, 50, 100, 200, args.horizon]
    selected = [t for t in selected if t <= args.horizon]
    selected = sorted(set(selected))

    fig, axes = plt.subplots(len(selected), 1, figsize=(4, 2.8 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax, t in zip(axes, selected):
        ax.imshow(decoded_u8[t])
        ax.axis("off")
        ax.set_title(f"Dream frame t={t}")

    grid_path = out_dir / f"dream_actor_grid_start_{args.start}_horizon_{args.horizon}.png"
    plt.tight_layout()
    plt.savefig(grid_path, dpi=200)

    print()
    print("Saved:")
    print(" ", gif_path)
    print(" ", csv_path)
    print(" ", reward_plot)
    print(" ", action_plot)
    print(" ", grid_path)
    print()
    print("Dream rollout summary")
    print("---------------------")
    print(f"horizon:               {args.horizon}")
    print(f"mean predicted reward: {rewards.mean():.4f}")
    print(f"sum predicted reward:  {rewards.sum():.4f}")
    print(f"min predicted reward:  {rewards.min():.4f}")
    print(f"max predicted reward:  {rewards.max():.4f}")
    print()
    print(df.tail())


if __name__ == "__main__":
    main()