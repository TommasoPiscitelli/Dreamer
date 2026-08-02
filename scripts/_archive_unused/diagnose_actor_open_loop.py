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

    checkpoint = torch.load(actor_ckpt, map_location=device)

    if isinstance(checkpoint, dict):
        if "actor_state_dict" in checkpoint:
            state_dict = checkpoint["actor_state_dict"]
        elif "actor" in checkpoint:
            state_dict = checkpoint["actor"]
        elif "actor_state" in checkpoint:
            state_dict = checkpoint["actor_state"]
        else:
            state_dict = checkpoint
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

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="logs/actor_open_loop_diagnostics")
    parser.add_argument("--fps", type=int, default=8)

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

    max_horizon = min(args.horizon, len(obs) - args.start - 1, len(actions) - args.start)
    if max_horizon <= 0:
        raise ValueError("Invalid start/horizon for this episode.")

    obs_seq = obs[args.start : args.start + max_horizon + 1]
    action_seq = actions[args.start : args.start + max_horizon]

    real_frames = preprocess_frames(obs_seq, device)

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )
    world_model.eval()

    actor = load_actor(args.actor_ckpt, device=device, feature_dim=288)

    # Encode real frames
    z_real = world_model.encode_obs(real_frames)

    # Static VAE recon
    recon_frames = decode_z(world_model, z_real)

    # Open-loop with dataset actions
    z_dataset = [z_real[0:1]]
    state_dataset = world_model.initial_state(batch_size=1, device=device)
    state_dataset = LatentState(z=z_real[0:1], h=state_dataset.h, c=state_dataset.c)

    dataset_actions_list = []

    for t in range(max_horizon):
        action_t = torch.from_numpy(action_seq[t]).float().to(device).unsqueeze(0)
        dataset_actions_list.append(action_t.squeeze(0).cpu().numpy())

        imagined = world_model.imagine_step(state_dataset, action_t)
        state_dataset = imagined.next_state
        z_dataset.append(state_dataset.z)

    z_dataset = torch.cat(z_dataset, dim=0)
    dataset_frames = decode_z(world_model, z_dataset)

    # Open-loop with actor actions
    z_actor = [z_real[0:1]]
    state_actor = world_model.initial_state(batch_size=1, device=device)
    state_actor = LatentState(z=z_real[0:1], h=state_actor.h, c=state_actor.c)

    actor_actions_list = []

    for t in range(max_horizon):
        action_t, entropy_t = actor.sample(state_actor.features, deterministic=True)
        actor_actions_list.append(action_t.squeeze(0).cpu().numpy())

        imagined = world_model.imagine_step(state_actor, action_t)
        state_actor = imagined.next_state
        z_actor.append(state_actor.z)

    z_actor = torch.cat(z_actor, dim=0)
    actor_frames = decode_z(world_model, z_actor)

    real_u8 = tensor_to_uint8(real_frames)
    recon_u8 = tensor_to_uint8(recon_frames)
    dataset_u8 = tensor_to_uint8(dataset_frames)
    actor_u8 = tensor_to_uint8(actor_frames)

    # Save GIF
    gif_frames = []
    for t in range(max_horizon + 1):
        panel = np.concatenate(
            [real_u8[t], recon_u8[t], dataset_u8[t], actor_u8[t]],
            axis=1,
        )
        gif_frames.append(panel)

    gif_path = out_dir / "real_recon_dataset_actor_openloop.gif"
    imageio.mimsave(gif_path, gif_frames, fps=args.fps)

    # Save action comparison CSV
    dataset_actions_arr = np.stack(dataset_actions_list, axis=0)
    actor_actions_arr = np.stack(actor_actions_list, axis=0)

    df_actions = pd.DataFrame({
        "t": np.arange(max_horizon),
        "steer_dataset": dataset_actions_arr[:, 0],
        "gas_dataset": dataset_actions_arr[:, 1],
        "brake_dataset": dataset_actions_arr[:, 2],
        "steer_actor": actor_actions_arr[:, 0],
        "gas_actor": actor_actions_arr[:, 1],
        "brake_actor": actor_actions_arr[:, 2],
    })
    actions_csv = out_dir / "action_comparison.csv"
    df_actions.to_csv(actions_csv, index=False)

    # MSE vs real (strictly meaningful for dataset actions, less so for actor actions)
    real_float = real_frames.detach().cpu()
    recon_float = recon_frames.detach().cpu()
    dataset_float = dataset_frames.detach().cpu()
    actor_float = actor_frames.detach().cpu()

    def mse_per_t(x):
        return ((x - real_float) ** 2).mean(dim=(1, 2, 3)).numpy()

    recon_mse = mse_per_t(recon_float)
    dataset_mse = mse_per_t(dataset_float)
    actor_mse = mse_per_t(actor_float)

    metrics = pd.DataFrame({
        "t": np.arange(max_horizon + 1),
        "vae_reconstruction_mse": recon_mse,
        "dataset_openloop_mse": dataset_mse,
        "actor_openloop_mse": actor_mse,
    })

    csv_path = out_dir / "diagnostic_mse.csv"
    metrics.to_csv(csv_path, index=False)

    # MSE plot
    plt.figure(figsize=(9, 5))
    plt.plot(metrics["t"], metrics["vae_reconstruction_mse"], label="VAE reconstruction")
    plt.plot(metrics["t"], metrics["dataset_openloop_mse"], label="Open-loop dataset actions")
    plt.plot(metrics["t"], metrics["actor_openloop_mse"], label="Open-loop actor actions")
    plt.xlabel("Time step")
    plt.ylabel("Pixel MSE")
    plt.title("World model diagnostic")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = out_dir / "diagnostic_mse.png"
    plt.savefig(plot_path, dpi=200)

    # Static grid
    selected = [0, 1, 2, 5, 10, 20, 50, max_horizon]
    selected = [t for t in selected if t <= max_horizon]
    selected = sorted(set(selected))

    fig, axes = plt.subplots(len(selected), 4, figsize=(10, 2.3 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]

    col_titles = [
        "Real",
        "VAE recon",
        "Open-loop (dataset actions)",
        "Open-loop (actor actions)",
    ]

    for row, t in enumerate(selected):
        imgs = [real_u8[t], recon_u8[t], dataset_u8[t], actor_u8[t]]
        for col in range(4):
            axes[row, col].imshow(imgs[col])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col])
            if col == 0:
                axes[row, col].set_ylabel(f"t={t}", rotation=0, labelpad=25)

    grid_path = out_dir / "diagnostic_grid.png"
    plt.tight_layout()
    plt.savefig(grid_path, dpi=200)

    # Action plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_actions["t"], df_actions["steer_dataset"], label="steer dataset")
    plt.plot(df_actions["t"], df_actions["steer_actor"], label="steer actor")
    plt.plot(df_actions["t"], df_actions["gas_dataset"], label="gas dataset")
    plt.plot(df_actions["t"], df_actions["gas_actor"], label="gas actor")
    plt.plot(df_actions["t"], df_actions["brake_dataset"], label="brake dataset")
    plt.plot(df_actions["t"], df_actions["brake_actor"], label="brake actor")
    plt.xlabel("Time step")
    plt.ylabel("Action value")
    plt.title("Dataset actions vs actor actions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    action_plot = out_dir / "action_comparison.png"
    plt.savefig(action_plot, dpi=200)

    print()
    print("Saved:")
    print(" ", gif_path)
    print(" ", grid_path)
    print(" ", plot_path)
    print(" ", csv_path)
    print(" ", actions_csv)
    print(" ", action_plot)
    print()
    print(metrics.tail())
    print()
    print(df_actions.head())


if __name__ == "__main__":
    main()