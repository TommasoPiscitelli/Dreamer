from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dreamer_carracing.world_model.load_legacy import load_legacy_world_model


def preprocess_obs(obs_np, device):
    """
    obs_np: [B, H, W, 3] uint8
    returns: [B, 3, 64, 64] float
    """
    obs = torch.from_numpy(obs_np).to(device).float() / 255.0
    obs = obs.permute(0, 3, 1, 2).contiguous()

    if obs.shape[-2:] != (64, 64):
        obs = F.interpolate(obs, size=(64, 64), mode="bilinear", align_corners=False)

    return obs


@torch.no_grad()
def encode_all_obs(world_model, obs_np, device, batch_size):
    zs = []

    for start in range(0, len(obs_np), batch_size):
        batch = obs_np[start:start + batch_size]
        obs = preprocess_obs(batch, device)
        z = world_model.encode_obs(obs)
        zs.append(z.cpu())

    return torch.cat(zs, dim=0)


@torch.no_grad()
def cache_episode(world_model, in_path, out_path, device, batch_size):
    data = np.load(in_path)

    obs_np = data["obs"]
    actions_np = data["action"].astype(np.float32)
    rewards_np = data["reward"].astype(np.float32)

    T = actions_np.shape[0]

    z_all = encode_all_obs(world_model, obs_np, device, batch_size=batch_size)

    h = torch.zeros(1, 1, world_model.h_dim, device=device)
    c = torch.zeros(1, 1, world_model.h_dim, device=device)

    h_next_list = []
    c_next_list = []

    for t in range(T):
        z_t = z_all[t:t + 1].to(device)
        action_t = torch.from_numpy(actions_np[t:t + 1]).to(device)

        _, _, _, h, c = world_model._mdn_step(
            z=z_t,
            action=action_t,
            h=h,
            c=c,
        )

        h_next_list.append(h[-1, 0].detach().cpu())
        c_next_list.append(c[-1, 0].detach().cpu())

    h_next = torch.stack(h_next_list, dim=0).numpy().astype(np.float32)
    c_next = torch.stack(c_next_list, dim=0).numpy().astype(np.float32)

    z_np = z_all.numpy().astype(np.float32)
    z_next = z_np[1:T + 1]

    features = np.concatenate([z_next, h_next], axis=-1).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        features=features,
        z=z_np,
        h_next=h_next,
        c_next=c_next,
        action=actions_np,
        reward=rewards_np,
    )

    print(
        f"cached {in_path.name} -> {out_path} | "
        f"T={T} | return={rewards_np.sum():.2f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/raw/train")
    parser.add_argument("--out-dir", type=str, default="data/latent/train")
    parser.add_argument("--vae-ckpt", type=str, default="checkpoints/legacy/vae/vae.pt")
    parser.add_argument("--mdn-rnn-ckpt", type=str, default="checkpoints/legacy/mdn_rnn/mdn_rnn.pt")
    parser.add_argument("--reward-ckpt", type=str, default="checkpoints/reward_model/reward_model.pt")
    parser.add_argument("--reward-calibration", type=str, default="checkpoints/reward_model/reward_calibration_bias_train.json")
    parser.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    files = sorted(data_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    for path in files:
        out_path = out_dir / path.name
        cache_episode(
            world_model=world_model,
            in_path=path,
            out_path=out_path,
            device=device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
