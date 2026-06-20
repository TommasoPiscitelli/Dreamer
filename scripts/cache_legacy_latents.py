from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dreamer_carracing.world_model.load_legacy import load_legacy_world_model


def preprocess_obs(obs_np, image_size: int, device):
    obs = torch.from_numpy(obs_np).float().to(device) / 255.0
    obs = obs.permute(0, 3, 1, 2).contiguous()

    if obs.shape[-1] != image_size or obs.shape[-2] != image_size:
        obs = F.interpolate(
            obs,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    return obs


@torch.no_grad()
def encode_all_obs(world_model, obs_np, image_size: int, device, batch_size: int):
    zs = []

    for start in range(0, len(obs_np), batch_size):
        obs_batch = obs_np[start : start + batch_size]
        obs = preprocess_obs(obs_batch, image_size=image_size, device=device)
        z = world_model.encode_obs(obs)
        zs.append(z.cpu())

    return torch.cat(zs, dim=0)


@torch.no_grad()
def cache_episode(world_model, path: Path, out_path: Path, image_size: int, device, encode_batch_size: int):
    data = np.load(path)

    obs_np = data["obs"]          # [T+1, 96, 96, 3]
    actions_np = data["action"]   # [T, 3]
    rewards_np = data["reward"]   # [T]

    T = actions_np.shape[0]

    z_all = encode_all_obs(
        world_model=world_model,
        obs_np=obs_np,
        image_size=image_size,
        device=device,
        batch_size=encode_batch_size,
    )  # [T+1, z_dim], CPU

    actions = torch.from_numpy(actions_np.astype(np.float32)).to(device)

    h = torch.zeros(world_model.num_layers, 1, world_model.h_dim, device=device)
    c = torch.zeros(world_model.num_layers, 1, world_model.h_dim, device=device)

    h_list = []

    for t in range(T):
        z_t = z_all[t : t + 1].to(device)
        a_t = actions[t : t + 1]

        _, _, _, h, c = world_model._mdn_step(
            z=z_t,
            action=a_t,
            h=h,
            c=c,
        )

        h_list.append(h[-1, 0].cpu())

    h_next = torch.stack(h_list, dim=0)       # [T, h_dim]
    z_next = z_all[1 : T + 1]                 # [T, z_dim]
    features = torch.cat([z_next, h_next], dim=-1)  # [T, z_dim + h_dim]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        features=features.numpy().astype(np.float32),
        z=z_all.numpy().astype(np.float32),
        h_next=h_next.numpy().astype(np.float32),
        action=actions_np.astype(np.float32),
        reward=rewards_np.astype(np.float32),
    )

    print(f"cached {path.name} -> {out_path} | T={T} | return={float(rewards_np.sum()):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw/train")
    parser.add_argument("--out-dir", type=str, default="data/latent/train")
    parser.add_argument("--vae-ckpt", type=str, default="checkpoints/legacy/vae/vae.pt")
    parser.add_argument("--mdn-rnn-ckpt", type=str, default="checkpoints/legacy/mdn_rnn/mdn_rnn.pt")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=None,
        device=device,
    )
    world_model.eval()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    for path in files:
        out_path = out_dir / path.name
        cache_episode(
            world_model=world_model,
            path=path,
            out_path=out_path,
            image_size=args.image_size,
            device=device,
            encode_batch_size=args.encode_batch_size,
        )


if __name__ == "__main__":
    main()
