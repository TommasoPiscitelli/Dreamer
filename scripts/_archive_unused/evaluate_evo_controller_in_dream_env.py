import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

from world_model.api import LatentState
from world_model.load_legacy import load_legacy_world_model
from dreamer.evo_controller import EvoController


def get_key(data, candidates):
    for k in candidates:
        if k in data:
            return k
    raise KeyError(f"None of {candidates} found. Available keys: {list(data.keys())}")


def features_from_state(state):
    return torch.cat([state.z, state.h[-1]], dim=-1)


def extract_next_state(out):
    for attr in ["next_state", "state", "latent_state"]:
        if hasattr(out, attr):
            return getattr(out, attr)
    if isinstance(out, LatentState):
        return out
    raise TypeError(f"Cannot extract next state from object of type {type(out)}")


def predict_reward(world_model, state):
    features = features_from_state(state)

    for name in ["predict_reward", "reward", "reward_model", "_reward_model", "rm"]:
        if hasattr(world_model, name):
            obj = getattr(world_model, name)

            try:
                r = obj(state)
            except Exception:
                r = obj(features)

            if isinstance(r, tuple):
                r = r[0]

            return r.view(-1)

    raise AttributeError("Could not find reward predictor in world_model")


def decode_z(world_model, z):
    candidates = [
        ("decode_obs", world_model),
        ("decode", world_model),
    ]

    if hasattr(world_model, "vae"):
        candidates.extend([
            ("decode", world_model.vae),
            ("decoder", world_model.vae),
        ])

    last_error = None

    for name, obj in candidates:
        if not hasattr(obj, name):
            continue

        fn = getattr(obj, name)

        try:
            x = fn(z)
        except Exception as e:
            last_error = e
            continue

        if isinstance(x, tuple):
            x = x[0]

        return x

    raise RuntimeError(f"Could not decode z. Last error: {last_error}")


def tensor_to_frame(x):
    x = x.detach().cpu()

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 3 and x.shape[0] in [1, 3]:
        x = x.permute(1, 2, 0)

    x = x.numpy()

    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    x = np.nan_to_num(x)

    if x.min() < 0:
        x = (x + 1.0) / 2.0

    if x.max() <= 1.5:
        x = x * 255.0

    x = np.clip(x, 0, 255).astype(np.uint8)

    return x


def load_start_state(latent_dir, pattern, seed, device):
    rng = np.random.default_rng(seed)
    files = sorted(Path(latent_dir).rglob(pattern))

    if len(files) == 0:
        raise RuntimeError(f"No files found in {latent_dir} with pattern {pattern}")

    while True:
        f = files[int(rng.integers(0, len(files)))]

        try:
            data = np.load(f)
        except Exception:
            continue

        z_key = get_key(data, ["z", "latents", "latent"])
        h_key = get_key(data, ["h_next", "h", "hidden", "hidden_next"])
        c_key = get_key(data, ["c_next", "c", "cell", "cell_next"])

        z = data[z_key].astype(np.float32)
        h = data[h_key].astype(np.float32)
        c = data[c_key].astype(np.float32)

        T = min(len(h), len(c), len(z) - 1)

        if T <= 2:
            continue

        t = int(rng.integers(1, T))

        z_t = z[t]
        h_t = h[t - 1]
        c_t = c[t - 1]

        if h_t.ndim == 2:
            h_t = h_t[0]
        if c_t.ndim == 2:
            c_t = c_t[0]

        z_t = torch.tensor(z_t, dtype=torch.float32, device=device).unsqueeze(0)
        h_t = torch.tensor(h_t, dtype=torch.float32, device=device).view(1, 1, -1)
        c_t = torch.tensor(c_t, dtype=torch.float32, device=device).view(1, 1, -1)

        return LatentState(z=z_t, h=h_t, c=c_t, extra={}), str(f), t


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--controller-json", type=str, required=True)
    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--latent-pattern", type=str, default="*.npz")

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
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

    state, source_file, source_t = load_start_state(
        latent_dir=args.latent_dir,
        pattern=args.latent_pattern,
        seed=args.seed,
        device=device,
    )

    frames = []
    rows = []

    total_reward = 0.0

    with torch.no_grad():
        for t in range(args.horizon):
            frame = tensor_to_frame(decode_z(world_model, state.z))
            frames.append(frame)

            features = features_from_state(state)
            action = controller(features)

            out = world_model.imagine_step(state, action)
            next_state = extract_next_state(out)

            reward = predict_reward(world_model, next_state)
            r = float(reward[0].cpu())
            total_reward += r

            rows.append({
                "t": t,
                "reward": r,
                "cumulative_reward": total_reward,
                "steer": float(action[0, 0].cpu()),
                "gas": float(action[0, 1].cpu()),
                "brake": float(action[0, 2].cpu()),
                "source_file": source_file,
                "source_t": source_t,
            })

            state = next_state

    gif_path = out_dir / "evo_dream_rollout.gif"
    csv_path = out_dir / "evo_dream_rollout.csv"

    imageio.mimsave(gif_path, frames, fps=args.fps)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("Saved GIF:", gif_path)
    print("Saved CSV:", csv_path)
    print("Undiscounted imagined return:", total_reward)


if __name__ == "__main__":
    main()
