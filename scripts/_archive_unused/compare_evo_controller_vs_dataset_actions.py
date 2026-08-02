import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dreamer.evo_controller import EvoController


def get_key(data, candidates):
    for k in candidates:
        if k in data:
            return k
    raise KeyError(f"None of {candidates} found. Available keys: {list(data.keys())}")


def sample_states_and_actions(files, num_samples, seed, device):
    rng = np.random.default_rng(seed)
    zs, hs, dataset_actions = [], [], []

    while len(zs) < num_samples:
        f = files[int(rng.integers(0, len(files)))]

        try:
            data = np.load(f)
        except Exception:
            continue

        z_key = get_key(data, ["z", "latents", "latent"])
        h_key = get_key(data, ["h_next", "h", "hidden", "hidden_next"])
        a_key = get_key(data, ["action", "actions", "a"])

        z = data[z_key].astype(np.float32)
        h = data[h_key].astype(np.float32)
        a = data[a_key].astype(np.float32)

        T = min(len(h), len(a), len(z) - 1)

        if T <= 2:
            continue

        t = int(rng.integers(1, T))

        z_t = z[t]
        h_t = h[t - 1]

        if h_t.ndim == 2:
            h_t = h_t[0]

        zs.append(z_t)
        hs.append(h_t)
        dataset_actions.append(a[t])

    z = torch.tensor(np.stack(zs), dtype=torch.float32, device=device)
    h = torch.tensor(np.stack(hs), dtype=torch.float32, device=device)
    dataset_actions = np.stack(dataset_actions).astype(np.float32)

    return z, h, dataset_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-json", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--dataset-pattern", type=str, default="*heuristic*.npz")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.dataset_dir).rglob(args.dataset_pattern))

    if len(files) == 0:
        raise RuntimeError(f"No files found in {args.dataset_dir} with pattern {args.dataset_pattern}")

    controller = EvoController.from_json(args.controller_json).to(device)
    controller.eval()

    z, h, dataset_actions = sample_states_and_actions(
        files=files,
        num_samples=args.num_samples,
        seed=args.seed,
        device=device,
    )

    with torch.no_grad():
        features = torch.cat([z, h], dim=-1)
        evo_actions = controller(features).cpu().numpy()

    df = pd.DataFrame({
        "dataset_steer": dataset_actions[:, 0],
        "dataset_gas": dataset_actions[:, 1],
        "dataset_brake": dataset_actions[:, 2],
        "evo_steer": evo_actions[:, 0],
        "evo_gas": evo_actions[:, 1],
        "evo_brake": evo_actions[:, 2],
    })

    df.to_csv(out_dir / "evo_vs_dataset_actions.csv", index=False)

    print("\nSummary:")
    print(df.describe())

    for name in ["steer", "gas", "brake"]:
        plt.figure(figsize=(8, 5))
        plt.hist(df[f"dataset_{name}"], bins=60, alpha=0.6, label="dataset heuristic")
        plt.hist(df[f"evo_{name}"], bins=60, alpha=0.6, label="evo controller")
        plt.xlabel(name)
        plt.ylabel("count")
        plt.title(f"Action distribution: {name}")
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"hist_{name}.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print("saved:", path)

    plt.figure(figsize=(6, 6))
    plt.scatter(df["evo_steer"], df["evo_gas"], s=4, alpha=0.25)
    plt.xlabel("evo steer")
    plt.ylabel("evo gas")
    plt.title("Evo controller: steer vs gas")
    plt.tight_layout()
    path = out_dir / "scatter_evo_steer_gas.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print("saved:", path)

    plt.figure(figsize=(6, 6))
    plt.scatter(df["evo_steer"], df["evo_brake"], s=4, alpha=0.25)
    plt.xlabel("evo steer")
    plt.ylabel("evo brake")
    plt.title("Evo controller: steer vs brake")
    plt.tight_layout()
    path = out_dir / "scatter_evo_steer_brake.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print("saved:", path)


if __name__ == "__main__":
    main()
