import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dreamer.actor_critic import Actor


def get_key(data, candidates):
    for k in candidates:
        if k in data:
            return k
    raise KeyError(f"None of these keys found: {candidates}. Available keys: {list(data.keys())}")


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


def load_dataset_actions(raw_dir, pattern, max_actions, seed):
    rng = np.random.default_rng(seed)
    files = sorted(Path(raw_dir).rglob(pattern))

    if len(files) == 0:
        raise RuntimeError(f"No raw .npz files found in {raw_dir} with pattern {pattern}")

    actions_list = []

    for f in files:
        data = np.load(f)
        action_key = get_key(data, ["action", "actions", "acts"])
        actions = data[action_key].astype(np.float32)

        if actions.ndim != 2 or actions.shape[1] != 3:
            raise ValueError(f"Invalid action shape in {f}: {actions.shape}")

        actions_list.append(actions)

    actions = np.concatenate(actions_list, axis=0)

    if max_actions is not None and len(actions) > max_actions:
        idx = rng.choice(len(actions), size=max_actions, replace=False)
        actions = actions[idx]

    return actions, files


def load_actor_features_from_latents(latent_dir, pattern, max_states, seed):
    rng = np.random.default_rng(seed)
    files = sorted(Path(latent_dir).rglob(pattern))

    if len(files) == 0:
        raise RuntimeError(f"No latent .npz files found in {latent_dir} with pattern {pattern}")

    features_list = []

    for f in files:
        data = np.load(f)

        z_key = get_key(data, ["z", "latents", "latent"])
        h_key = get_key(data, ["h_next", "h", "hidden", "hidden_next"])

        z = data[z_key].astype(np.float32)
        h = data[h_key].astype(np.float32)

        # Expected cached format:
        # z:      [T+1, 32]
        # h_next: [T, 256]
        #
        # State at time t >= 1:
        # z_t = z[t]
        # h_t = h_next[t-1]
        T = min(len(h), len(z) - 1)

        if T <= 1:
            continue

        z_t = z[1 : T + 1]
        h_t = h[0:T]

        feats = np.concatenate([z_t, h_t], axis=-1)

        if feats.shape[1] != 288:
            raise ValueError(f"Expected feature dim 288, got {feats.shape} in {f}")

        features_list.append(feats)

    features = np.concatenate(features_list, axis=0)

    if max_states is not None and len(features) > max_states:
        idx = rng.choice(len(features), size=max_states, replace=False)
        features = features[idx]

    return features, files


@torch.no_grad()
def compute_actor_actions(actor, features, device, batch_size, deterministic):
    actions = []

    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).float().to(device)
        action, entropy = actor.sample(batch, deterministic=deterministic)
        actions.append(action.cpu().numpy())

    return np.concatenate(actions, axis=0)


def summarize_actions(name, actions):
    dims = ["steer", "gas", "brake"]
    rows = []

    for i, dim in enumerate(dims):
        x = actions[:, i]

        row = {
            "source": name,
            "dimension": dim,
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "p01": float(np.percentile(x, 1)),
            "p05": float(np.percentile(x, 5)),
            "p25": float(np.percentile(x, 25)),
            "p50": float(np.percentile(x, 50)),
            "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x)),
        }

        rows.append(row)

    extra = {
        "source": name,
        "steer_abs_gt_0_9": float(np.mean(np.abs(actions[:, 0]) > 0.9)),
        "gas_lt_0_05": float(np.mean(actions[:, 1] < 0.05)),
        "gas_gt_0_95": float(np.mean(actions[:, 1] > 0.95)),
        "brake_lt_0_05": float(np.mean(actions[:, 2] < 0.05)),
        "brake_gt_0_5": float(np.mean(actions[:, 2] > 0.5)),
        "brake_gt_0_95": float(np.mean(actions[:, 2] > 0.95)),
    }

    return rows, extra


def make_plots(dataset_actions, actor_actions, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = ["steer", "gas", "brake"]
    ranges = [(-1.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    for i, ax in enumerate(axes):
        bins = np.linspace(ranges[i][0], ranges[i][1], 80)

        ax.hist(dataset_actions[:, i], bins=bins, alpha=0.55, density=True, label="Dataset actions")
        ax.hist(actor_actions[:, i], bins=bins, alpha=0.55, density=True, label="Actor actions")

        ax.set_title(f"{dims[i]} distribution")
        ax.set_xlabel(dims[i])
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "action_histograms_dataset_vs_actor.png", dpi=200)
    plt.close()

    # Scatter steer-gas
    n = min(5000, len(dataset_actions), len(actor_actions))
    rng = np.random.default_rng(0)
    d_idx = rng.choice(len(dataset_actions), size=n, replace=False)
    a_idx = rng.choice(len(actor_actions), size=n, replace=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(dataset_actions[d_idx, 0], dataset_actions[d_idx, 1], s=4, alpha=0.25, label="Dataset")
    plt.scatter(actor_actions[a_idx, 0], actor_actions[a_idx, 1], s=4, alpha=0.25, label="Actor")
    plt.xlabel("Steer")
    plt.ylabel("Gas")
    plt.title("Steer vs gas")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_steer_gas_dataset_vs_actor.png", dpi=200)
    plt.close()

    # Scatter steer-brake
    plt.figure(figsize=(7, 6))
    plt.scatter(dataset_actions[d_idx, 0], dataset_actions[d_idx, 2], s=4, alpha=0.25, label="Dataset")
    plt.scatter(actor_actions[a_idx, 0], actor_actions[a_idx, 2], s=4, alpha=0.25, label="Actor")
    plt.xlabel("Steer")
    plt.ylabel("Brake")
    plt.title("Steer vs brake")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_steer_brake_dataset_vs_actor.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--actor-ckpt", type=str, required=True)

    parser.add_argument("--raw-pattern", type=str, default="*.npz")
    parser.add_argument("--latent-pattern", type=str, default="*.npz")

    parser.add_argument("--max-actions", type=int, default=100000)
    parser.add_argument("--max-states", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="logs/action_distribution_diagnostics")

    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    deterministic = not args.stochastic

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("deterministic actor:", deterministic)

    dataset_actions, raw_files = load_dataset_actions(
        raw_dir=args.raw_dir,
        pattern=args.raw_pattern,
        max_actions=args.max_actions,
        seed=args.seed,
    )

    print("raw files:", len(raw_files))
    print("dataset actions:", dataset_actions.shape)

    features, latent_files = load_actor_features_from_latents(
        latent_dir=args.latent_dir,
        pattern=args.latent_pattern,
        max_states=args.max_states,
        seed=args.seed,
    )

    print("latent files:", len(latent_files))
    print("actor states:", features.shape)

    actor = load_actor(args.actor_ckpt, device=device, feature_dim=288)

    actor_actions = compute_actor_actions(
        actor=actor,
        features=features,
        device=device,
        batch_size=args.batch_size,
        deterministic=deterministic,
    )

    print("actor actions:", actor_actions.shape)

    # Save raw samples
    pd.DataFrame(dataset_actions, columns=["steer", "gas", "brake"]).to_csv(
        out_dir / "dataset_actions_sample.csv", index=False
    )
    pd.DataFrame(actor_actions, columns=["steer", "gas", "brake"]).to_csv(
        out_dir / "actor_actions_sample.csv", index=False
    )

    # Summary
    rows_dataset, extra_dataset = summarize_actions("dataset", dataset_actions)
    rows_actor, extra_actor = summarize_actions("actor", actor_actions)

    summary_df = pd.DataFrame(rows_dataset + rows_actor)
    summary_df.to_csv(out_dir / "action_summary_by_dimension.csv", index=False)

    extra_df = pd.DataFrame([extra_dataset, extra_actor])
    extra_df.to_csv(out_dir / "action_boundary_summary.csv", index=False)

    make_plots(dataset_actions, actor_actions, out_dir)

    print()
    print("Saved:")
    print(" ", out_dir / "action_histograms_dataset_vs_actor.png")
    print(" ", out_dir / "scatter_steer_gas_dataset_vs_actor.png")
    print(" ", out_dir / "scatter_steer_brake_dataset_vs_actor.png")
    print(" ", out_dir / "action_summary_by_dimension.csv")
    print(" ", out_dir / "action_boundary_summary.csv")
    print()
    print("Summary by dimension:")
    print(summary_df)
    print()
    print("Boundary summary:")
    print(extra_df)


if __name__ == "__main__":
    main()