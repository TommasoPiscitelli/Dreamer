import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from world_model.api import LatentState
from world_model.model import load_legacy_world_model
from dreamer.actor import Actor, Value
from dreamer.imagination import imagine_rollout


def load_actor_value(ckpt_path, device, feature_dim=288):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    actor = Actor(feature_dim=feature_dim).to(device)
    value = Value(feature_dim=feature_dim).to(device)

    actor.load_state_dict(ckpt["actor_state_dict"])
    value.load_state_dict(ckpt["value_state_dict"])

    actor.eval()
    value.eval()

    return actor, value


def sample_latent_states(latent_dir, num_states, seed, device):
    rng = np.random.default_rng(seed)
    files = sorted(Path(latent_dir).glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {latent_dir}")

    zs, hs, cs = [], [], []
    sources = []

    for _ in range(num_states):
        path = files[int(rng.integers(0, len(files)))]

        with np.load(path) as data:
            z = data["z"].astype(np.float32)
            h = data["h_next"].astype(np.float32)
            c = data["c_next"].astype(np.float32)

        T = h.shape[0]
        t = int(rng.integers(1, T + 1))

        zs.append(z[t].reshape(-1))
        hs.append(h[t - 1].reshape(-1))
        cs.append(c[t - 1].reshape(-1))
        sources.append((path.name, t))

    z = torch.tensor(np.stack(zs), dtype=torch.float32, device=device)
    h = torch.tensor(np.stack(hs), dtype=torch.float32, device=device).unsqueeze(0)
    c = torch.tensor(np.stack(cs), dtype=torch.float32, device=device).unsqueeze(0)

    return LatentState(z=z, h=h, c=c, extra={}), sources


def lambda_returns(rewards, values, discounts, lambda_):
    """
    rewards:   [H, B, 1]
    values:    [H + 1, B, 1]
    discounts: [H, B, 1]
    """
    H = rewards.shape[0]

    next_return = values[-1]
    returns = []

    for t in reversed(range(H)):
        next_value = values[t + 1]
        next_return = rewards[t] + discounts[t] * (
            (1.0 - lambda_) * next_value + lambda_ * next_return
        )
        returns.append(next_return)

    returns.reverse()
    return torch.stack(returns, dim=0)


def discounted_sum(rewards, discounts):
    """
    rewards:   [H, B, 1]
    discounts: [H, B, 1]
    """
    H = rewards.shape[0]
    out = torch.zeros_like(rewards[0])
    discount_prod = torch.ones_like(rewards[0])

    for t in range(H):
        out = out + discount_prod * rewards[t]
        discount_prod = discount_prod * discounts[t]

    return out


def plot_mean_curves(per_step_df, out_path):
    mean_step = per_step_df.groupby("t")[
        [
            "reward",
            "value",
            "lambda_return",
            "reward_component",
            "value_component",
        ]
    ].mean()

    plt.figure(figsize=(9, 6))
    plt.plot(mean_step.index, mean_step["reward"], label="reward")
    plt.plot(mean_step.index, mean_step["value"], label="value")
    plt.plot(mean_step.index, mean_step["lambda_return"], label="lambda return")
    plt.plot(mean_step.index, mean_step["reward_component"], label="reward component")
    plt.plot(mean_step.index, mean_step["value_component"], label="value component")

    plt.xlabel("Imagined step")
    plt.ylabel("Mean value")
    plt.title("Mean imagined-return decomposition")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--actor-ckpt", type=str, required=True)
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)

    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="logs/imagined_return_decomposition")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        device=device,
    )
    world_model.eval()

    actor, value = load_actor_value(
        ckpt_path=args.actor_ckpt,
        device=device,
        feature_dim=world_model.feature_dim,
    )

    summary_rows = []
    per_step_rows = []

    seq_id = 0

    while seq_id < args.num_sequences:
        B = min(args.batch_size, args.num_sequences - seq_id)

        start_state, sources = sample_latent_states(
            latent_dir=args.latent_dir,
            num_states=B,
            seed=args.seed + seq_id,
            device=device,
        )

        with torch.no_grad():
            rollout = imagine_rollout(
                world_model=world_model,
                actor=actor,
                value=value,
                start_state=start_state,
                horizon=args.horizon,
                deterministic=args.deterministic,
            )

        rewards = rollout.rewards          # [H, B, 1]
        discounts = rollout.discounts      # [H, B, 1]
        values = rollout.values            # [H + 1, B, 1]

        full_lambda = lambda_returns(
            rewards=rewards,
            values=values,
            discounts=discounts,
            lambda_=args.lambda_,
        )

        reward_component = lambda_returns(
            rewards=rewards,
            values=torch.zeros_like(values),
            discounts=discounts,
            lambda_=args.lambda_,
        )

        value_component = lambda_returns(
            rewards=torch.zeros_like(rewards),
            values=values,
            discounts=discounts,
            lambda_=args.lambda_,
        )

        reward_discounted = discounted_sum(rewards, discounts)

        for b in range(B):
            source_file, source_t = sources[b]

            summary_rows.append({
                "seq_id": seq_id + b,
                "source_file": source_file,
                "source_t": source_t,
                "reward_sum": float(rewards[:, b].sum().cpu()),
                "discounted_reward_sum": float(reward_discounted[b].cpu()),
                "lambda_t0": float(full_lambda[0, b].cpu()),
                "lambda_mean": float(full_lambda[:, b].mean().cpu()),
                "reward_component_t0": float(reward_component[0, b].cpu()),
                "value_component_t0": float(value_component[0, b].cpu()),
                "value_t0": float(values[0, b].cpu()),
                "terminal_value": float(values[-1, b].cpu()),
            })

            for t in range(args.horizon):
                per_step_rows.append({
                    "seq_id": seq_id + b,
                    "t": t,
                    "reward": float(rewards[t, b].cpu()),
                    "discount": float(discounts[t, b].cpu()),
                    "value": float(values[t, b].cpu()),
                    "lambda_return": float(full_lambda[t, b].cpu()),
                    "reward_component": float(reward_component[t, b].cpu()),
                    "value_component": float(value_component[t, b].cpu()),
                })

        seq_id += B
        print(f"processed {seq_id}/{args.num_sequences}")

    summary_df = pd.DataFrame(summary_rows)
    per_step_df = pd.DataFrame(per_step_rows)

    summary_csv = out_dir / "sequence_return_decomposition.csv"
    per_step_csv = out_dir / "per_step_return_decomposition.csv"
    plot_path = out_dir / "mean_per_step_decomposition.png"

    summary_df.to_csv(summary_csv, index=False)
    per_step_df.to_csv(per_step_csv, index=False)
    plot_mean_curves(per_step_df, plot_path)

    print()
    print("Main averages:")
    print(summary_df.mean(numeric_only=True))

    print()
    print("Saved outputs:")
    print(summary_csv)
    print(per_step_csv)
    print(plot_path)


if __name__ == "__main__":
    main()