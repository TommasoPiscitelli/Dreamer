import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from world_model.api import LatentState
from world_model.load_legacy import load_legacy_world_model
from dreamer.actor_critic import Actor, Value
from dreamer.imagination import imagine_rollout


def get_key(data, candidates):
    for k in candidates:
        if k in data:
            return k
    raise KeyError(f"None of {candidates} found. Available keys: {list(data.keys())}")


def load_actor_value(ckpt_path, device, feature_dim=288):
    actor = Actor(feature_dim=feature_dim).to(device)
    value = Value(feature_dim=feature_dim).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    actor_state = (
        ckpt.get("actor_state_dict")
        or ckpt.get("actor")
        or ckpt.get("actor_state")
    )
    value_state = (
        ckpt.get("value_state_dict")
        or ckpt.get("value")
        or ckpt.get("value_state")
    )

    if actor_state is None or value_state is None:
        raise KeyError(f"Could not find actor/value keys. Available keys: {ckpt.keys()}")

    actor.load_state_dict(actor_state)
    value.load_state_dict(value_state)

    actor.eval()
    value.eval()

    return actor, value


def load_random_latent_states(latent_dir, pattern, num_states, seed, device):
    rng = np.random.default_rng(seed)
    files = sorted(Path(latent_dir).rglob(pattern))

    if len(files) == 0:
        raise RuntimeError(f"No latent files found in {latent_dir} with pattern {pattern}")

    zs, hs, cs = [], [], []
    used = []

    tries = 0
    while len(zs) < num_states and tries < num_states * 20:
        tries += 1
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

        if T <= 1:
            continue

        t = int(rng.integers(1, T + 1))

        z_t = z[t]
        h_t = h[t - 1]
        c_t = c[t - 1]

        if h_t.ndim == 2 and h_t.shape[0] == 1:
            h_t = h_t[0]
        if c_t.ndim == 2 and c_t.shape[0] == 1:
            c_t = c_t[0]

        zs.append(z_t)
        hs.append(h_t)
        cs.append(c_t)
        used.append((str(f), t))

    if len(zs) < num_states:
        raise RuntimeError(f"Only sampled {len(zs)} states out of requested {num_states}")

    z = torch.tensor(np.stack(zs), dtype=torch.float32, device=device)
    h = torch.tensor(np.stack(hs), dtype=torch.float32, device=device).unsqueeze(0)
    c = torch.tensor(np.stack(cs), dtype=torch.float32, device=device).unsqueeze(0)

    return LatentState(z=z, h=h, c=c, extra={}), used


def orient_time_batch(x, horizon, name):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x = x.squeeze(-1) if x.ndim >= 3 and x.shape[-1] == 1 else x

    if x.ndim == 1:
        x = x.unsqueeze(1)

    if x.shape[0] in {horizon, horizon + 1}:
        return x

    if x.ndim >= 2 and x.shape[1] in {horizon, horizon + 1}:
        return x.transpose(0, 1)

    raise ValueError(f"Cannot orient {name}, shape={tuple(x.shape)}, horizon={horizon}")


def lambda_returns(rewards, values_ext, discounts, lambda_):
    """
    rewards:   [H, B]
    values_ext:[H+1, B]
    discounts: [H, B]
    """
    H = rewards.shape[0]

    next_return = values_ext[-1]
    returns = []

    for t in reversed(range(H)):
        next_value = values_ext[t + 1]
        next_return = rewards[t] + discounts[t] * (
            (1.0 - lambda_) * next_value + lambda_ * next_return
        )
        returns.append(next_return)

    returns.reverse()
    return torch.stack(returns, dim=0)


def discounted_reward_sum(rewards, discounts):
    H, B = rewards.shape
    out = torch.zeros(B, device=rewards.device)
    prod = torch.ones(B, device=rewards.device)

    for t in range(H):
        out = out + prod * rewards[t]
        prod = prod * discounts[t]

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--latent-pattern", type=str, default="*.npz")

    parser.add_argument("--actor-ckpt", type=str, required=True)
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

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

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("horizon:", args.horizon)
    print("lambda:", args.lambda_)
    print("deterministic actor:", args.deterministic)

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )
    world_model.eval()

    actor, value = load_actor_value(args.actor_ckpt, device=device)

    all_summary = []
    all_per_step = []

    seq_offset = 0

    while seq_offset < args.num_sequences:
        B = min(args.batch_size, args.num_sequences - seq_offset)

        start_state, used = load_random_latent_states(
            latent_dir=args.latent_dir,
            pattern=args.latent_pattern,
            num_states=B,
            seed=args.seed + seq_offset,
            device=device,
        )

        with torch.no_grad():
            out = imagine_rollout(
                world_model=world_model,
                actor=actor,
                value=value,
                start_state=start_state,
                horizon=args.horizon,
                deterministic=args.deterministic,
            )

        rewards = orient_time_batch(out.rewards, args.horizon, "rewards").to(device)
        discounts = getattr(out, "discounts", None)

        if discounts is None:
            discounts = torch.ones_like(rewards) * args.gamma
        else:
            discounts = orient_time_batch(discounts, args.horizon, "discounts").to(device)

        values = orient_time_batch(out.values, args.horizon, "values").to(device)

        if values.shape[0] == args.horizon:
            values_ext = torch.cat([values, values[-1:].clone()], dim=0)
        else:
            values_ext = values

        actions = getattr(out, "actions", None)
        if actions is not None:
            actions = torch.as_tensor(actions).detach()
            if actions.shape[0] != args.horizon and actions.shape[1] == args.horizon:
                actions = actions.transpose(0, 1)
            actions = actions.to(device)
        else:
            actions = None

        full_lambda = lambda_returns(rewards, values_ext, discounts, args.lambda_)
        reward_component = lambda_returns(
            rewards,
            torch.zeros_like(values_ext),
            discounts,
            args.lambda_,
        )
        value_component = lambda_returns(
            torch.zeros_like(rewards),
            values_ext,
            discounts,
            args.lambda_,
        )

        reward_mc = discounted_reward_sum(rewards, discounts)

        for b in range(B):
            seq_id = seq_offset + b

            row = {
                "seq_id": seq_id,
                "source_file": used[b][0],
                "source_t": used[b][1],
                "reward_sum_undiscounted": float(rewards[:, b].sum().cpu()),
                "reward_sum_discounted": float(reward_mc[b].cpu()),
                "reward_mean": float(rewards[:, b].mean().cpu()),
                "reward_max": float(rewards[:, b].max().cpu()),
                "lambda_t0": float(full_lambda[0, b].cpu()),
                "lambda_mean_all_steps": float(full_lambda[:, b].mean().cpu()),
                "reward_component_t0": float(reward_component[0, b].cpu()),
                "value_component_t0": float(value_component[0, b].cpu()),
                "value_t0": float(values_ext[0, b].cpu()),
                "terminal_value": float(values_ext[-1, b].cpu()),
                "value_mean": float(values_ext[:, b].mean().cpu()),
                "value_max": float(values_ext[:, b].max().cpu()),
                "value_min": float(values_ext[:, b].min().cpu()),
            }

            if actions is not None:
                row.update({
                    "steer_mean": float(actions[:, b, 0].mean().cpu()),
                    "gas_mean": float(actions[:, b, 1].mean().cpu()),
                    "brake_mean": float(actions[:, b, 2].mean().cpu()),
                    "steer_abs_gt_09_frac": float((actions[:, b, 0].abs() > 0.9).float().mean().cpu()),
                    "gas_gt_095_frac": float((actions[:, b, 1] > 0.95).float().mean().cpu()),
                    "brake_lt_005_frac": float((actions[:, b, 2] < 0.05).float().mean().cpu()),
                })

            all_summary.append(row)

            for t in range(args.horizon):
                step_row = {
                    "seq_id": seq_id,
                    "t": t,
                    "reward": float(rewards[t, b].cpu()),
                    "discount": float(discounts[t, b].cpu()),
                    "value": float(values_ext[t, b].cpu()),
                    "next_value": float(values_ext[t + 1, b].cpu()),
                    "lambda_return": float(full_lambda[t, b].cpu()),
                    "reward_component": float(reward_component[t, b].cpu()),
                    "value_component": float(value_component[t, b].cpu()),
                }

                if actions is not None:
                    step_row.update({
                        "steer": float(actions[t, b, 0].cpu()),
                        "gas": float(actions[t, b, 1].cpu()),
                        "brake": float(actions[t, b, 2].cpu()),
                    })

                all_per_step.append(step_row)

        seq_offset += B
        print(f"processed {seq_offset}/{args.num_sequences}")

    summary = pd.DataFrame(all_summary)
    per_step = pd.DataFrame(all_per_step)

    summary.to_csv(out_dir / "sequence_return_decomposition.csv", index=False)
    per_step.to_csv(out_dir / "per_step_return_decomposition.csv", index=False)

    print()
    print("Main averages:")
    cols = [
        "reward_sum_undiscounted",
        "reward_sum_discounted",
        "lambda_t0",
        "lambda_mean_all_steps",
        "reward_component_t0",
        "value_component_t0",
        "value_t0",
        "terminal_value",
        "value_mean",
    ]
    print(summary[cols].mean())

    # Plot 1: reward sum vs lambda
    plt.figure(figsize=(7, 6))
    plt.scatter(summary["reward_sum_discounted"], summary["lambda_t0"], s=12, alpha=0.5)
    plt.xlabel("Discounted predicted reward sum over horizon")
    plt.ylabel("Lambda return at t=0")
    plt.title("Predicted reward sum vs lambda return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_reward_sum_vs_lambda_t0.png", dpi=200)
    plt.close()

    # Plot 2: components
    plt.figure(figsize=(8, 6))
    plt.scatter(summary["reward_component_t0"], summary["value_component_t0"], s=12, alpha=0.5)
    plt.xlabel("Reward component of lambda return")
    plt.ylabel("Value/bootstrap component of lambda return")
    plt.title("Lambda-return decomposition")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_reward_component_vs_value_component.png", dpi=200)
    plt.close()

    # Plot 3: mean curves
    mean_step = per_step.groupby("t")[[
        "reward",
        "value",
        "lambda_return",
        "reward_component",
        "value_component",
    ]].mean().reset_index()

    plt.figure(figsize=(9, 6))
    plt.plot(mean_step["t"], mean_step["reward"], label="predicted reward")
    plt.plot(mean_step["t"], mean_step["value"], label="predicted value")
    plt.plot(mean_step["t"], mean_step["lambda_return"], label="lambda return")
    plt.plot(mean_step["t"], mean_step["reward_component"], label="reward component")
    plt.plot(mean_step["t"], mean_step["value_component"], label="value component")
    plt.xlabel("Imagined step")
    plt.ylabel("Mean value")
    plt.title("Average imagined-return decomposition")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_per_step_decomposition.png", dpi=200)
    plt.close()

    # Plot 4: histograms
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    axes[0].hist(summary["reward_sum_discounted"], bins=40, alpha=0.8)
    axes[0].set_title("Discounted predicted reward sum")
    axes[0].grid(True)

    axes[1].hist(summary["value_component_t0"], bins=40, alpha=0.8)
    axes[1].set_title("Value/bootstrap component at t=0")
    axes[1].grid(True)

    axes[2].hist(summary["lambda_t0"], bins=40, alpha=0.8)
    axes[2].set_title("Lambda return at t=0")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "hist_return_components.png", dpi=200)
    plt.close()

    print()
    print("Saved outputs in:", out_dir)
    print(" - sequence_return_decomposition.csv")
    print(" - per_step_return_decomposition.csv")
    print(" - scatter_reward_sum_vs_lambda_t0.png")
    print(" - scatter_reward_component_vs_value_component.png")
    print(" - mean_per_step_decomposition.png")
    print(" - hist_return_components.png")


if __name__ == "__main__":
    main()