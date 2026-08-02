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

    actor_state = ckpt.get("actor_state_dict") or ckpt.get("actor") or ckpt.get("actor_state")
    value_state = ckpt.get("value_state_dict") or ckpt.get("value") or ckpt.get("value_state")

    if actor_state is None or value_state is None:
        raise KeyError(f"Could not find actor/value keys. Available keys: {ckpt.keys()}")

    actor.load_state_dict(actor_state)
    value.load_state_dict(value_state)

    actor.eval()
    value.eval()

    return actor, value


class FixedActionPolicy:
    def __init__(self, actions_hb3):
        """
        actions_hb3: tensor [H, B, 3]
        """
        self.actions = actions_hb3
        self.step = 0

    def eval(self):
        return self

    def reset(self):
        self.step = 0

    @torch.no_grad()
    def sample(self, features, deterministic=False):
        action = self.actions[self.step]
        self.step += 1
        entropy = torch.zeros(features.shape[0], device=features.device)
        return action, entropy


def orient_time_batch(x, horizon, name):
    x = torch.as_tensor(x)

    if x.ndim >= 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.shape[0] in {horizon, horizon + 1}:
        return x

    if x.ndim >= 2 and x.shape[1] in {horizon, horizon + 1}:
        return x.transpose(0, 1)

    raise ValueError(f"Cannot orient {name}, shape={tuple(x.shape)}")


def lambda_returns(rewards, values_ext, discounts, lambda_):
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


def load_samples(latent_dir, pattern, num_sequences, horizon, seed, device):
    rng = np.random.default_rng(seed)
    files = sorted(Path(latent_dir).rglob(pattern))

    if not files:
        raise RuntimeError(f"No latent files found in {latent_dir} with pattern {pattern}")

    zs, hs, cs, action_seqs, used = [], [], [], [], []

    tries = 0
    while len(zs) < num_sequences and tries < num_sequences * 50:
        tries += 1

        f = files[int(rng.integers(0, len(files)))]

        try:
            data = np.load(f)
        except Exception:
            continue

        z_key = get_key(data, ["z", "latents", "latent"])
        h_key = get_key(data, ["h_next", "h", "hidden", "hidden_next"])
        c_key = get_key(data, ["c_next", "c", "cell", "cell_next"])
        a_key = get_key(data, ["action", "actions", "acts"])

        z = data[z_key].astype(np.float32)
        h = data[h_key].astype(np.float32)
        c = data[c_key].astype(np.float32)
        actions = data[a_key].astype(np.float32)

        T = min(len(h), len(c), len(actions), len(z) - 1)

        if T <= horizon + 1:
            continue

        # state at time t uses z[t], h_next[t-1], c_next[t-1]
        # then fixed actions are a[t], ..., a[t+H-1]
        t = int(rng.integers(1, T - horizon))

        z_t = z[t]
        h_t = h[t - 1]
        c_t = c[t - 1]
        a_seq = actions[t : t + horizon]

        if h_t.ndim == 2:
            h_t = h_t[0]
        if c_t.ndim == 2:
            c_t = c_t[0]

        zs.append(z_t)
        hs.append(h_t)
        cs.append(c_t)
        action_seqs.append(a_seq)
        used.append((str(f), t))

    if len(zs) < num_sequences:
        raise RuntimeError(f"Only sampled {len(zs)} sequences out of {num_sequences}")

    z = torch.tensor(np.stack(zs), dtype=torch.float32, device=device)
    h = torch.tensor(np.stack(hs), dtype=torch.float32, device=device).unsqueeze(0)
    c = torch.tensor(np.stack(cs), dtype=torch.float32, device=device).unsqueeze(0)

    # [B, H, 3] -> [H, B, 3]
    actions = torch.tensor(np.stack(action_seqs), dtype=torch.float32, device=device).transpose(0, 1)

    start_state = LatentState(z=z, h=h, c=c, extra={})

    return start_state, actions, used


def extract_rollout_quantities(out, horizon, gamma, lambda_, device):
    rewards = orient_time_batch(out.rewards, horizon, "rewards").to(device)

    discounts = getattr(out, "discounts", None)
    if discounts is None:
        discounts = torch.ones_like(rewards) * gamma
    else:
        discounts = orient_time_batch(discounts, horizon, "discounts").to(device)

    values = orient_time_batch(out.values, horizon, "values").to(device)

    if values.shape[0] == horizon:
        values_ext = torch.cat([values, values[-1:].clone()], dim=0)
    else:
        values_ext = values

    full_lambda = lambda_returns(rewards, values_ext, discounts, lambda_)
    reward_component = lambda_returns(
        rewards,
        torch.zeros_like(values_ext),
        discounts,
        lambda_,
    )
    value_component = lambda_returns(
        torch.zeros_like(rewards),
        values_ext,
        discounts,
        lambda_,
    )
    reward_sum_discounted = discounted_reward_sum(rewards, discounts)

    return rewards, values_ext, full_lambda, reward_component, value_component, reward_sum_discounted


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--latent-pattern", type=str, default="*heuristic*.npz")

    parser.add_argument("--actor-ckpt", type=str, required=True)
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--num-sequences", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="logs/decompose_actor_vs_true_heuristic")

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

    actor, value = load_actor_value(args.actor_ckpt, device=device)

    per_step_rows = []
    seq_rows = []

    done = 0

    while done < args.num_sequences:
        B = min(args.batch_size, args.num_sequences - done)

        start_state, heuristic_actions, used = load_samples(
            latent_dir=args.latent_dir,
            pattern=args.latent_pattern,
            num_sequences=B,
            horizon=args.horizon,
            seed=args.seed + done,
            device=device,
        )

        heuristic_policy = FixedActionPolicy(heuristic_actions)

        with torch.no_grad():
            actor_out = imagine_rollout(
                world_model=world_model,
                actor=actor,
                value=value,
                start_state=start_state,
                horizon=args.horizon,
                deterministic=True,
            )

            heuristic_policy.reset()
            heuristic_out = imagine_rollout(
                world_model=world_model,
                actor=heuristic_policy,
                value=value,
                start_state=start_state,
                horizon=args.horizon,
                deterministic=True,
            )

        for mode, out in [
            ("actor", actor_out),
            ("true_heuristic_actions", heuristic_out),
        ]:
            rewards, values_ext, lambdas, reward_comp, value_comp, reward_sum = extract_rollout_quantities(
                out=out,
                horizon=args.horizon,
                gamma=args.gamma,
                lambda_=args.lambda_,
                device=device,
            )

            actions = orient_time_batch(out.actions, args.horizon, "actions").to(device)

            for b in range(B):
                seq_id = done + b

                seq_rows.append({
                    "seq_id": seq_id,
                    "mode": mode,
                    "source_file": used[b][0],
                    "source_t": used[b][1],
                    "reward_sum_discounted": float(reward_sum[b].cpu()),
                    "reward_sum_undiscounted": float(rewards[:, b].sum().cpu()),
                    "reward_mean": float(rewards[:, b].mean().cpu()),
                    "lambda_t0": float(lambdas[0, b].cpu()),
                    "lambda_mean": float(lambdas[:, b].mean().cpu()),
                    "reward_component_t0": float(reward_comp[0, b].cpu()),
                    "value_component_t0": float(value_comp[0, b].cpu()),
                    "value_t0": float(values_ext[0, b].cpu()),
                    "terminal_value": float(values_ext[-1, b].cpu()),
                    "value_mean": float(values_ext[:, b].mean().cpu()),
                    "steer_mean": float(actions[:, b, 0].mean().cpu()),
                    "gas_mean": float(actions[:, b, 1].mean().cpu()),
                    "brake_mean": float(actions[:, b, 2].mean().cpu()),
                })

                for t in range(args.horizon):
                    per_step_rows.append({
                        "seq_id": seq_id,
                        "mode": mode,
                        "t": t,
                        "reward": float(rewards[t, b].cpu()),
                        "value": float(values_ext[t, b].cpu()),
                        "lambda_return": float(lambdas[t, b].cpu()),
                        "reward_component": float(reward_comp[t, b].cpu()),
                        "value_component": float(value_comp[t, b].cpu()),
                        "steer": float(actions[t, b, 0].cpu()),
                        "gas": float(actions[t, b, 1].cpu()),
                        "brake": float(actions[t, b, 2].cpu()),
                    })

        done += B
        print(f"processed {done}/{args.num_sequences}")

    seq_df = pd.DataFrame(seq_rows)
    step_df = pd.DataFrame(per_step_rows)

    seq_df.to_csv(out_dir / "sequence_decomposition_actor_vs_true_heuristic.csv", index=False)
    step_df.to_csv(out_dir / "per_step_decomposition_actor_vs_true_heuristic.csv", index=False)

    print()
    print("Sequence-level means:")
    print(seq_df.groupby("mode")[[
        "reward_sum_discounted",
        "reward_mean",
        "lambda_t0",
        "reward_component_t0",
        "value_component_t0",
        "value_t0",
        "terminal_value",
        "value_mean",
        "steer_mean",
        "gas_mean",
        "brake_mean",
    ]].mean())

    # Mean per-step decomposition
    mean_step = step_df.groupby(["mode", "t"])[[
        "reward",
        "value",
        "lambda_return",
        "reward_component",
        "value_component",
    ]].mean().reset_index()

    for quantity in ["reward", "value", "lambda_return", "reward_component", "value_component"]:
        plt.figure(figsize=(9, 6))
        for mode in mean_step["mode"].unique():
            tmp = mean_step[mean_step["mode"] == mode]
            plt.plot(tmp["t"], tmp[quantity], label=mode)
        plt.xlabel("Imagined step")
        plt.ylabel(quantity)
        plt.title(f"{quantity}: actor vs true heuristic actions")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"mean_per_step_{quantity}.png", dpi=200)
        plt.close()

    # Bar plot of decomposition at t0
    means = seq_df.groupby("mode")[[
        "reward_component_t0",
        "value_component_t0",
        "lambda_t0",
        "reward_sum_discounted",
    ]].mean().reset_index()

    means.to_csv(out_dir / "mean_sequence_summary.csv", index=False)

    x = np.arange(len(means))
    width = 0.22

    plt.figure(figsize=(9, 6))
    plt.bar(x - width, means["reward_component_t0"], width, label="reward component")
    plt.bar(x, means["value_component_t0"], width, label="value component")
    plt.bar(x + width, means["lambda_t0"], width, label="lambda return")
    plt.xticks(x, means["mode"], rotation=10)
    plt.ylabel("Mean value")
    plt.title("Lambda-return decomposition at t=0")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bar_lambda_decomposition_t0.png", dpi=200)
    plt.close()

    # Scatter reward sum vs lambda
    plt.figure(figsize=(8, 6))
    for mode in seq_df["mode"].unique():
        tmp = seq_df[seq_df["mode"] == mode]
        plt.scatter(tmp["reward_sum_discounted"], tmp["lambda_t0"], s=14, alpha=0.45, label=mode)
    plt.xlabel("Discounted predicted reward sum")
    plt.ylabel("Lambda return at t=0")
    plt.title("Reward sum vs lambda return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_reward_sum_vs_lambda_t0.png", dpi=200)
    plt.close()

    print()
    print("Saved outputs in:", out_dir)


if __name__ == "__main__":
    main()