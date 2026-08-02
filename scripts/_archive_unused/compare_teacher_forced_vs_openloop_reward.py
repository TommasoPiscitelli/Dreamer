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
from dreamer.imagination import imagine_rollout


def get_key(data, candidates, required=True):
    for k in candidates:
        if k in data:
            return k
    if required:
        raise KeyError(f"None of {candidates} found. Available keys: {list(data.keys())}")
    return None


def orient_time_batch(x, horizon, name):
    x = torch.as_tensor(x)

    if x.ndim >= 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.shape[0] == horizon:
        return x

    if x.ndim >= 2 and x.shape[1] == horizon:
        return x.transpose(0, 1)

    raise ValueError(f"Cannot orient {name}, shape={tuple(x.shape)}, horizon={horizon}")


def state_features(state):
    return torch.cat([state.z, state.h[-1]], dim=-1)


def predict_reward(world_model, state):
    """
    Predict reward from a LatentState.

    The Ha world model adapter does not necessarily expose a public
    reward(state) method, so we try the common internal reward-model names.
    """

    features = state_features(state)

    if hasattr(world_model, "predict_reward"):
        r = world_model.predict_reward(state)

    elif hasattr(world_model, "reward"):
        r = world_model.reward(state)

    elif hasattr(world_model, "reward_model"):
        r = world_model.reward_model(features)

    elif hasattr(world_model, "_reward_model"):
        r = world_model._reward_model(features)

    elif hasattr(world_model, "rm"):
        r = world_model.rm(features)

    else:
        print("Available world_model attributes:")
        print([name for name in dir(world_model) if "reward" in name.lower() or "model" in name.lower()])
        raise AttributeError("Could not find a reward model inside world_model.")

    if isinstance(r, tuple):
        r = r[0]

    r = torch.as_tensor(r, device=features.device).squeeze(-1)

    # Apply reward calibration if the adapter stores it.
    # These names cover the likely cases; if none exist, nothing is changed.
    scale = None
    bias = None

    for name in ["reward_scale", "_reward_scale", "scale", "_scale"]:
        if hasattr(world_model, name):
            scale = getattr(world_model, name)
            break

    for name in ["reward_bias", "_reward_bias", "bias", "_bias"]:
        if hasattr(world_model, name):
            bias = getattr(world_model, name)
            break

    if scale is not None:
        r = r * float(scale)

    if bias is not None:
        r = r + float(bias)

    return r


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


class ZeroValue(torch.nn.Module):
    def forward(self, features):
        return torch.zeros(features.shape[0], device=features.device)


def discounted_sum(x, gamma):
    """
    x: [H, B]
    """
    H, B = x.shape
    out = torch.zeros(B, device=x.device)
    discount = torch.ones(B, device=x.device)

    for t in range(H):
        out += discount * x[t]
        discount *= gamma

    return out


def load_samples(latent_dir, pattern, num_sequences, horizon, seed, device):
    rng = np.random.default_rng(seed)
    files = sorted(Path(latent_dir).rglob(pattern))

    if not files:
        raise RuntimeError(f"No latent files found in {latent_dir} with pattern {pattern}")

    zs0, hs0, cs0 = [], [], []
    future_zs, future_hs, future_cs = [], [], []
    action_seqs = []
    true_reward_seqs = []
    has_true_reward = None
    used = []

    tries = 0

    while len(zs0) < num_sequences and tries < num_sequences * 100:
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
        r_key = get_key(data, ["reward", "rewards", "r"], required=False)

        z = data[z_key].astype(np.float32)
        h = data[h_key].astype(np.float32)
        c = data[c_key].astype(np.float32)
        actions = data[a_key].astype(np.float32)

        rewards = None
        if r_key is not None:
            rewards = data[r_key].astype(np.float32)
            if has_true_reward is None:
                has_true_reward = True
        elif has_true_reward is None:
            has_true_reward = False

        T = min(len(h), len(c), len(actions), len(z) - 1)

        if rewards is not None:
            T = min(T, len(rewards))

        if T <= horizon + 2:
            continue

        # Stato reale al tempo t:
        # z[t], h_next[t-1], c_next[t-1]
        #
        # Stati reali futuri dopo le azioni:
        # z[t+k+1], h_next[t+k], c_next[t+k]
        #
        # Azioni vere heuristic:
        # a[t], ..., a[t+H-1]
        t = int(rng.integers(1, T - horizon))

        z0 = z[t]
        h0 = h[t - 1]
        c0 = c[t - 1]

        if h0.ndim == 2:
            h0 = h0[0]
        if c0.ndim == 2:
            c0 = c0[0]

        fz = z[t + 1 : t + horizon + 1]
        fh = h[t : t + horizon]
        fc = c[t : t + horizon]

        if fh.ndim == 3:
            fh = fh[:, 0, :]
        if fc.ndim == 3:
            fc = fc[:, 0, :]

        aseq = actions[t : t + horizon]

        if rewards is not None:
            rseq = rewards[t : t + horizon]
        else:
            rseq = np.full((horizon,), np.nan, dtype=np.float32)

        zs0.append(z0)
        hs0.append(h0)
        cs0.append(c0)

        future_zs.append(fz)
        future_hs.append(fh)
        future_cs.append(fc)

        action_seqs.append(aseq)
        true_reward_seqs.append(rseq)
        used.append((str(f), t))

    if len(zs0) < num_sequences:
        raise RuntimeError(f"Only sampled {len(zs0)} sequences out of {num_sequences}")

    z0 = torch.tensor(np.stack(zs0), dtype=torch.float32, device=device)
    h0 = torch.tensor(np.stack(hs0), dtype=torch.float32, device=device).unsqueeze(0)
    c0 = torch.tensor(np.stack(cs0), dtype=torch.float32, device=device).unsqueeze(0)

    future_z = torch.tensor(np.stack(future_zs), dtype=torch.float32, device=device).transpose(0, 1)
    future_h = torch.tensor(np.stack(future_hs), dtype=torch.float32, device=device).transpose(0, 1)
    future_c = torch.tensor(np.stack(future_cs), dtype=torch.float32, device=device).transpose(0, 1)

    actions = torch.tensor(np.stack(action_seqs), dtype=torch.float32, device=device).transpose(0, 1)
    true_rewards = torch.tensor(np.stack(true_reward_seqs), dtype=torch.float32, device=device).transpose(0, 1)

    start_state = LatentState(z=z0, h=h0, c=c0, extra={})

    return start_state, future_z, future_h, future_c, actions, true_rewards, used, bool(has_true_reward)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--latent-pattern", type=str, default="*heuristic*.npz")

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--num-sequences", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="logs/teacher_forced_vs_openloop_heuristic_reward")

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

    zero_value = ZeroValue().to(device)

    seq_rows = []
    step_rows = []

    done = 0

    while done < args.num_sequences:
        B = min(args.batch_size, args.num_sequences - done)

        (
            start_state,
            future_z,
            future_h,
            future_c,
            heuristic_actions,
            true_rewards,
            used,
            has_true_reward,
        ) = load_samples(
            latent_dir=args.latent_dir,
            pattern=args.latent_pattern,
            num_sequences=B,
            horizon=args.horizon,
            seed=args.seed + done,
            device=device,
        )

        # A) Teacher-forced:
        # valutiamo il reward model sui veri stati latenti futuri del dataset.
        teacher_rewards = []

        with torch.no_grad():
            for k in range(args.horizon):
                real_next_state = LatentState(
                    z=future_z[k],
                    h=future_h[k].unsqueeze(0),
                    c=future_c[k].unsqueeze(0),
                    extra={},
                )
                r = predict_reward(world_model, real_next_state)
                teacher_rewards.append(r)

            teacher_rewards = torch.stack(teacher_rewards, dim=0)

            # B) Open-loop:
            # partiamo dallo stesso stato iniziale e usiamo le vere azioni heuristic,
            # ma lasciamo evolvere il transition model.
            fixed_policy = FixedActionPolicy(heuristic_actions)
            fixed_policy.reset()

            openloop_out = imagine_rollout(
                world_model=world_model,
                actor=fixed_policy,
                value=zero_value,
                start_state=start_state,
                horizon=args.horizon,
                deterministic=True,
            )

            openloop_rewards = orient_time_batch(
                openloop_out.rewards,
                args.horizon,
                "openloop_rewards",
            ).to(device)

        teacher_reward_sum = discounted_sum(teacher_rewards, args.gamma)
        openloop_reward_sum = discounted_sum(openloop_rewards, args.gamma)

        if has_true_reward:
            true_reward_sum = discounted_sum(true_rewards, args.gamma)
        else:
            true_reward_sum = torch.full_like(teacher_reward_sum, float("nan"))

        for b in range(B):
            seq_id = done + b

            seq_rows.append({
                "seq_id": seq_id,
                "source_file": used[b][0],
                "source_t": used[b][1],
                "teacher_forced_pred_reward_sum_discounted": float(teacher_reward_sum[b].cpu()),
                "openloop_pred_reward_sum_discounted": float(openloop_reward_sum[b].cpu()),
                "true_reward_sum_discounted": float(true_reward_sum[b].cpu()),
                "teacher_forced_pred_reward_mean": float(teacher_rewards[:, b].mean().cpu()),
                "openloop_pred_reward_mean": float(openloop_rewards[:, b].mean().cpu()),
                "true_reward_mean": float(true_rewards[:, b].mean().cpu()) if has_true_reward else np.nan,
                "diff_teacher_minus_openloop_sum": float((teacher_reward_sum[b] - openloop_reward_sum[b]).cpu()),
            })

            for t in range(args.horizon):
                step_rows.append({
                    "seq_id": seq_id,
                    "t": t,
                    "source_file": used[b][0],
                    "source_t": used[b][1],
                    "teacher_forced_pred_reward": float(teacher_rewards[t, b].cpu()),
                    "openloop_pred_reward": float(openloop_rewards[t, b].cpu()),
                    "true_reward": float(true_rewards[t, b].cpu()) if has_true_reward else np.nan,
                    "steer": float(heuristic_actions[t, b, 0].cpu()),
                    "gas": float(heuristic_actions[t, b, 1].cpu()),
                    "brake": float(heuristic_actions[t, b, 2].cpu()),
                })

        done += B
        print(f"processed {done}/{args.num_sequences}")

    seq_df = pd.DataFrame(seq_rows)
    step_df = pd.DataFrame(step_rows)

    seq_df.to_csv(out_dir / "sequence_teacher_forced_vs_openloop_reward.csv", index=False)
    step_df.to_csv(out_dir / "per_step_teacher_forced_vs_openloop_reward.csv", index=False)

    print()
    print("Sequence-level means:")
    print(seq_df[[
        "teacher_forced_pred_reward_sum_discounted",
        "openloop_pred_reward_sum_discounted",
        "true_reward_sum_discounted",
        "teacher_forced_pred_reward_mean",
        "openloop_pred_reward_mean",
        "true_reward_mean",
        "diff_teacher_minus_openloop_sum",
    ]].mean())

    # Plot 1: reward medio per step
    mean_step = step_df.groupby("t")[[
        "teacher_forced_pred_reward",
        "openloop_pred_reward",
        "true_reward",
    ]].mean().reset_index()

    plt.figure(figsize=(9, 6))
    plt.plot(mean_step["t"], mean_step["teacher_forced_pred_reward"], label="teacher-forced predicted reward")
    plt.plot(mean_step["t"], mean_step["openloop_pred_reward"], label="open-loop predicted reward")

    if not mean_step["true_reward"].isna().all():
        plt.plot(mean_step["t"], mean_step["true_reward"], label="true environment reward")

    plt.xlabel("Step")
    plt.ylabel("Mean reward")
    plt.title("Teacher-forced vs open-loop reward prediction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_per_step_teacher_forced_vs_openloop_reward.png", dpi=200)
    plt.close()

    # Plot 2: scatter somme reward
    plt.figure(figsize=(7, 6))
    plt.scatter(
        seq_df["teacher_forced_pred_reward_sum_discounted"],
        seq_df["openloop_pred_reward_sum_discounted"],
        s=14,
        alpha=0.45,
    )

    lo = min(
        seq_df["teacher_forced_pred_reward_sum_discounted"].min(),
        seq_df["openloop_pred_reward_sum_discounted"].min(),
    )
    hi = max(
        seq_df["teacher_forced_pred_reward_sum_discounted"].max(),
        seq_df["openloop_pred_reward_sum_discounted"].max(),
    )

    plt.plot([lo, hi], [lo, hi], "--", label="y=x")
    plt.xlabel("Teacher-forced predicted reward sum")
    plt.ylabel("Open-loop predicted reward sum")
    plt.title("Does open-loop transition preserve reward predictions?")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_teacher_forced_vs_openloop_reward_sum.png", dpi=200)
    plt.close()

    # Plot 3: istogramma differenze
    plt.figure(figsize=(8, 5))
    plt.hist(seq_df["diff_teacher_minus_openloop_sum"], bins=40, alpha=0.8)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Teacher-forced reward sum - open-loop reward sum")
    plt.ylabel("Count")
    plt.title("Reward loss caused by open-loop transition")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "hist_teacher_minus_openloop_reward_sum.png", dpi=200)
    plt.close()

    # Plot 4: confronto con true reward, se presente nel file latent
    if not seq_df["true_reward_sum_discounted"].isna().all():
        plt.figure(figsize=(7, 6))
        plt.scatter(
            seq_df["true_reward_sum_discounted"],
            seq_df["teacher_forced_pred_reward_sum_discounted"],
            s=14,
            alpha=0.45,
            label="teacher-forced",
        )
        plt.scatter(
            seq_df["true_reward_sum_discounted"],
            seq_df["openloop_pred_reward_sum_discounted"],
            s=14,
            alpha=0.45,
            label="open-loop",
        )
        plt.xlabel("True reward sum")
        plt.ylabel("Predicted reward sum")
        plt.title("Predicted reward sums vs true reward sum")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_true_vs_predicted_reward_sums.png", dpi=200)
        plt.close()

    print()
    print("Saved outputs in:", out_dir)
    print(" - sequence_teacher_forced_vs_openloop_reward.csv")
    print(" - per_step_teacher_forced_vs_openloop_reward.csv")
    print(" - mean_per_step_teacher_forced_vs_openloop_reward.png")
    print(" - scatter_teacher_forced_vs_openloop_reward_sum.png")
    print(" - hist_teacher_minus_openloop_reward_sum.png")
    print(" - scatter_true_vs_predicted_reward_sums.png")


if __name__ == "__main__":
    main()