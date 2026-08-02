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


def state_features(state):
    return torch.cat([state.z, state.h[-1]], dim=-1)


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


def load_random_latent_states(latent_dir, pattern, num_states, seed, device):
    rng = np.random.default_rng(seed)
    files = sorted(Path(latent_dir).rglob(pattern))

    if len(files) == 0:
        raise RuntimeError(f"No latent files found in {latent_dir} with pattern {pattern}")

    zs, hs, cs, used = [], [], [], []

    while len(zs) < num_states:
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

        if h_t.ndim == 2:
            h_t = h_t[0]
        if c_t.ndim == 2:
            c_t = c_t[0]

        zs.append(z_t)
        hs.append(h_t)
        cs.append(c_t)
        used.append((str(f), t))

    z = torch.tensor(np.stack(zs), dtype=torch.float32, device=device)
    h = torch.tensor(np.stack(hs), dtype=torch.float32, device=device).unsqueeze(0)
    c = torch.tensor(np.stack(cs), dtype=torch.float32, device=device).unsqueeze(0)

    return LatentState(z=z, h=h, c=c, extra={}), used


def load_dataset_actions(raw_dir, pattern, max_actions, seed):
    rng = np.random.default_rng(seed)
    files = sorted(Path(raw_dir).rglob(pattern))

    actions_list = []

    for f in files:
        try:
            data = np.load(f)
            key = get_key(data, ["action", "actions", "acts"])
            a = data[key].astype(np.float32)

            if a.ndim == 2 and a.shape[1] == 3:
                actions_list.append(a)
        except Exception:
            continue

    if not actions_list:
        raise RuntimeError("No valid action arrays found.")

    actions = np.concatenate(actions_list, axis=0)

    if len(actions) > max_actions:
        idx = rng.choice(len(actions), size=max_actions, replace=False)
        actions = actions[idx]

    return actions


class DatasetLikePolicy:
    def __init__(self, actions_np, device, seed=0):
        self.actions = torch.tensor(actions_np, dtype=torch.float32, device=device)
        self.device = device
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

    def eval(self):
        return self

    @torch.no_grad()
    def sample(self, features, deterministic=False):
        B = features.shape[0]
        idx = torch.randint(
            low=0,
            high=self.actions.shape[0],
            size=(B,),
            generator=self.rng,
            device=self.device,
        )
        action = self.actions[idx]
        entropy = torch.zeros(B, device=self.device)
        return action, entropy


def orient_time_batch(x, horizon):
    x = torch.as_tensor(x)

    if x.ndim >= 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.shape[0] in {horizon, horizon + 1}:
        return x

    if x.ndim >= 2 and x.shape[1] in {horizon, horizon + 1}:
        return x.transpose(0, 1)

    raise ValueError(f"Cannot orient tensor with shape {tuple(x.shape)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--latent-pattern", type=str, default="*heuristic*.npz")
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--raw-pattern", type=str, default="*heuristic*.npz")

    parser.add_argument("--actor-ckpt", type=str, required=True)
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--num-states", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="logs/value_actor_vs_dataset_like")

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

    dataset_actions = load_dataset_actions(
        raw_dir=args.raw_dir,
        pattern=args.raw_pattern,
        max_actions=100000,
        seed=args.seed,
    )

    dataset_like_policy = DatasetLikePolicy(dataset_actions, device=device, seed=args.seed)

    rows = []

    done = 0

    while done < args.num_states:
        B = min(args.batch_size, args.num_states - done)

        start_state, used = load_random_latent_states(
            latent_dir=args.latent_dir,
            pattern=args.latent_pattern,
            num_states=B,
            seed=args.seed + done,
            device=device,
        )

        with torch.no_grad():
            real_value_t0 = value(state_features(start_state)).squeeze(-1)

            actor_out = imagine_rollout(
                world_model=world_model,
                actor=actor,
                value=value,
                start_state=start_state,
                horizon=args.horizon,
                deterministic=True,
            )

            dataset_out = imagine_rollout(
                world_model=world_model,
                actor=dataset_like_policy,
                value=value,
                start_state=start_state,
                horizon=args.horizon,
                deterministic=False,
            )

        actor_values = orient_time_batch(actor_out.values, args.horizon).to(device)
        dataset_values = orient_time_batch(dataset_out.values, args.horizon).to(device)

        actor_actions = orient_time_batch(actor_out.actions, args.horizon).to(device)
        dataset_actions_roll = orient_time_batch(dataset_out.actions, args.horizon).to(device)

        for b in range(B):
            seq_id = done + b

            rows.append({
                "seq_id": seq_id,
                "mode": "real_start",
                "t": 0,
                "value": float(real_value_t0[b].cpu()),
                "steer": np.nan,
                "gas": np.nan,
                "brake": np.nan,
                "source_file": used[b][0],
                "source_t": used[b][1],
            })

            for t in range(args.horizon):
                rows.append({
                    "seq_id": seq_id,
                    "mode": "actor_rollout",
                    "t": t + 1,
                    "value": float(actor_values[t, b].cpu()),
                    "steer": float(actor_actions[t, b, 0].cpu()),
                    "gas": float(actor_actions[t, b, 1].cpu()),
                    "brake": float(actor_actions[t, b, 2].cpu()),
                    "source_file": used[b][0],
                    "source_t": used[b][1],
                })

                rows.append({
                    "seq_id": seq_id,
                    "mode": "dataset_like_rollout",
                    "t": t + 1,
                    "value": float(dataset_values[t, b].cpu()),
                    "steer": float(dataset_actions_roll[t, b, 0].cpu()),
                    "gas": float(dataset_actions_roll[t, b, 1].cpu()),
                    "brake": float(dataset_actions_roll[t, b, 2].cpu()),
                    "source_file": used[b][0],
                    "source_t": used[b][1],
                })

        done += B
        print(f"processed {done}/{args.num_states}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "value_actor_vs_dataset_like.csv", index=False)

    print()
    print("Mean value by mode:")
    print(df.groupby("mode")["value"].describe())

    # Mean value over imagined time
    mean_step = df[df["mode"] != "real_start"].groupby(["mode", "t"])["value"].mean().reset_index()

    plt.figure(figsize=(9, 6))
    for mode in mean_step["mode"].unique():
        tmp = mean_step[mean_step["mode"] == mode]
        plt.plot(tmp["t"], tmp["value"], label=mode)

    real_mean = df[df["mode"] == "real_start"]["value"].mean()
    plt.axhline(real_mean, linestyle="--", label="real_start mean")

    plt.xlabel("Step")
    plt.ylabel("Mean predicted value")
    plt.title("Value along imagined rollouts")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_value_along_rollouts.png", dpi=200)
    plt.close()

    # Histograms
    plt.figure(figsize=(9, 6))

    real_values = df[df["mode"] == "real_start"]["value"]
    actor_final = df[(df["mode"] == "actor_rollout") & (df["t"] == args.horizon)]["value"]
    dataset_final = df[(df["mode"] == "dataset_like_rollout") & (df["t"] == args.horizon)]["value"]

    plt.hist(real_values, bins=40, alpha=0.5, density=True, label="real_start")
    plt.hist(actor_final, bins=40, alpha=0.5, density=True, label=f"actor_t{args.horizon}")
    plt.hist(dataset_final, bins=40, alpha=0.5, density=True, label=f"dataset_like_t{args.horizon}")

    plt.xlabel("Predicted value")
    plt.ylabel("Density")
    plt.title("Value distribution: real states vs imagined states")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hist_value_real_vs_actor_vs_dataset_like.png", dpi=200)
    plt.close()

    # Value increase per sequence
    real_df = df[df["mode"] == "real_start"][["seq_id", "value"]].rename(columns={"value": "value_start"})
    actor_last = df[(df["mode"] == "actor_rollout") & (df["t"] == args.horizon)][
        ["seq_id", "value", "steer", "gas", "brake"]
    ].rename(columns={"value": "value_actor_final"})
    data_last = df[(df["mode"] == "dataset_like_rollout") & (df["t"] == args.horizon)][
        ["seq_id", "value"]
    ].rename(columns={"value": "value_dataset_like_final"})

    comp = real_df.merge(actor_last, on="seq_id").merge(data_last, on="seq_id")
    comp["delta_actor"] = comp["value_actor_final"] - comp["value_start"]
    comp["delta_dataset_like"] = comp["value_dataset_like_final"] - comp["value_start"]
    comp.to_csv(out_dir / "value_delta_by_sequence.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(comp["delta_dataset_like"], comp["delta_actor"], s=12, alpha=0.5)
    plt.axline((0, 0), slope=1, linestyle="--", label="y=x")
    plt.xlabel("Value increase with dataset-like actions")
    plt.ylabel("Value increase with actor actions")
    plt.title("Does the actor reach higher-value states?")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "delta_value_actor_vs_dataset_like.png", dpi=200)
    plt.close()

    print()
    print("Saved outputs in:", out_dir)


if __name__ == "__main__":
    main()