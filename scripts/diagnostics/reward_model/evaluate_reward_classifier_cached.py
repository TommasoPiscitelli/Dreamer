import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES_DEFAULT = [
    "terminal_minus_100",
    "step_minus_0.1",
    "positive_small",
    "positive_medium",
    "positive_large",
]


class RewardClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        super().__init__()

        layers = []
        dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim

        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def reward_to_class(rewards: np.ndarray) -> np.ndarray:
    y = np.ones_like(rewards, dtype=np.int64)

    y[rewards <= -50.0] = 0
    y[np.isclose(rewards, -0.1, atol=1e-5)] = 1
    y[(rewards > 0.0) & (rewards <= 4.5)] = 2
    y[(rewards > 4.5) & (rewards <= 9.0)] = 3
    y[rewards > 9.0] = 4

    return y


def get_first_key(data, keys):
    for k in keys:
        if k in data:
            return k
    raise KeyError(f"None of keys {keys} found. Available keys: {list(data.keys())}")


def flatten_time_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x.reshape(x.shape[0], -1)


def align_to_rewards(x: np.ndarray, T: int, prefer_next: bool = True) -> np.ndarray:
    if x.shape[0] == T + 1 and prefer_next:
        return x[1:T + 1]
    if x.shape[0] >= T:
        return x[:T]
    raise ValueError(f"Cannot align array of length {x.shape[0]} with reward length {T}")


def load_one_npz(path: Path):
    data = np.load(path)

    z_key = get_first_key(data, ["z", "latent", "latents"])
    h_key = get_first_key(data, ["h_next", "h", "hidden", "hidden_next"])
    r_key = get_first_key(data, ["reward", "rewards", "r"])

    z = flatten_time_array(data[z_key]).astype(np.float32)
    h = flatten_time_array(data[h_key]).astype(np.float32)
    r = np.asarray(data[r_key], dtype=np.float32).reshape(-1)

    T = len(r)

    z = align_to_rewards(z, T, prefer_next=True)
    h = align_to_rewards(h, T, prefer_next=False)

    X = np.concatenate([z, h], axis=-1).astype(np.float32)

    if X.shape[0] != T:
        raise ValueError(f"Feature length {X.shape[0]} != reward length {T} in {path}")

    return X, r


def load_dataset(latent_dir: Path, pattern: str):
    files = sorted(latent_dir.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {latent_dir} with pattern {pattern}")

    Xs = []
    rs = []
    file_names = []
    timesteps = []
    episode_rows = []

    for path in files:
        try:
            X, r = load_one_npz(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        Xs.append(X)
        rs.append(r)

        file_names.extend([path.name] * len(r))
        timesteps.extend(list(range(len(r))))

        episode_rows.append({
            "file": path.name,
            "steps": len(r),
            "true_return": float(r.sum()),
            "true_mean_reward": float(r.mean()),
            "true_min_reward": float(r.min()),
            "true_max_reward": float(r.max()),
        })

    if len(Xs) == 0:
        raise RuntimeError("No valid files loaded.")

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    r = np.concatenate(rs, axis=0).astype(np.float32)
    y = reward_to_class(r)

    return X, r, y, np.array(file_names), np.array(timesteps), pd.DataFrame(episode_rows)


def explained_variance(y_true, y_pred):
    var_y = np.var(y_true)
    if var_y < 1e-12:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


@torch.no_grad()
def predict(model, X, batch_size, device, class_values):
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    logits_all = []

    model.eval()

    for (xb,) in loader:
        xb = xb.to(device)
        logits = model(xb)
        logits_all.append(logits.cpu())

    logits = torch.cat(logits_all, dim=0)
    probs = torch.softmax(logits, dim=-1)
    pred_class = probs.argmax(dim=-1)

    class_values_t = torch.tensor(class_values, dtype=torch.float32)
    pred_expected_reward = (probs * class_values_t[None, :]).sum(dim=-1)

    return (
        logits.numpy(),
        probs.numpy(),
        pred_class.numpy(),
        pred_expected_reward.numpy(),
    )


def compute_metrics(true_reward, true_class, pred_class, pred_expected_reward, class_names):
    err = pred_expected_reward - true_reward

    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    if np.std(true_reward) > 1e-8 and np.std(pred_expected_reward) > 1e-8:
        pearson = float(np.corrcoef(true_reward, pred_expected_reward)[0, 1])
    else:
        pearson = float("nan")

    acc = float(np.mean(pred_class == true_class))

    per_class_acc = {}
    accs = []
    for k, name in enumerate(class_names):
        mask = true_class == k
        if mask.any():
            v = float(np.mean(pred_class[mask] == true_class[mask]))
            per_class_acc[f"accuracy_{name}"] = v
            accs.append(v)
        else:
            per_class_acc[f"accuracy_{name}"] = float("nan")

    balanced_acc = float(np.mean(accs)) if len(accs) > 0 else float("nan")

    baseline_mean = float(np.mean(true_reward))
    baseline_pred = np.full_like(true_reward, baseline_mean)
    baseline_mse = float(np.mean((baseline_pred - true_reward) ** 2))

    metrics = {
        "num_samples": int(len(true_reward)),
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "expected_reward_mse": mse,
        "expected_reward_rmse": rmse,
        "expected_reward_mae": mae,
        "expected_reward_pearson": pearson,
        "expected_reward_explained_variance": explained_variance(true_reward, pred_expected_reward),
        "baseline_mean_reward": baseline_mean,
        "baseline_mse_predict_mean_reward": baseline_mse,
        "mse_improvement_over_mean_baseline": baseline_mse - mse,
        "true_reward_mean": float(np.mean(true_reward)),
        "true_reward_std": float(np.std(true_reward)),
        "pred_reward_mean": float(np.mean(pred_expected_reward)),
        "pred_reward_std": float(np.std(pred_expected_reward)),
        "true_reward_min": float(np.min(true_reward)),
        "true_reward_max": float(np.max(true_reward)),
        "pred_reward_min": float(np.min(pred_expected_reward)),
        "pred_reward_max": float(np.max(pred_expected_reward)),
    }

    metrics.update(per_class_acc)

    return metrics


def confusion_matrix(true_class, pred_class, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for yt, yp in zip(true_class, pred_class):
        cm[int(yt), int(yp)] += 1
    return cm


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title("Reward classifier confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_global_outputs(df, out_dir):
    """
    Produce only:
    - true_vs_pred_expected_reward.png
    """

    rng = np.random.default_rng(0)

    true = df["true_reward"].to_numpy(dtype=np.float32)
    pred = df["pred_expected_reward"].to_numpy(dtype=np.float32)
    err = pred - true

    finite = np.isfinite(true) & np.isfinite(pred)

    true_f = true[finite]
    pred_f = pred[finite]
    err_f = err[finite]

    if len(true_f) == 0:
        raise RuntimeError("No finite values found for plotting.")

    mse = float(np.mean(err_f ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err_f)))

    if np.std(true_f) > 1e-8 and np.std(pred_f) > 1e-8:
        pearson = float(np.corrcoef(true_f, pred_f)[0, 1])
    else:
        pearson = float("nan")

    max_points = 50000

    if len(true_f) > max_points:
        idx = rng.choice(len(true_f), size=max_points, replace=False)
        true_plot = true_f[idx]
        pred_plot = pred_f[idx]
    else:
        true_plot = true_f
        pred_plot = pred_f

    # assi fissati
    lo = -10.0
    hi = 20.0

    plt.figure(figsize=(8, 8))

    plt.scatter(
        true_plot,
        pred_plot,
        s=5,
        alpha=0.20,
        linewidths=0,
        label="samples",
    )

    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=2,
        label="ideal: predicted = true",
    )

    plt.axhline(0.0, linewidth=1, alpha=0.4)
    plt.axvline(0.0, linewidth=1, alpha=0.4)

    plt.xlabel("True reward")
    plt.ylabel("Predicted expected reward")
    plt.title(
        "True reward vs predicted expected reward\n"
        f"RMSE={rmse:.4f} | MAE={mae:.4f} | Pearson={pearson:.4f} | N={len(true_f)}"
    )

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "true_vs_pred_expected_reward.png", dpi=250)
    plt.close()

def plot_episode_curves(pred_df, episode_df, out_dir, num_episode_plots):
    episode_plot_dir = out_dir / "episode_curves"
    episode_plot_dir.mkdir(parents=True, exist_ok=True)

    selected_files = episode_df.sort_values("true_return", ascending=False)["file"].head(num_episode_plots).tolist()

    for file_name in selected_files:
        sub = pred_df[pred_df["file"] == file_name].sort_values("timestep")

        true_return = sub["true_reward"].sum()
        pred_return = sub["pred_expected_reward"].sum()

        plt.figure(figsize=(14, 6))
        plt.plot(sub["timestep"], sub["true_reward"], label="true reward", linewidth=2)
        plt.plot(sub["timestep"], sub["pred_expected_reward"], label="predicted expected reward", linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.title(f"{file_name}\ntrue return={true_return:.2f}, predicted expected return={pred_return:.2f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(episode_plot_dir / f"{Path(file_name).stem}_reward_curve.png", dpi=200)
        plt.close()

        plt.figure(figsize=(14, 6))
        plt.plot(sub["timestep"], sub["true_reward"].cumsum(), label="true cumulative return", linewidth=2)
        plt.plot(sub["timestep"], sub["pred_expected_reward"].cumsum(), label="predicted cumulative return", linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative return")
        plt.title(f"{file_name}\ntrue return={true_return:.2f}, predicted expected return={pred_return:.2f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(episode_plot_dir / f"{Path(file_name).stem}_cumulative_return.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--classifier-ckpt", type=str, required=True)
    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.npz")

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-episode-plots", type=int, default=5)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("Loading checkpoint...")
    ckpt = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)

    input_dim = int(ckpt.get("feature_dim", ckpt.get("input_dim", 288)))
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    num_layers = int(ckpt.get("num_layers", 3))
    num_classes = int(ckpt.get("num_classes", 5))

    class_names = ckpt.get("class_names", CLASS_NAMES_DEFAULT)
    class_values = np.asarray(ckpt["class_values"], dtype=np.float32)

    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)

    model = RewardClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Loading dataset...")
    X, true_reward, true_class, file_names, timesteps, episode_df = load_dataset(
        Path(args.latent_dir),
        args.pattern,
    )

    X_norm = (X - feature_mean) / feature_std

    print(f"Samples: {len(true_reward)}")
    print(f"Feature dim: {X.shape[1]}")

    logits, probs, pred_class, pred_expected_reward = predict(
        model=model,
        X=X_norm,
        batch_size=args.batch_size,
        device=device,
        class_values=class_values,
    )

    metrics = compute_metrics(
        true_reward=true_reward,
        true_class=true_class,
        pred_class=pred_class,
        pred_expected_reward=pred_expected_reward,
        class_names=class_names,
    )

    cm = confusion_matrix(true_class, pred_class, num_classes)

    pred_df = pd.DataFrame({
        "file": file_names,
        "timestep": timesteps,
        "true_reward": true_reward,
        "pred_expected_reward": pred_expected_reward,
        "true_class": true_class,
        "pred_class": pred_class,
        "error": pred_expected_reward - true_reward,
    })

    for k, name in enumerate(class_names):
        pred_df[f"prob_{name}"] = probs[:, k]

    episode_pred = (
        pred_df.groupby("file")
        .agg(
            steps=("true_reward", "size"),
            true_return=("true_reward", "sum"),
            pred_expected_return=("pred_expected_reward", "sum"),
            true_mean_reward=("true_reward", "mean"),
            pred_mean_reward=("pred_expected_reward", "mean"),
            reward_mse=("error", lambda x: float(np.mean(np.asarray(x) ** 2))),
            reward_mae=("error", lambda x: float(np.mean(np.abs(np.asarray(x))))),
        )
        .reset_index()
    )

    episode_pred["return_error"] = episode_pred["pred_expected_return"] - episode_pred["true_return"]

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    episode_pred.to_csv(out_dir / "episode_return_predictions.csv", index=False)

    plot_global_outputs(pred_df, out_dir)
    plot_episode_curves(pred_df, episode_pred, out_dir, args.num_episode_plots)

    true_ep = episode_pred["true_return"].to_numpy(dtype=np.float32)
    pred_ep = episode_pred["pred_expected_return"].to_numpy(dtype=np.float32)
    err_ep = pred_ep - true_ep

    finite_ep = np.isfinite(true_ep) & np.isfinite(pred_ep)
    true_ep_f = true_ep[finite_ep]
    pred_ep_f = pred_ep[finite_ep]
    err_ep_f = err_ep[finite_ep]

    ep_mse = float(np.mean(err_ep_f ** 2))
    ep_rmse = float(np.sqrt(ep_mse))
    ep_mae = float(np.mean(np.abs(err_ep_f)))

    if np.std(true_ep_f) > 1e-8 and np.std(pred_ep_f) > 1e-8:
        ep_pearson = float(np.corrcoef(true_ep_f, pred_ep_f)[0, 1])
    else:
        ep_pearson = float("nan")

    lo = float(min(np.min(true_ep_f), np.min(pred_ep_f)))
    hi = float(max(np.max(true_ep_f), np.max(pred_ep_f)))

    pad = 0.05 * max(1e-6, hi - lo)
    lo -= pad
    hi += pad

    plt.figure(figsize=(8, 8))

    plt.scatter(
        true_ep_f,
        pred_ep_f,
        s=16,
        alpha=0.7,
        linewidths=0,
        label="episodes",
    )

    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=2,
        label="ideal: predicted = true",
    )

    plt.axhline(0.0, linewidth=1, alpha=0.4)
    plt.axvline(0.0, linewidth=1, alpha=0.4)

    plt.xlabel("True episode return")
    plt.ylabel("Predicted expected episode return")
    plt.title(
        "Episode return: true vs predicted\n"
        f"RMSE={ep_rmse:.4f} | MAE={ep_mae:.4f} | Pearson={ep_pearson:.4f} | N={len(true_ep_f)}"
    )

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "episode_true_vs_pred_return.png", dpi=250)
    plt.close()

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    print("\nSaved outputs to:")
    print(out_dir)


if __name__ == "__main__":
    main()
