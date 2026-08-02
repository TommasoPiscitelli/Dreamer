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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = [
    "terminal_minus_100",
    "step_minus_0.1",
    "positive_small",
    "positive_medium",
    "positive_large",
]

DEFAULT_CLASS_VALUES = np.array([-100.0, -0.1, 3.3, 6.8, 11.0], dtype=np.float32)


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
    """
    Align arrays with rewards.

    If x has T+1 states, use x[1:T+1], because reward_t usually corresponds
    to the transition into the next latent state.
    If x has T states, use x[:T].
    """
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
    rows = []

    for path in files:
        try:
            X, r = load_one_npz(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        Xs.append(X)
        rs.append(r)

        rows.append({
            "file": path.name,
            "steps": len(r),
            "return": float(r.sum()),
            "mean_reward": float(r.mean()),
            "min_reward": float(r.min()),
            "max_reward": float(r.max()),
        })

    if len(Xs) == 0:
        raise RuntimeError("No valid episodes loaded.")

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    r = np.concatenate(rs, axis=0).astype(np.float32)
    y = reward_to_class(r)

    episode_df = pd.DataFrame(rows)

    return X, r, y, episode_df


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


@torch.no_grad()
def evaluate(model, loader, class_values, device):
    model.eval()

    all_logits = []
    all_y = []
    all_r = []

    total_loss = 0.0
    total_n = 0

    for xb, yb, rb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        rb = rb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        total_loss += float(loss.item()) * xb.shape[0]
        total_n += xb.shape[0]

        all_logits.append(logits.cpu())
        all_y.append(yb.cpu())
        all_r.append(rb.cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    r = torch.cat(all_r, dim=0)

    probs = torch.softmax(logits, dim=-1)
    pred_class = probs.argmax(dim=-1)

    class_values_t = torch.tensor(class_values, dtype=torch.float32)
    pred_reward = (probs * class_values_t[None, :]).sum(dim=-1)

    acc = float((pred_class == y).float().mean().item())

    class_accs = []
    for k in range(len(CLASS_NAMES)):
        mask = y == k
        if mask.any():
            class_accs.append(float((pred_class[mask] == y[mask]).float().mean().item()))
        else:
            class_accs.append(float("nan"))

    valid_class_accs = [a for a in class_accs if not np.isnan(a)]
    balanced_acc = float(np.mean(valid_class_accs)) if valid_class_accs else float("nan")

    err = pred_reward - r
    mse = float((err ** 2).mean().item())
    rmse = float(np.sqrt(mse))
    mae = float(err.abs().mean().item())

    r_np = r.numpy()
    pr_np = pred_reward.numpy()

    if np.std(r_np) > 1e-8 and np.std(pr_np) > 1e-8:
        pearson = float(np.corrcoef(r_np, pr_np)[0, 1])
    else:
        pearson = float("nan")

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for yt, yp in zip(y.numpy(), pred_class.numpy()):
        cm[int(yt), int(yp)] += 1

    metrics = {
        "loss": total_loss / max(total_n, 1),
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "expected_reward_mse": mse,
        "expected_reward_rmse": rmse,
        "expected_reward_mae": mae,
        "expected_reward_pearson": pearson,
    }

    for k, name in enumerate(CLASS_NAMES):
        metrics[f"accuracy_{name}"] = class_accs[k]

    return metrics, cm, r_np, pr_np, y.numpy(), pred_class.numpy()


def plot_training_curves(history, out_dir: Path):
    df = pd.DataFrame(history)
    df.to_csv(out_dir / "training_log.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy")
    plt.title("Reward classifier loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_accuracy"], label="val accuracy")
    plt.plot(df["epoch"], df["val_balanced_accuracy"], label="val balanced accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Reward classifier accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_expected_reward_mse"], label="val expected reward MSE")
    plt.plot(df["epoch"], df["val_expected_reward_mae"], label="val expected reward MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Expected reward prediction error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_expected_reward_error.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm, out_path: Path, title: str):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_eval_outputs(name, metrics, cm, true_r, pred_r, true_y, pred_y, out_dir: Path):
    split_dir = out_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(split_dir / "confusion_matrix.csv")
    plot_confusion_matrix(cm, split_dir / "confusion_matrix.png", f"{name} confusion matrix")

    pred_df = pd.DataFrame({
        "true_reward": true_r,
        "pred_expected_reward": pred_r,
        "true_class": true_y,
        "pred_class": pred_y,
    })
    pred_df.to_csv(split_dir / "predictions.csv", index=False)

    plt.figure(figsize=(8, 8))
    plt.scatter(true_r, pred_r, s=4, alpha=0.25)
    plt.xlabel("True reward")
    plt.ylabel("Predicted expected reward")
    plt.title(f"{name}: true reward vs predicted expected reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(split_dir / "true_vs_pred_expected_reward.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(true_r, bins=100, alpha=0.6, label="true reward")
    plt.hist(pred_r, bins=100, alpha=0.6, label="predicted expected reward")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.title(f"{name}: reward distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(split_dir / "reward_histograms.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.npz")

    parser.add_argument("--eval-latent-dir", type=str, default=None)
    parser.add_argument("--eval-pattern", type=str, default="*.npz")

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)

    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-class-weight", type=float, default=20.0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("Loading training dataset...")
    X, r, y, episode_df = load_dataset(Path(args.latent_dir), args.pattern)

    episode_df.to_csv(log_dir / "train_episode_summary.csv", index=False)

    N = len(X)
    input_dim = X.shape[1]
    num_classes = len(CLASS_NAMES)

    print(f"Loaded {N} samples")
    print(f"Feature dim: {input_dim}")

    class_counts = np.bincount(y, minlength=num_classes)

    class_summary = pd.DataFrame({
        "class_id": list(range(num_classes)),
        "class_name": CLASS_NAMES,
        "count": class_counts,
        "fraction": class_counts / class_counts.sum(),
    })

    class_summary.to_csv(log_dir / "class_summary.csv", index=False)
    print(class_summary.to_string(index=False))

    indices = np.arange(N)
    rng.shuffle(indices)

    val_size = int(args.val_frac * N)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train = X[train_idx]
    r_train = r[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    r_val = r[val_idx]
    y_val = y[val_idx]

    feature_mean = X_train.mean(axis=0, keepdims=True)
    feature_std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_train = (X_train - feature_mean) / feature_std
    X_val = (X_val - feature_mean) / feature_std

    class_values = DEFAULT_CLASS_VALUES.copy()
    for k in range(num_classes):
        mask = y_train == k
        if mask.any():
            class_values[k] = float(r_train[mask].mean())

    print("\nClass values used for expected reward:")
    for k, v in enumerate(class_values):
        print(f"{k} {CLASS_NAMES[k]}: {v:.6f}")

    train_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = np.sqrt(train_counts.sum() / (num_classes * np.maximum(train_counts, 1.0)))
    class_weights = class_weights / class_weights.mean()
    class_weights = np.clip(class_weights, 0.1, args.max_class_weight)

    print("\nClass weights:")
    for k, w in enumerate(class_weights):
        print(f"{k} {CLASS_NAMES[k]}: {w:.6f}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(r_train).float(),
    )

    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
        torch.from_numpy(r_val).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)

    model = RewardClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    weight_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_n = 0

        for xb, yb, rb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weight_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()

            total_loss += float(loss.item()) * xb.shape[0]
            total_n += xb.shape[0]

        train_loss = total_loss / max(total_n, 1)

        val_metrics, val_cm, val_true_r, val_pred_r, val_true_y, val_pred_y = evaluate(
            model=model,
            loader=val_loader,
            class_values=class_values,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_expected_reward_mse": val_metrics["expected_reward_mse"],
            "val_expected_reward_mae": val_metrics["expected_reward_mae"],
            "val_expected_reward_pearson": val_metrics["expected_reward_pearson"],
        }

        history.append(row)

        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_mse={val_metrics['expected_reward_mse']:.6f} "
            f"val_pearson={val_metrics['expected_reward_pearson']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            ckpt = {
                "model_state_dict": model.state_dict(),
                "feature_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_classes": num_classes,
                "class_names": CLASS_NAMES,
                "class_values": class_values,
                "class_weights": class_weights,
                "feature_mean": feature_mean.astype(np.float32),
                "feature_std": feature_std.astype(np.float32),
                "args": vars(args),
            }

            torch.save(ckpt, out_dir / "reward_classifier.pt")

            with open(out_dir / "reward_classifier_config.json", "w") as f:
                json.dump({
                    "feature_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_classes": num_classes,
                    "class_names": CLASS_NAMES,
                    "class_values": class_values.tolist(),
                    "class_weights": class_weights.tolist(),
                }, f, indent=2)

    plot_training_curves(history, log_dir)

    # Final validation outputs from best current model
    val_metrics, val_cm, val_true_r, val_pred_r, val_true_y, val_pred_y = evaluate(
        model=model,
        loader=val_loader,
        class_values=class_values,
        device=device,
    )

    save_eval_outputs(
        name="val",
        metrics=val_metrics,
        cm=val_cm,
        true_r=val_true_r,
        pred_r=val_pred_r,
        true_y=val_true_y,
        pred_y=val_pred_y,
        out_dir=log_dir,
    )

    # Optional external evaluation set
    if args.eval_latent_dir is not None:
        print("\nLoading external evaluation dataset...")
        X_eval, r_eval, y_eval, eval_episode_df = load_dataset(Path(args.eval_latent_dir), args.eval_pattern)
        eval_episode_df.to_csv(log_dir / "eval_episode_summary.csv", index=False)

        X_eval = (X_eval - feature_mean) / feature_std

        eval_ds = TensorDataset(
            torch.from_numpy(X_eval).float(),
            torch.from_numpy(y_eval).long(),
            torch.from_numpy(r_eval).float(),
        )

        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        eval_metrics, eval_cm, eval_true_r, eval_pred_r, eval_true_y, eval_pred_y = evaluate(
            model=model,
            loader=eval_loader,
            class_values=class_values,
            device=device,
        )

        save_eval_outputs(
            name="eval",
            metrics=eval_metrics,
            cm=eval_cm,
            true_r=eval_true_r,
            pred_r=eval_pred_r,
            true_y=eval_true_y,
            pred_y=eval_pred_y,
            out_dir=log_dir,
        )

        print("\nExternal eval metrics:")
        print(json.dumps(eval_metrics, indent=2))

    print("\nSaved checkpoint:")
    print(out_dir / "reward_classifier.pt")

    print("\nSaved logs:")
    print(log_dir)


if __name__ == "__main__":
    main()
