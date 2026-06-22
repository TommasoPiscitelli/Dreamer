from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dreamer_carracing.world_model.reward_model import RewardModel


def pearson_corr_np(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    x = x - x.mean()
    y = y - y.mean()

    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0

    return float(np.dot(x, y) / denom)


def load_reward_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    feature_dim = ckpt.get("feature_dim", 288)

    model = RewardModel(feature_dim=feature_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt


@torch.no_grad()
def predict_file(model, path, device, batch_size):
    data = np.load(path)

    features = data["features"].astype(np.float32)
    rewards = data["reward"].astype(np.float32)

    preds = []

    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start:start + batch_size]).to(device)
        y_hat = model(x).squeeze(-1).cpu().numpy()
        preds.append(y_hat)

    preds = np.concatenate(preds, axis=0)

    return rewards, preds


def describe_array(name, x):
    x = np.asarray(x).reshape(-1)

    print(f"\n{name}")
    print(f"  mean: {x.mean(): .6f}")
    print(f"  std:  {x.std(): .6f}")
    print(f"  min:  {x.min(): .6f}")
    print(f"  p05:  {np.percentile(x, 5): .6f}")
    print(f"  p50:  {np.percentile(x, 50): .6f}")
    print(f"  p95:  {np.percentile(x, 95): .6f}")
    print(f"  max:  {x.max(): .6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/latent/train")
    parser.add_argument("--ckpt", type=str, default="checkpoints/reward_model/reward_model.pt")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-csv", type=str, default="logs/reward_eval_per_episode.csv")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    model, ckpt = load_reward_model(args.ckpt, device)

    print("checkpoint:", args.ckpt)
    print("checkpoint epoch:", ckpt.get("epoch"))
    print("checkpoint metrics:", ckpt.get("metrics"))

    all_true = []
    all_pred = []

    rows = []

    for path in files:
        true_r, pred_r = predict_file(
            model=model,
            path=path,
            device=device,
            batch_size=args.batch_size,
        )

        all_true.append(true_r)
        all_pred.append(pred_r)

        mse = float(np.mean((pred_r - true_r) ** 2))
        mae = float(np.mean(np.abs(pred_r - true_r)))
        corr = pearson_corr_np(pred_r, true_r)

        true_return = float(true_r.sum())
        pred_return = float(pred_r.sum())

        rows.append({
            "file": path.name,
            "T": len(true_r),
            "mse": mse,
            "mae": mae,
            "corr": corr,
            "true_return": true_return,
            "pred_return": pred_return,
            "return_error": pred_return - true_return,
        })

    y = np.concatenate(all_true)
    y_hat = np.concatenate(all_pred)

    mse = float(np.mean((y_hat - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_hat - y)))
    corr = pearson_corr_np(y_hat, y)

    mean_baseline = float(y.mean())
    baseline_pred = np.full_like(y, fill_value=mean_baseline)

    baseline_mse = float(np.mean((baseline_pred - y) ** 2))
    baseline_rmse = float(np.sqrt(baseline_mse))
    baseline_mae = float(np.mean(np.abs(baseline_pred - y)))

    explained_variance = 1.0 - mse / (np.var(y) + 1e-8)

    print("\n=== Global metrics ===")
    print(f"N transitions:       {len(y)}")
    print(f"MSE:                 {mse:.6f}")
    print(f"RMSE:                {rmse:.6f}")
    print(f"MAE:                 {mae:.6f}")
    print(f"Pearson corr:        {corr:.6f}")
    print(f"Explained variance:  {explained_variance:.6f}")

    print("\n=== Mean-reward baseline ===")
    print(f"Baseline mean r:     {mean_baseline:.6f}")
    print(f"Baseline MSE:        {baseline_mse:.6f}")
    print(f"Baseline RMSE:       {baseline_rmse:.6f}")
    print(f"Baseline MAE:        {baseline_mae:.6f}")

    if mse < baseline_mse:
        print(f"\nModel beats mean baseline: yes, improvement={baseline_mse - mse:.6f}")
    else:
        print(f"\nModel beats mean baseline: no, worse_by={mse - baseline_mse:.6f}")

    describe_array("True rewards", y)
    describe_array("Predicted rewards", y_hat)
    describe_array("Prediction error", y_hat - y)

    print("\n=== Per-episode returns, first 20 ===")
    for row in rows[:20]:
        print(
            f"{row['file']} | "
            f"true={row['true_return']:8.2f} | "
            f"pred={row['pred_return']:8.2f} | "
            f"err={row['return_error']:8.2f} | "
            f"mse={row['mse']:.4f} | "
            f"corr={row['corr']:.4f}"
        )

    save_csv = Path(args.save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(save_csv, "w") as f:
        f.write("file,T,mse,mae,corr,true_return,pred_return,return_error\n")
        for row in rows:
            f.write(
                f"{row['file']},{row['T']},{row['mse']},{row['mae']},"
                f"{row['corr']},{row['true_return']},{row['pred_return']},"
                f"{row['return_error']}\n"
            )

    print(f"\nSaved per-episode CSV to {save_csv}")


if __name__ == "__main__":
    main()
