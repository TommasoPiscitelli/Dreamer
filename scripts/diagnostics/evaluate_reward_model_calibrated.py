from pathlib import Path
import argparse
import json

import numpy as np
import torch

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
def predict_file(model, path, device, batch_size, scale, bias):
    data = np.load(path)

    features = data["features"].astype(np.float32)
    rewards = data["reward"].astype(np.float32)

    preds = []

    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start:start + batch_size]).to(device)
        y_hat = model(x).squeeze(-1).cpu().numpy()
        preds.append(y_hat)

    preds = np.concatenate(preds, axis=0)
    preds_cal = scale * preds + bias

    return rewards, preds, preds_cal


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


def global_metrics(y, y_hat):
    mse = float(np.mean((y_hat - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_hat - y)))
    corr = pearson_corr_np(y_hat, y)
    ev = float(1.0 - mse / (np.var(y) + 1e-8))

    return mse, rmse, mae, corr, ev


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/latent/test")
    parser.add_argument("--ckpt", type=str, default="checkpoints/reward_model/reward_model.pt")
    parser.add_argument("--calibration", type=str, default="checkpoints/reward_model/reward_calibration.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-csv", type=str, default="logs/reward_eval_calibrated_per_episode.csv")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    with open(args.calibration, "r") as f:
        calibration = json.load(f)

    scale = float(calibration["scale"])
    bias = float(calibration["bias"])

    print("scale:", scale)
    print("bias:", bias)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    model, ckpt = load_reward_model(args.ckpt, device)

    all_true = []
    all_raw = []
    all_cal = []

    rows = []

    for path in files:
        true_r, pred_raw, pred_cal = predict_file(
            model=model,
            path=path,
            device=device,
            batch_size=args.batch_size,
            scale=scale,
            bias=bias,
        )

        all_true.append(true_r)
        all_raw.append(pred_raw)
        all_cal.append(pred_cal)

        raw_mse = float(np.mean((pred_raw - true_r) ** 2))
        cal_mse = float(np.mean((pred_cal - true_r) ** 2))

        true_return = float(true_r.sum())
        raw_return = float(pred_raw.sum())
        cal_return = float(pred_cal.sum())

        rows.append({
            "file": path.name,
            "T": len(true_r),
            "true_return": true_return,
            "raw_return": raw_return,
            "cal_return": cal_return,
            "raw_error": raw_return - true_return,
            "cal_error": cal_return - true_return,
            "raw_mse": raw_mse,
            "cal_mse": cal_mse,
            "cal_corr": pearson_corr_np(pred_cal, true_r),
        })

    y = np.concatenate(all_true)
    raw = np.concatenate(all_raw)
    cal = np.concatenate(all_cal)

    raw_mse, raw_rmse, raw_mae, raw_corr, raw_ev = global_metrics(y, raw)
    cal_mse, cal_rmse, cal_mae, cal_corr, cal_ev = global_metrics(y, cal)

    print("\n=== Global raw ===")
    print(f"MSE:                 {raw_mse:.6f}")
    print(f"RMSE:                {raw_rmse:.6f}")
    print(f"MAE:                 {raw_mae:.6f}")
    print(f"Pearson corr:        {raw_corr:.6f}")
    print(f"Explained variance:  {raw_ev:.6f}")

    print("\n=== Global calibrated ===")
    print(f"MSE:                 {cal_mse:.6f}")
    print(f"RMSE:                {cal_rmse:.6f}")
    print(f"MAE:                 {cal_mae:.6f}")
    print(f"Pearson corr:        {cal_corr:.6f}")
    print(f"Explained variance:  {cal_ev:.6f}")

    describe_array("True rewards", y)
    describe_array("Raw predicted rewards", raw)
    describe_array("Calibrated predicted rewards", cal)

    print("\n=== Per-episode returns, first 20 ===")
    for row in rows[:20]:
        print(
            f"{row['file']} | "
            f"true={row['true_return']:8.2f} | "
            f"raw={row['raw_return']:8.2f} | "
            f"cal={row['cal_return']:8.2f} | "
            f"raw_err={row['raw_error']:8.2f} | "
            f"cal_err={row['cal_error']:8.2f}"
        )

    save_csv = Path(args.save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(save_csv, "w") as f:
        f.write("file,T,true_return,raw_return,cal_return,raw_error,cal_error,raw_mse,cal_mse,cal_corr\n")
        for row in rows:
            f.write(
                f"{row['file']},{row['T']},{row['true_return']},"
                f"{row['raw_return']},{row['cal_return']},"
                f"{row['raw_error']},{row['cal_error']},"
                f"{row['raw_mse']},{row['cal_mse']},{row['cal_corr']}\n"
            )

    print(f"\nSaved CSV to {save_csv}")


if __name__ == "__main__":
    main()
