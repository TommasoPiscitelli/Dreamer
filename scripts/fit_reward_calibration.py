from pathlib import Path
import argparse
import json

import numpy as np
import torch

from dreamer_carracing.world_model.reward_model import RewardModel


def load_reward_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    feature_dim = ckpt.get("feature_dim", 288)

    model = RewardModel(feature_dim=feature_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt


@torch.no_grad()
def predict_all(model, data_dir, device, batch_size):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    all_true = []
    all_pred = []

    for path in files:
        data = np.load(path)
        features = data["features"].astype(np.float32)
        rewards = data["reward"].astype(np.float32)

        preds = []

        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start:start + batch_size]).to(device)
            y_hat = model(x).squeeze(-1).cpu().numpy()
            preds.append(y_hat)

        preds = np.concatenate(preds, axis=0)

        all_true.append(rewards)
        all_pred.append(preds)

    y = np.concatenate(all_true)
    y_hat = np.concatenate(all_pred)

    return y, y_hat


def metrics(y, y_hat):
    mse = float(np.mean((y_hat - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_hat - y)))

    y0 = y - y.mean()
    yhat0 = y_hat - y_hat.mean()
    denom = np.linalg.norm(y0) * np.linalg.norm(yhat0)
    corr = float(np.dot(y0, yhat0) / denom) if denom > 0 else 0.0

    ev = float(1.0 - mse / (np.var(y) + 1e-8))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "explained_variance": ev,
        "mean_true": float(y.mean()),
        "mean_pred": float(y_hat.mean()),
        "std_true": float(y.std()),
        "std_pred": float(y_hat.std()),
    }


def fit_affine_calibration(y_true, y_pred):
    """
    Fits:
        y_true ≈ scale * y_pred + bias
    """
    x = y_pred.reshape(-1)
    y = y_true.reshape(-1)

    A = np.stack([x, np.ones_like(x)], axis=1)
    scale, bias = np.linalg.lstsq(A, y, rcond=None)[0]

    return float(scale), float(bias)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/reward_model/reward_model.pt")
    parser.add_argument("--out", type=str, default="checkpoints/reward_model/reward_calibration.json")
    parser.add_argument("--batch-size", type=int, default=1024)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model, ckpt = load_reward_model(args.ckpt, device)
    y_true, y_pred = predict_all(model, args.data_dir, device, args.batch_size)

    before = metrics(y_true, y_pred)

    scale, bias = fit_affine_calibration(y_true, y_pred)
    y_cal = scale * y_pred + bias

    after = metrics(y_true, y_cal)

    result = {
        "scale": scale,
        "bias": bias,
        "fit_data_dir": args.data_dir,
        "reward_ckpt": args.ckpt,
        "metrics_before": before,
        "metrics_after": after,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Calibration ===")
    print(f"scale: {scale:.8f}")
    print(f"bias:  {bias:.8f}")

    print("\n=== Before ===")
    for k, v in before.items():
        print(f"{k:20s}: {v:.6f}")

    print("\n=== After ===")
    for k, v in after.items():
        print(f"{k:20s}: {v:.6f}")

    print(f"\nSaved calibration to {out_path}")


if __name__ == "__main__":
    main()
