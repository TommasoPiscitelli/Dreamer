from pathlib import Path
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dreamer_carracing.world_model.reward_model import RewardModel


def load_reward_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    feature_dim = ckpt.get("feature_dim", 288)

    model = RewardModel(feature_dim=feature_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_rewards(model, features, device, batch_size=1024):
    preds = []

    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start:start + batch_size]).float().to(device)
        y_hat = model(x).squeeze(-1).cpu().numpy()
        preds.append(y_hat)

    return np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/reward_model/reward_model.pt")
    parser.add_argument("--out-dir", type=str, default="logs/reward_plots")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(args.file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(path)
    features = data["features"].astype(np.float32)
    true_r = data["reward"].astype(np.float32)

    model = load_reward_model(args.ckpt, device)
    pred_r = predict_rewards(model, features, device, args.batch_size)

    true_return = np.cumsum(true_r)
    pred_return = np.cumsum(pred_r)

    print("file:", path)
    print("T:", len(true_r))
    print("true return:", float(true_r.sum()))
    print("pred return:", float(pred_r.sum()))
    print("reward MSE:", float(np.mean((pred_r - true_r) ** 2)))
    print("reward MAE:", float(np.mean(np.abs(pred_r - true_r))))

    t = np.arange(len(true_r))

    plt.figure(figsize=(14, 5))
    plt.plot(t, true_r, label="true reward", linewidth=1.0)
    plt.plot(t, pred_r, label="predicted reward", linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title(f"Step rewards: {path.name}")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{path.stem}_step_rewards.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(t, true_return, label="true cumulative reward", linewidth=1.5)
    plt.plot(t, pred_return, label="predicted cumulative reward", linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.title(f"Cumulative reward: {path.name}")
    plt.legend()
    plt.tight_layout()
    out_path_2 = out_dir / f"{path.stem}_cumulative_rewards.png"
    plt.savefig(out_path_2, dpi=150)
    plt.close()

    print("saved:")
    print(out_path)
    print(out_path_2)


if __name__ == "__main__":
    main()
