import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dreamer_carracing.data.latent_reward_dataset import LatentRewardDataset
from dreamer_carracing.world_model.reward_model import RewardModel


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten()
    y = y.flatten()

    x = x - x.mean()
    y = y - y.mean()

    denom = x.norm() * y.norm()
    if denom.item() == 0:
        return torch.tensor(0.0, device=x.device)

    return (x @ y) / denom


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_corr = 0.0
    total_batches = 0

    total_true_reward = 0.0
    total_pred_reward = 0.0
    total_count = 0

    for batch in loader:
        features = batch["features"].to(device)
        reward = batch["reward"].to(device)

        pred = model(features)

        loss = F.mse_loss(pred, reward)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            corr = pearson_corr(pred, reward)

            total_true_reward += reward.sum().item()
            total_pred_reward += pred.sum().item()
            total_count += reward.numel()

        total_loss += loss.item()
        total_corr += corr.item()
        total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "corr": total_corr / total_batches,
        "mean_true_reward": total_true_reward / total_count,
        "mean_pred_reward": total_pred_reward / total_count,
    }


def save_checkpoint(model, path: str | Path, epoch: int, metrics: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "feature_dim": 288,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/latent/train")
    parser.add_argument("--out-ckpt", type=str, default="checkpoints/reward_model/reward_model.pt")

    parser.add_argument("--feature-dim", type=int, default=288)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = LatentRewardDataset(args.data_dir)

    val_size = int(len(dataset) * args.val_frac)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"dataset size: {len(dataset)}")
    print(f"train size:   {len(train_dataset)}")
    print(f"val size:     {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = RewardModel(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
            )

        print(
            f"epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.6f} "
            f"train corr={train_metrics['corr']:.4f} "
            f"train mean r={train_metrics['mean_true_reward']:.4f} "
            f"train mean pred={train_metrics['mean_pred_reward']:.4f} | "
            f"val loss={val_metrics['loss']:.6f} "
            f"val corr={val_metrics['corr']:.4f} "
            f"val mean r={val_metrics['mean_true_reward']:.4f} "
            f"val mean pred={val_metrics['mean_pred_reward']:.4f}",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model=model,
                path=args.out_ckpt,
                epoch=epoch,
                metrics=val_metrics,
            )
            print(f"saved best reward model to {args.out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
