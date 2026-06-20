import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dreamer_carracing.data.episode_dataset import CarRacingEpisodeDataset
from dreamer_carracing.world_model.load_legacy import load_legacy_world_model


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten()
    y = y.flatten()

    x = x - x.mean()
    y = y - y.mean()

    denom = x.norm() * y.norm()

    if denom.item() == 0:
        return torch.tensor(0.0, device=x.device)

    return (x @ y) / denom


def train_one_epoch(world_model, loader, optimizer, device, grad_clip: float):
    world_model.vae.eval()
    world_model.mdn_rnn.eval()
    world_model.reward_model.train()

    total_loss = 0.0
    total_corr = 0.0
    total_batches = 0

    for batch in loader:
        obs = batch["obs"].to(device)          # [B, T+1, 3, 64, 64]
        actions = batch["actions"].to(device)  # [B, T, 3]
        rewards = batch["rewards"].to(device)  # [B, T]

        B, T = actions.shape[:2]

        state = world_model.initial_state(B, device)

        pred_rewards = []

        for t in range(T):
            state = world_model.observe_step(
                prev_state=state,
                action=actions[:, t],
                next_obs=obs[:, t + 1],
            )

            pred_r_t = world_model.predict_reward(state).squeeze(-1)  # [B]
            pred_rewards.append(pred_r_t)

        pred_rewards = torch.stack(pred_rewards, dim=1)  # [B, T]

        loss = F.mse_loss(pred_rewards, rewards)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                world_model.reward_model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        with torch.no_grad():
            corr = pearson_corr(pred_rewards, rewards)

        total_loss += loss.item()
        total_corr += corr.item()
        total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "corr": total_corr / total_batches,
    }


@torch.no_grad()
def evaluate(world_model, loader, device):
    world_model.vae.eval()
    world_model.mdn_rnn.eval()
    world_model.reward_model.eval()

    total_loss = 0.0
    total_corr = 0.0
    total_batches = 0

    total_true_return = 0.0
    total_pred_return = 0.0

    for batch in loader:
        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)

        B, T = actions.shape[:2]

        state = world_model.initial_state(B, device)

        pred_rewards = []

        for t in range(T):
            state = world_model.observe_step(
                prev_state=state,
                action=actions[:, t],
                next_obs=obs[:, t + 1],
            )

            pred_r_t = world_model.predict_reward(state).squeeze(-1)
            pred_rewards.append(pred_r_t)

        pred_rewards = torch.stack(pred_rewards, dim=1)

        loss = F.mse_loss(pred_rewards, rewards)
        corr = pearson_corr(pred_rewards, rewards)

        total_loss += loss.item()
        total_corr += corr.item()
        total_batches += 1

        total_true_return += rewards.sum(dim=1).mean().item()
        total_pred_return += pred_rewards.sum(dim=1).mean().item()

    return {
        "loss": total_loss / total_batches,
        "corr": total_corr / total_batches,
        "true_return": total_true_return / total_batches,
        "pred_return": total_pred_return / total_batches,
    }


def save_reward_checkpoint(world_model, path: str | Path, epoch: int, metrics: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": world_model.reward_model.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/raw/train")
    parser.add_argument("--vae-ckpt", type=str, default="checkpoints/legacy/vae/vae.pt")
    parser.add_argument("--mdn-rnn-ckpt", type=str, default="checkpoints/legacy/mdn_rnn/mdn_rnn.pt")
    parser.add_argument("--out-ckpt", type=str, default="checkpoints/reward_model/reward_model.pt")

    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = CarRacingEpisodeDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=None,
        device=device,
    )

    # Train only the reward model.
    for p in world_model.vae.parameters():
        p.requires_grad_(False)

    for p in world_model.mdn_rnn.parameters():
        p.requires_grad_(False)

    for p in world_model.reward_model.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        world_model.reward_model.parameters(),
        lr=args.lr,
    )

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            world_model=world_model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        eval_metrics = evaluate(
            world_model=world_model,
            loader=loader,
            device=device,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.6f} "
            f"train corr={train_metrics['corr']:.4f} | "
            f"eval loss={eval_metrics['loss']:.6f} "
            f"eval corr={eval_metrics['corr']:.4f} | "
            f"true return={eval_metrics['true_return']:.3f} "
            f"pred return={eval_metrics['pred_return']:.3f}"
        )

        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            save_reward_checkpoint(
                world_model=world_model,
                path=args.out_ckpt,
                epoch=epoch,
                metrics=eval_metrics,
            )
            print(f"saved best reward model to {args.out_ckpt}")


if __name__ == "__main__":
    main()
