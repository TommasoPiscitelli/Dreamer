from pathlib import Path
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dreamer_carracing.data.latent_state_dataset import LatentStateDataset
from dreamer_carracing.world_model.api import LatentState
from dreamer_carracing.world_model.load_legacy import load_legacy_world_model
from dreamer_carracing.dreamer import (
    Actor,
    Value,
    imagine_rollout,
    compute_behavior_losses,
    lambda_returns,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_requires_grad(module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def next_batch(loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def make_start_state(batch, device):
    return LatentState(
        z=batch["z"].to(device),
        h=batch["h"].to(device).unsqueeze(0),
        c=batch["c"].to(device).unsqueeze(0),
    )


def compute_value_loss_from_rollout(value, rollout, lambda_):
    """
    Train V(s_t) to match lambda returns computed from an imagined rollout.
    The rollout is treated as fixed data for the value update.
    """
    with torch.no_grad():
        targets = lambda_returns(
            rewards=rollout.rewards,
            discounts=rollout.discounts,
            values=rollout.values,
            lambda_=lambda_,
        ).detach()

        features = rollout.features[:-1].detach()

    horizon, batch_size, feature_dim = features.shape

    pred = value(features.reshape(horizon * batch_size, feature_dim))
    pred = pred.reshape(horizon, batch_size, 1)

    return F.mse_loss(pred, targets), targets


def save_checkpoint(path, actor, value, actor_opt, value_opt, step, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "actor_state_dict": actor.state_dict(),
            "value_state_dict": value.state_dict(),
            "actor_optimizer_state_dict": actor_opt.state_dict(),
            "value_optimizer_state_dict": value_opt.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/latent/train")

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument(
        "--reward-ckpt",
        type=str,
        default="checkpoints/reward_model/reward_model.pt",
    )
    parser.add_argument(
        "--reward-calibration",
        type=str,
        default="checkpoints/reward_model/reward_calibration_bias_train.json",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints/dreamer_behavior",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--updates", type=int, default=2000)

    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--value-lr", type=float, default=1e-4)

    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--entropy-scale", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=100.0)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("device:", device)
    
    #Load world model
    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )

    world_model.eval()
    set_requires_grad(world_model, False)

    # Action model and Value model
    actor = Actor(feature_dim=world_model.feature_dim).to(device)
    value = Value(feature_dim=world_model.feature_dim).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    value_opt = torch.optim.Adam(value.parameters(), lr=args.value_lr)

    # Load latent data
    dataset = LatentStateDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    
    iterator = iter(loader)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("num latent states:", len(dataset))
    print("batch size:", args.batch_size)
    print("horizon:", args.horizon)
    print("updates:", args.updates)

    for step in range(1, args.updates + 1):
        batch, iterator = next_batch(loader, iterator)
        start_state = make_start_state(batch, device)

        # -------------------------
        # Actor update
        # -------------------------
        actor.train()
        value.eval()

        set_requires_grad(actor, True)
        set_requires_grad(value, False)

        actor_opt.zero_grad(set_to_none=True)
         
        world_model.train()

        rollout = imagine_rollout(
            world_model=world_model,
            actor=actor,
            value=value,
            start_state=start_state,
            horizon=args.horizon,
            deterministic=False,
        )

        actor_losses = compute_behavior_losses(
            rollout=rollout,
            lambda_=args.lambda_,
            entropy_scale=args.entropy_scale,
        )

        actor_losses.actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
        actor_opt.step()

        # -------------------------
        # Value update
        # -------------------------
        actor.eval()
        value.train()

        set_requires_grad(actor, False)
        set_requires_grad(value, True)

        value_opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            rollout_for_value = imagine_rollout(
                world_model=world_model,
                actor=actor,
                value=value,
                start_state=start_state,
                horizon=args.horizon,
                deterministic=False,
            )

        value_loss, value_targets = compute_value_loss_from_rollout(
            value=value,
            rollout=rollout_for_value,
            lambda_=args.lambda_,
        )

        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), args.grad_clip)
        value_opt.step()

        # -------------------------
        # Logging
        # -------------------------
        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:06d} | "
                f"actor_loss={actor_losses.actor_loss.item(): .4f} | "
                f"value_loss={value_loss.item(): .4f} | "
                f"mean_return={actor_losses.mean_return.item(): .4f} | "
                f"mean_reward={actor_losses.mean_reward.item(): .4f} | "
                f"mean_value={actor_losses.mean_value.item(): .4f} | "
                f"mean_entropy={actor_losses.mean_entropy.item(): .4f}"
            )

        # -------------------------
        # Checkpoint
        # -------------------------
        if step % args.save_every == 0 or step == args.updates:
            ckpt_path = out_dir / "actor_value.pt"
            save_checkpoint(
                path=ckpt_path,
                actor=actor,
                value=value,
                actor_opt=actor_opt,
                value_opt=value_opt,
                step=step,
                args=args,
            )
            print(f"saved checkpoint to {ckpt_path}")

    save_checkpoint(
        path=out_dir / "actor_value_final.pt",
        actor=actor,
        value=value,
        actor_opt=actor_opt,
        value_opt=value_opt,
        step=args.updates,
        args=args,
    )

    print("training completed")


if __name__ == "__main__":
    main()
