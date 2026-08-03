from pathlib import Path
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from data.latent_state_dataset import LatentStateDataset
from world_model.api import LatentState
from world_model.reward_model import RewardClassifierExpectedReward
from world_model.model import load_legacy_world_model
from dreamer import (
    Actor,
    Value,
    imagine_rollout,
    compute_behavior_losses,
    lambda_returns,
)

class ZeroValue(torch.nn.Module):
    def forward(self, features):
        return torch.zeros(features.shape[0], device=features.device)


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


def mc_returns(rewards, gamma):
    """
    rewards: [H, B]
    returns[t] = r_t + gamma r_{t+1} + ...
    """
    H, B = rewards.shape
    returns = []
    running = torch.zeros(B, device=rewards.device)

    for t in reversed(range(H)):
        running = rewards[t] + gamma * running
        returns.append(running)

    returns.reverse()
    return torch.stack(returns, dim=0)


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


def orient_time_batch(x, horizon, name):
    x = torch.as_tensor(x)

    if x.ndim >= 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.shape[0] == horizon:
        return x

    if x.ndim >= 2 and x.shape[1] == horizon:
        return x.transpose(0, 1)

    raise ValueError(f"Cannot orient {name}, shape={tuple(x.shape)}, horizon={horizon}")


def freeze_module(module):
    if hasattr(module, "parameters"):
        for p in module.parameters():
            p.requires_grad_(False)


def save_behavior_checkpoint(
    path,
    actor,
    value,
    actor_opt,
    value_opt,
    step,
    args,
):
    ckpt = {
        "mode": args.mode,
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "actor_opt_state_dict": actor_opt.state_dict(),
        "args": vars(args),
    }

    if value is not None:
        ckpt["value_state_dict"] = value.state_dict()

    if value_opt is not None:
        ckpt["value_opt_state_dict"] = value_opt.state_dict()

    torch.save(ckpt, path)



@torch.no_grad()
def decode_z_batch_to_uint8(world_model, z_batch):
    """
    Decodifica un batch di latenti z usando il decoder del VAE.

    Input:
        z_batch: [N, z_dim]

    Output:
        immagini uint8 [N, H, W, 3]
    """
    x = world_model.vae.decode(z_batch)

    if isinstance(x, tuple):
        x = x[0]

    x = x.detach()

    if x.ndim != 4:
        raise ValueError(f"VAE decode output must be 4D, got shape {tuple(x.shape)}")

    # Caso PyTorch standard: [N, 3, H, W]
    if x.shape[1] == 3:
        x = x.permute(0, 2, 3, 1)

    # Se il decoder restituisse già [N, H, W, 3], non facciamo nulla.
    elif x.shape[-1] == 3:
        pass

    else:
        raise ValueError(f"Cannot interpret decoded image shape: {tuple(x.shape)}")

    # Se il decoder usa tanh e produce valori in [-1, 1], li riportiamo in [0, 1].
    if float(x.min()) < 0.0:
        x = (x + 1.0) / 2.0

    x = x.clamp(0.0, 1.0)

    return (x.cpu().numpy() * 255.0).astype(np.uint8)


@torch.no_grad()
def save_imagined_rollout_grid(
    world_model,
    rollout,
    out_path,
    num_rows=8,
    frame_stride=1,
    cell_size=96,
    z_dim=32,
):
    """
    Salva una tabella PNG di alcuni rollout immaginati.

    Ogni riga è un rollout del batch.
    Ogni colonna è uno step temporale.
    Ogni immagine è ottenuta decodificando z con il VAE.
    """
    features = rollout.features.detach()  # [H + 1, B, feature_dim]

    num_times, batch_size, feature_dim = features.shape

    if z_dim > feature_dim:
        raise ValueError(
            f"z_dim={z_dim} maggiore di feature_dim={feature_dim}"
        )

    rows = min(num_rows, batch_size)
    time_indices = list(range(0, num_times, frame_stride))

    # Prendiamo i primi `rows` rollout del batch.
    rollout_indices = list(range(rows))

    # features[t, b, :z_dim] contiene z_t se feature = concat(z, h).
    z_list = []

    for t in time_indices:
        for b in rollout_indices:
            z = features[t, b, :z_dim]
            z_list.append(z)

    z_batch = torch.stack(z_list, dim=0).to(features.device)  # [T * rows, z_dim]

    decoded = decode_z_batch_to_uint8(world_model, z_batch)

    label_w = 90
    header_h = 28
    reward_h = 18

    cols = len(time_indices)
    canvas_w = label_w + cols * cell_size
    canvas_h = header_h + rows * (cell_size + reward_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Header temporale.
    for col, t in enumerate(time_indices):
        x = label_w + col * cell_size
        draw.text((x + 4, 6), f"t={t}", fill=(0, 0, 0))

    rewards = rollout.rewards.detach().cpu() if rollout.rewards is not None else None

    for row, b in enumerate(rollout_indices):
        y = header_h + row * (cell_size + reward_h)

        draw.text((6, y + cell_size // 2 - 8), f"rollout {b}", fill=(0, 0, 0))

        for col, t in enumerate(time_indices):
            img_idx = col * rows + row

            img = Image.fromarray(decoded[img_idx]).convert("RGB")
            img = img.resize((cell_size, cell_size), resample=Image.NEAREST)

            x = label_w + col * cell_size
            canvas.paste(img, (x, y))

            # Reward associato al frame.
            # t=0 è lo stato iniziale, quindi non ha reward precedente.
            if rewards is not None and t > 0 and t - 1 < rewards.shape[0]:
                r = float(rewards[t - 1, b].reshape(-1)[0])
                draw.text((x + 4, y + cell_size + 1), f"r={r:+.2f}", fill=(0, 0, 0))
            else:
                draw.text((x + 4, y + cell_size + 1), "r= --", fill=(0, 0, 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    print(f"saved rollout debug grid: {out_path}", flush=True)




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["mc", "critic"])

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--latent-pattern", type=str, default="*.npz")

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--updates", type=int, default=3000)

    parser.add_argument("--actor-lr", type=float, default=8e-5)
    parser.add_argument("--value-lr", type=float, default=1e-4)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_", type=float, default=0.95)

    parser.add_argument("--entropy-scale", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=100.0)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--debug-rollout-every", type=int, default=0)
    parser.add_argument("--debug-rollout-rows", type=int, default=8)
    parser.add_argument("--debug-rollout-frame-stride", type=int, default=1)
    parser.add_argument("--debug-rollout-cell-size", type=int, default=96)
    parser.add_argument("--debug-z-dim", type=int, default=32)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("device:", device)
    
    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )

    freeze_module(world_model)

    # Utile su CUDA per backward attraverso RNN cuDNN.
    world_model.train()

    actor = Actor(feature_dim=world_model.feature_dim).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    if args.mode == "critic":
        value = Value(feature_dim=world_model.feature_dim).to(device)
        value_opt = torch.optim.Adam(value.parameters(), lr=args.value_lr)
    else:
        value = ZeroValue().to(device)
        value_opt = None

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


    # Reward classifier override
    if getattr(args, "reward_classifier_ckpt", None) is not None:
        print("Using reward classifier expected reward:", args.reward_classifier_ckpt)
        reward_model = RewardClassifierExpectedReward(args.reward_classifier_ckpt).to(device)
        reward_model.eval()
        for p in reward_model.parameters():
            p.requires_grad_(False)

    for step in range(1, args.updates + 1):
        batch, iterator = next_batch(loader, iterator)
        start_state = make_start_state(batch, device)

        # ============================================================
        # ACTOR UPDATE
        # ============================================================

        actor.train()
        set_requires_grad(actor, True)

        if args.mode == "critic":
            value.eval()
            set_requires_grad(value, False)

        actor_opt.zero_grad(set_to_none=True)

        rollout = imagine_rollout(
            world_model=world_model,
            actor=actor,
            value=value,
            start_state=start_state,
            horizon=args.horizon,
            deterministic=False,
        )

        if args.debug_rollout_every > 0:
            if step == 1 or step % args.debug_rollout_every == 0:
                save_imagined_rollout_grid(
                    world_model=world_model,
                    rollout=rollout,
                    out_path=out_dir / "debug_rollouts" / f"step_{step:06d}.png",
                    num_rows=args.debug_rollout_rows,
                    frame_stride=args.debug_rollout_frame_stride,
                    cell_size=args.debug_rollout_cell_size,
                    z_dim=args.debug_z_dim,
                )

        if args.mode == "critic":
            actor_losses = compute_behavior_losses(
                rollout=rollout,
                lambda_=args.lambda_,
                entropy_scale=args.entropy_scale,
            )

            actor_loss = actor_losses.actor_loss
            mean_return = actor_losses.mean_return
            mean_reward = actor_losses.mean_reward
            mean_entropy = actor_losses.mean_entropy
            mean_value = actor_losses.mean_value

        elif args.mode == "mc":
            rewards = orient_time_batch(
                rollout.rewards,
                args.horizon,
                "rewards",
            ).to(device)

            returns = mc_returns(
                rewards,
                gamma=args.gamma,
            )

            entropies = getattr(rollout, "entropies", None)

            if entropies is not None:
                entropies = orient_time_batch(
                    entropies,
                    args.horizon,
                    "entropies",
                ).to(device)
                entropy_bonus = entropies.mean()
            else:
                entropy_bonus = torch.tensor(0.0, device=device)

            actor_loss = -returns.mean() - args.entropy_scale * entropy_bonus

            mean_return = returns.mean()
            mean_reward = rewards.mean()
            mean_entropy = entropy_bonus
            mean_value = torch.tensor(0.0, device=device)

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        actor_loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)

        actor_opt.step()

        # ============================================================
        # VALUE UPDATE: solo critic mode
        # ============================================================

        if args.mode == "critic":
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

        else:
            value_loss = torch.tensor(0.0, device=device)

        # ============================================================
        # LOGGING
        # ============================================================

        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:06d} | "
                f"mode={args.mode} | "
                f"actor_loss={actor_loss.item(): .4f} | "
                f"value_loss={value_loss.item(): .4f} | "
                f"mean_return={mean_return.item(): .4f} | "
                f"mean_reward={mean_reward.item(): .4f} | "
                f"mean_value={mean_value.item(): .4f} | "
                f"mean_entropy={mean_entropy.item(): .4f}",
                flush=True,
            )

        # ============================================================
        # CHECKPOINT
        # ============================================================

        if step % args.save_every == 0 or step == args.updates:
            ckpt_path = out_dir / "behavior.pt"

            save_behavior_checkpoint(
                path=ckpt_path,
                actor=actor,
                value=value if args.mode == "critic" else None,
                actor_opt=actor_opt,
                value_opt=value_opt,
                step=step,
                args=args,
            )

            print(f"saved checkpoint to {ckpt_path}")

    
    print("training completed")


if __name__ == "__main__":
    main()
