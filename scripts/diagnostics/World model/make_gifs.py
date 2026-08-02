import argparse
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

from world_model.load_legacy import load_legacy_world_model
from world_model.api import LatentState
from dreamer.actor_critic import Actor
from dreamer.evo_controller import EvoController


# ---------------------------------------------------------------------
# Latent state utilities
# ---------------------------------------------------------------------

def make_state(z, h, c, like_state=None):
    try:
        return LatentState(
            z=z,
            h=h,
            c=c,
            extra=getattr(like_state, "extra", None),
        )
    except TypeError:
        return LatentState(z=z, h=h, c=c)


def clone_state(state):
    return make_state(
        z=state.z.clone(),
        h=state.h.clone(),
        c=state.c.clone(),
        like_state=state,
    )


def get_next_state(imagine_out):
    if hasattr(imagine_out, "next_state"):
        return imagine_out.next_state
    if hasattr(imagine_out, "state"):
        return imagine_out.state
    if isinstance(imagine_out, tuple):
        return imagine_out[0]
    return imagine_out


def state_features(state):
    if hasattr(state, "features"):
        return state.features

    h = state.h

    if h.ndim == 3:
        h = h[-1]

    if h.ndim == 1:
        h = h.unsqueeze(0)

    return torch.cat([state.z, h], dim=-1)


def load_initial_state_from_npz(path: Path, t: int, device):
    data = np.load(path)

    z_all = data["z"]

    if "h_next" in data:
        h_all = data["h_next"]
    elif "h" in data:
        h_all = data["h"]
    else:
        raise KeyError(f"No h/h_next in {path}. Keys: {list(data.keys())}")

    if "c_next" in data:
        c_all = data["c_next"]
    elif "c" in data:
        c_all = data["c"]
    else:
        raise KeyError(f"No c/c_next in {path}. Keys: {list(data.keys())}")

    z = torch.as_tensor(z_all[t], dtype=torch.float32, device=device).view(1, -1)

    idx = max(t - 1, 0)

    h = torch.as_tensor(h_all[idx], dtype=torch.float32, device=device)
    c = torch.as_tensor(c_all[idx], dtype=torch.float32, device=device)

    if h.ndim == 1:
        h = h.view(1, 1, -1)
    elif h.ndim == 2:
        h = h.unsqueeze(1)

    if c.ndim == 1:
        c = c.view(1, 1, -1)
    elif c.ndim == 2:
        c = c.unsqueeze(1)

    return make_state(z=z, h=h, c=c)


def sample_start(latent_files, min_t=1):
    while True:
        path = random.choice(latent_files)
        data = np.load(path)
        T = len(data["z"])

        if T > 2:
            t = 0
            return path, t


# ---------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------

def build_actor_robust(device):
    attempts = [
        lambda: Actor(288, 3),
        lambda: Actor(288, 256, 3),
        lambda: Actor(288, [256, 256], 3),
        lambda: Actor(feature_dim=288, action_dim=3),
        lambda: Actor(obs_dim=288, action_dim=3),
        lambda: Actor(state_dim=288, action_dim=3),
        lambda: Actor(input_size=288, action_size=3),
        lambda: Actor(features_dim=288, action_dim=3),
    ]

    errors = []

    for make_actor in attempts:
        try:
            actor = make_actor().to(device)
            print("Built Actor successfully.")
            return actor
        except Exception as e:
            errors.append(str(e))

    raise RuntimeError("Could not build Actor.\n" + "\n".join(errors))


def load_actor(actor_ckpt: Path, device):
    actor = build_actor_robust(device)

    ckpt = torch.load(actor_ckpt, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "actor" in ckpt:
            state_dict = ckpt["actor"]
        elif "actor_state_dict" in ckpt:
            state_dict = ckpt["actor_state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unsupported actor checkpoint format: {type(ckpt)}")

    missing, unexpected = actor.load_state_dict(state_dict, strict=False)

    if missing:
        print("Warning: missing actor keys:")
        for k in missing:
            print("  ", k)

    if unexpected:
        print("Warning: unexpected actor keys:")
        for k in unexpected:
            print("  ", k)

    actor.eval()
    return actor


def load_evo_controller(controller_json: Path, device):
    try:
        controller = EvoController.from_json(controller_json, device=device)
    except TypeError:
        controller = EvoController.from_json(controller_json)

    if hasattr(controller, "to"):
        controller = controller.to(device)

    if hasattr(controller, "eval"):
        controller.eval()

    return controller


def load_policy(args, device):
    if args.policy_type in ["dreamer", "dreamer_mc", "actor"]:
        if args.actor_ckpt is None:
            raise ValueError(f"--actor-ckpt is required for policy_type={args.policy_type}")
        return load_actor(Path(args.actor_ckpt), device)

    if args.policy_type == "evo":
        if args.controller_json is None:
            raise ValueError("--controller-json is required for policy_type=evo")
        return load_evo_controller(Path(args.controller_json), device)

    if args.policy_type in ["random", "forward", "heuristic"]:
        return None

    raise ValueError(f"Unknown policy_type: {args.policy_type}")


# ---------------------------------------------------------------------
# Action utilities
# ---------------------------------------------------------------------

def extract_tensor_from_output(out):
    if isinstance(out, torch.Tensor):
        return out

    if hasattr(out, "mean") and isinstance(out.mean, torch.Tensor):
        return out.mean

    if hasattr(out, "loc") and isinstance(out.loc, torch.Tensor):
        return out.loc

    if isinstance(out, (tuple, list)):
        for item in out:
            try:
                return extract_tensor_from_output(item)
            except Exception:
                pass

    if isinstance(out, dict):
        for key in ["action", "mean", "raw_action", "loc"]:
            if key in out and isinstance(out[key], torch.Tensor):
                return out[key]

    raise TypeError(f"Could not extract action tensor from output type {type(out)}")


def map_raw_to_carracing_action(raw):
    steer = torch.tanh(raw[:, 0])
    gas = (torch.tanh(raw[:, 1]) + 1.0) / 2.0
    brake = torch.clamp(torch.tanh(raw[:, 2]), 0.0, 1.0)
    return torch.stack([steer, gas, brake], dim=-1)


def clamp_carracing_action(action):
    action = action.clone()
    action[:, 0] = action[:, 0].clamp(-1.0, 1.0)
    action[:, 1] = action[:, 1].clamp(0.0, 1.0)
    action[:, 2] = action[:, 2].clamp(0.0, 1.0)
    return action


def ensure_valid_action(action):
    if action.ndim == 1:
        action = action.unsqueeze(0)

    invalid = (
        (action[:, 0].min() < -1.05)
        or (action[:, 0].max() > 1.05)
        or (action[:, 1].min() < -0.05)
        or (action[:, 1].max() > 1.05)
        or (action[:, 2].min() < -0.05)
        or (action[:, 2].max() > 1.05)
    )

    if invalid:
        return map_raw_to_carracing_action(action)

    return clamp_carracing_action(action)


def actor_policy_action(actor, features):
    with torch.no_grad():
        action = None

        for method_name in ["deterministic_action", "act", "get_action"]:
            if hasattr(actor, method_name):
                method = getattr(actor, method_name)

                try:
                    out = method(features, deterministic=True)
                    action = extract_tensor_from_output(out)
                    break
                except TypeError:
                    try:
                        out = method(features)
                        action = extract_tensor_from_output(out)
                        break
                    except Exception:
                        pass
                except Exception:
                    pass

        if action is None:
            out = actor(features)
            action = extract_tensor_from_output(out)

        return ensure_valid_action(action)


def evo_policy_action(controller, features):
    with torch.no_grad():
        out = controller(features)
        action = extract_tensor_from_output(out)
        return ensure_valid_action(action)


def scripted_policy_action(policy_type, device):
    if policy_type == "random":
        steer = torch.empty(1, device=device).uniform_(-1.0, 1.0)
        gas = torch.empty(1, device=device).uniform_(0.0, 1.0)
        brake = torch.empty(1, device=device).uniform_(0.0, 0.4)
        return torch.stack([steer, gas, brake], dim=-1)

    if policy_type == "forward":
        return torch.tensor([[0.0, 0.7, 0.0]], dtype=torch.float32, device=device)

    if policy_type == "heuristic":
        # Heuristic minimale nel world model: va avanti, sterza poco e non frena.
        # Se vuoi una heuristic più sofisticata, questo è il punto da modificare.
        return torch.tensor([[0.0, 0.55, 0.0]], dtype=torch.float32, device=device)

    raise ValueError(policy_type)


def policy_action(policy_type, policy, features, device):
    if policy_type in ["dreamer", "dreamer_mc", "actor"]:
        return actor_policy_action(policy, features)

    if policy_type == "evo":
        return evo_policy_action(policy, features)

    if policy_type in ["random", "forward", "heuristic"]:
        return scripted_policy_action(policy_type, device)

    raise ValueError(f"Unknown policy_type: {policy_type}")


# ---------------------------------------------------------------------
# Decode latent state to image
# ---------------------------------------------------------------------

def decode_z_to_frame(world_model, z):
    with torch.no_grad():
        decoded = None
        candidates = []

        if hasattr(world_model, "decode_z"):
            candidates.append(lambda: world_model.decode_z(z))

        if hasattr(world_model, "decode"):
            candidates.append(lambda: world_model.decode(z))

        if hasattr(world_model, "vae"):
            vae = world_model.vae

            if hasattr(vae, "decode"):
                candidates.append(lambda: vae.decode(z))

            if hasattr(vae, "decoder"):
                candidates.append(lambda: vae.decoder(z))

        last_error = None

        for fn in candidates:
            try:
                decoded = fn()
                break
            except Exception as e:
                last_error = e

        if decoded is None:
            raise RuntimeError(f"Could not decode z into image. Last error: {last_error}")

        if isinstance(decoded, tuple):
            decoded = decoded[0]

        if isinstance(decoded, dict):
            for key in ["obs", "recon", "x", "image"]:
                if key in decoded:
                    decoded = decoded[key]
                    break

        if not isinstance(decoded, torch.Tensor):
            raise TypeError(f"Decoded output is not a tensor: {type(decoded)}")

        x = decoded.detach().cpu()

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 3 and x.shape[0] in [1, 3, 4]:
        x = x.permute(1, 2, 0)

    arr = x.numpy()

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.max() <= 1.5:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)

    return arr


def save_gif(frames, out_gif, fps=20):
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=fps)



# ---------------------------------------------------------------------
# Reward utilities
# ---------------------------------------------------------------------

def maybe_replace_reward_with_classifier(world_model, reward_classifier_ckpt, device):
    """
    If a reward-classifier checkpoint is provided, replace the legacy reward
    model with RewardClassifierExpectedReward.
    """
    if reward_classifier_ckpt is None:
        return world_model

    from dreamer.reward_classifier_expected import RewardClassifierExpectedReward

    reward_model = RewardClassifierExpectedReward(reward_classifier_ckpt).to(device)
    reward_model.eval()

    for p in reward_model.parameters():
        p.requires_grad_(False)

    if hasattr(world_model, "reward_model"):
        world_model.reward_model = reward_model
    elif hasattr(world_model, "reward"):
        world_model.reward = reward_model
    else:
        raise RuntimeError(
            "Non trovo né world_model.reward_model né world_model.reward."
        )

    print("Using reward classifier expected reward:", reward_classifier_ckpt)
    return world_model


def evaluate_reward_model_on_state(world_model, state):
    """
    Explicitly evaluate the reward model on the reached latent state.

    This avoids ambiguity with rewards possibly returned by imagine_step.
    If world_model.reward_model has been replaced by the classifier wrapper,
    this function uses the new classifier reward.
    """
    features = state_features(state)

    if hasattr(world_model, "reward_model"):
        reward_model = world_model.reward_model
    elif hasattr(world_model, "reward"):
        reward_model = world_model.reward
    else:
        raise RuntimeError(
            "Non trovo né world_model.reward_model né world_model.reward."
        )

    reward = reward_model(features)

    if isinstance(reward, tuple):
        reward = reward[0]

    if isinstance(reward, dict):
        for key in ["reward", "expected_reward", "pred_reward", "mean"]:
            if key in reward:
                reward = reward[key]
                break

    if not isinstance(reward, torch.Tensor):
        raise TypeError(f"Reward model output non Tensor: {type(reward)}")

    return reward.reshape(-1)


# ---------------------------------------------------------------------
# Generic policy rollout in world model
# ---------------------------------------------------------------------

def make_one_policy_world_model_gif(
    episode_idx,
    policy_name,
    policy_type,
    policy,
    world_model,
    latent_files,
    horizon,
    gamma,
    device,
    out_dir,
    fps,
):
    source_path, source_t = sample_start(latent_files)

    print(
        f"{policy_name} rollout {episode_idx}: "
        f"start from {source_path.name}, t={source_t}"
    )

    state = load_initial_state_from_npz(source_path, source_t, device)

    frames = []
    rewards = []
    per_step_rows = []

    with torch.no_grad():
        for t in range(horizon):
            frame = decode_z_to_frame(world_model, state.z)
            frames.append(frame)

            features = state_features(state)

            action = policy_action(
                policy_type=policy_type,
                policy=policy,
                features=features,
                device=device,
            )

            action_np = action.detach().cpu().numpy()[0].astype(np.float32)

            # World model transition:
            # state_t, action_t -> state_{t+1}
            imagine_out = world_model.imagine_step(state, action)
            next_state = get_next_state(imagine_out)

            # Reward explicitly evaluated on the reached latent state.
            reward_t = evaluate_reward_model_on_state(world_model, next_state)
            reward = float(reward_t.detach().cpu().reshape(-1)[0])

            rewards.append(reward)

            per_step_rows.append({
                "rollout_id": episode_idx,
                "t": t,
                "reward": reward,
                "steer": float(action_np[0]),
                "gas": float(action_np[1]),
                "brake": float(action_np[2]),
            })

            state = next_state

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    discounts = np.asarray(
        [gamma ** t for t in range(len(rewards_arr))],
        dtype=np.float32,
    )

    undiscounted_return = float(rewards_arr.sum())
    discounted_return = float((rewards_arr * discounts).sum())

    rollout_row = {
        "rollout_id": episode_idx,
        "source_file": source_path.name,
        "source_t": int(source_t),
        "horizon": int(len(rewards_arr)),
        "discounted_return": discounted_return,
        "undiscounted_return": undiscounted_return,
    }

    safe_name = policy_name.lower().replace(" ", "_").replace("-", "_")
    out_gif = out_dir / f"{safe_name}_world_model_episode_{episode_idx:03d}.gif"

    save_gif(frames, out_gif, fps=fps)

    print(
        f"Saved: {out_gif} | "
        f"discounted_return={discounted_return:.3f} | "
        f"undiscounted_return={undiscounted_return:.3f}"
    )

    return rollout_row, per_step_rows



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy-type",
        type=str,
        required=True,
        choices=["dreamer", "dreamer_mc", "actor", "evo", "random", "forward", "heuristic"],
        help="Policy da far evolvere nel world model.",
    )

    parser.add_argument(
        "--policy-name",
        type=str,
        default=None,
        help="Nome usato nei file di output. Se omesso, usa policy-type.",
    )

    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--latent-pattern", type=str, default="*.npz")

    parser.add_argument("--actor-ckpt", type=str, default=None)
    parser.add_argument("--controller-json", type=str, default=None)

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)
    parser.add_argument("--reward-classifier-ckpt", type=str, default=None)

    parser.add_argument("--num-gifs", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--csv-prefix", type=str, default="actor_world_model")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_files = sorted(Path(args.latent_dir).glob(args.latent_pattern))
    if len(latent_files) == 0:
        raise FileNotFoundError(
            f"No files found in {args.latent_dir} with pattern {args.latent_pattern}"
        )

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )

    world_model = maybe_replace_reward_with_classifier(
        world_model=world_model,
        reward_classifier_ckpt=args.reward_classifier_ckpt,
        device=device,
    )

    world_model.eval()

    policy = load_policy(args, device)

    policy_name = args.policy_name
    if policy_name is None:
        policy_name = args.policy_type

    rollout_rows = []
    per_step_rows = []

    for i in range(args.num_gifs):
        rollout_row, step_rows = make_one_policy_world_model_gif(
            episode_idx=i,
            policy_name=policy_name,
            policy_type=args.policy_type,
            policy=policy,
            world_model=world_model,
            latent_files=latent_files,
            horizon=args.horizon,
            gamma=args.gamma,
            device=device,
            out_dir=out_dir,
            fps=args.fps,
        )

        rollout_rows.append(rollout_row)
        per_step_rows.extend(step_rows)

    rollout_csv = out_dir / f"{args.csv_prefix}_rollout_returns.csv"
    per_step_csv = out_dir / f"{args.csv_prefix}_per_step.csv"

    pd.DataFrame(
        rollout_rows,
        columns=[
            "rollout_id",
            "source_file",
            "source_t",
            "horizon",
            "discounted_return",
            "undiscounted_return",
        ],
    ).to_csv(rollout_csv, index=False)

    pd.DataFrame(
        per_step_rows,
        columns=[
            "rollout_id",
            "t",
            "reward",
            "steer",
            "gas",
            "brake",
        ],
    ).to_csv(per_step_csv, index=False)

    print()
    print("Saved CSV:")
    print(f"  {rollout_csv}")
    print(f"  {per_step_csv}")


if __name__ == "__main__":
    main()
