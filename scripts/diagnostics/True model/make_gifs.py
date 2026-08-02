import argparse
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from world_model.load_legacy import load_legacy_world_model
from world_model.api import LatentState
from dreamer.actor_critic import Actor
from dreamer.evo_controller import EvoController


# ---------------------------------------------------------------------
# Gym utilities
# ---------------------------------------------------------------------

def import_gym():
    try:
        import gymnasium as gym
        return gym
    except ImportError:
        import gym
        return gym


def reset_env(env, seed):
    out = env.reset(seed=seed)
    if isinstance(out, tuple):
        obs, info = out
    else:
        obs, info = out, {}
    return obs, info


def step_env(env, action):
    out = env.step(action)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out

    return obs, float(reward), bool(done), info


def get_frame(env, obs):
    try:
        frame = env.render()
        if frame is not None:
            return frame
    except Exception:
        pass

    return obs


def normalize_frame(frame):
    frame = np.asarray(frame)

    if frame.ndim == 3 and frame.shape[0] in [1, 3, 4] and frame.shape[-1] not in [1, 3, 4]:
        frame = np.transpose(frame, (1, 2, 0))

    if frame.dtype != np.uint8:
        if np.nanmax(frame) <= 1.5:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)

    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)

    return frame


def save_gif(frames, out_gif, fps=20, skip=1):
    frames = [normalize_frame(f) for f in frames]

    if skip > 1:
        frames = frames[::skip]

    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(out_gif, frames, fps=fps)


# ---------------------------------------------------------------------
# Observation -> latent state
# ---------------------------------------------------------------------

def obs_to_tensor(obs, device):
    obs = np.asarray(obs)

    if obs.ndim != 3:
        raise ValueError(f"Expected obs shape [H, W, C], got {obs.shape}")

    x = torch.from_numpy(obs).to(device)

    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0

    x = x.permute(2, 0, 1).unsqueeze(0)

    if x.shape[-2:] != (64, 64):
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

    return x


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


def get_next_state(imagine_out):
    if hasattr(imagine_out, "next_state"):
        return imagine_out.next_state
    if hasattr(imagine_out, "state"):
        return imagine_out.state
    if isinstance(imagine_out, tuple):
        return imagine_out[0]
    return imagine_out


def encode_real_obs_into_state(world_model, obs, previous_state, device):
    obs_tensor = obs_to_tensor(obs, device)
    z = world_model.encode_obs(obs_tensor).to(device)

    return make_state(
        z=z,
        h=previous_state.h,
        c=previous_state.c,
        like_state=previous_state,
    )


def state_features(state):
    if hasattr(state, "features"):
        return state.features

    h = state.h

    if h.ndim == 3:
        h = h[-1]

    if h.ndim == 1:
        h = h.unsqueeze(0)

    return torch.cat([state.z, h], dim=-1)


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
# Actions
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
# Generic policy rollout in real env
# ---------------------------------------------------------------------

def make_one_policy_real_env_gif(
    episode_idx,
    policy_name,
    policy_type,
    policy,
    world_model,
    env_id,
    seed,
    max_steps,
    gamma,
    device,
    out_dir,
    fps,
    gif_skip,
):
    gym = import_gym()

    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id)

    obs, _ = reset_env(env, seed)

    base_state = world_model.initial_state(batch_size=1, device=device)
    state = encode_real_obs_into_state(world_model, obs, base_state, device)

    frames = []
    rewards = []
    per_step_rows = []

    with torch.no_grad():
        for t in range(max_steps):
            frame = normalize_frame(get_frame(env, obs))
            frames.append(frame)

            features = state_features(state)

            action = policy_action(
                policy_type=policy_type,
                policy=policy,
                features=features,
                device=device,
            )

            action_np = action.detach().cpu().numpy()[0].astype(np.float32)

            # ---------------------------------------------------------
            # TRUE MODEL / REAL ENV TRANSITION:
            # next_obs viene prodotto dall'ambiente reale, non dal world model.
            # ---------------------------------------------------------
            next_obs, reward, done, _ = step_env(env, action_np)

            reward = float(reward)
            rewards.append(reward)

            per_step_rows.append({
                "rollout_id": episode_idx,
                "t": t,
                "reward": reward,
                "steer": float(action_np[0]),
                "gas": float(action_np[1]),
                "brake": float(action_np[2]),
            })

            # Aggiorniamo lo stato ricorrente usando il world model solo per
            # mantenere h,c coerenti con la policy. La transizione osservata
            # resta quella reale, cioè next_obs.
            imagine_out = world_model.imagine_step(state, action)
            predicted_next_state = get_next_state(imagine_out)

            state = encode_real_obs_into_state(
                world_model=world_model,
                obs=next_obs,
                previous_state=predicted_next_state,
                device=device,
            )

            obs = next_obs

            if done:
                break

    env.close()

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    discounts = np.asarray(
        [gamma ** t for t in range(len(rewards_arr))],
        dtype=np.float32,
    )

    undiscounted_return = float(rewards_arr.sum())
    discounted_return = float((rewards_arr * discounts).sum())

    rollout_row = {
        "rollout_id": episode_idx,
        "seed": int(seed),
        "horizon": int(len(rewards_arr)),
        "discounted_return": discounted_return,
        "undiscounted_return": undiscounted_return,
    }

    safe_name = policy_name.lower().replace(" ", "_").replace("-", "_")
    out_gif = out_dir / (
        f"{safe_name}_true_model_episode_{episode_idx:03d}"
        f"_return_{undiscounted_return:.1f}.gif"
    )

    save_gif(frames, out_gif, fps=fps, skip=gif_skip)

    print(
        f"Saved: {out_gif} | "
        f"steps={len(rewards_arr)} | "
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
        help="Policy da far evolvere nel real environment.",
    )

    parser.add_argument(
        "--policy-name",
        type=str,
        default=None,
        help="Nome usato nei file di output. Se omesso, usa policy-type.",
    )

    parser.add_argument("--actor-ckpt", type=str, default=None)
    parser.add_argument("--controller-json", type=str, default=None)

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--num-gifs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--gif-skip", type=int, default=2)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--csv-prefix", type=str, default="actor_true_model")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
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
        rollout_row, step_rows = make_one_policy_real_env_gif(
            episode_idx=i,
            policy_name=policy_name,
            policy_type=args.policy_type,
            policy=policy,
            world_model=world_model,
            env_id=args.env_id,
            seed=args.seed + i,
            max_steps=args.max_steps,
            gamma=args.gamma,
            device=device,
            out_dir=out_dir,
            fps=args.fps,
            gif_skip=args.gif_skip,
        )

        rollout_rows.append(rollout_row)
        per_step_rows.extend(step_rows)

    rollout_csv = out_dir / f"{args.csv_prefix}_rollout_returns.csv"
    per_step_csv = out_dir / f"{args.csv_prefix}_per_step.csv"

    pd.DataFrame(
        rollout_rows,
        columns=[
            "rollout_id",
            "seed",
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
