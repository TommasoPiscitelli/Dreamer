import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from world_model.model import load_legacy_world_model
from world_model.api import LatentState
from dreamer.actor_critic import Actor
from dreamer.evo_controller import EvoController


# ---------------------------------------------------------------------
# Gym compatibility
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


# ---------------------------------------------------------------------
# Observation / frame utilities
# ---------------------------------------------------------------------

def obs_to_tensor(obs, device):
    """
    Convert CarRacing observation HWC [96,96,3] to BCHW [1,3,64,64].
    """
    obs = np.asarray(obs)

    if obs.ndim != 3:
        raise ValueError(f"Expected obs shape [H,W,C], got {obs.shape}")

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


def to_rgb_pil(frame):
    return Image.fromarray(normalize_frame(frame)).convert("RGB")


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
    return make_state(z=z, h=previous_state.h, c=previous_state.c, like_state=previous_state)


# ---------------------------------------------------------------------
# Actor / controller loading
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

    raise RuntimeError(
        "Could not build Actor with any known constructor pattern.\n"
        + "\n".join(errors)
    )


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
        evo = EvoController.from_json(controller_json, device=device)
    except TypeError:
        evo = EvoController.from_json(controller_json)

    if hasattr(evo, "to"):
        evo = evo.to(device)

    if hasattr(evo, "eval"):
        evo.eval()

    return evo


# ---------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------

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

    raise ValueError(f"Could not extract action tensor from output type {type(out)}")


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


def dreamer_action(actor, features):
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


def evo_action(controller, features):
    with torch.no_grad():
        out = controller(features)
        action = extract_tensor_from_output(out)
        return ensure_valid_action(action)


# ---------------------------------------------------------------------
# Real environment rollout
# ---------------------------------------------------------------------

def collect_real_env_episode(
    policy_name,
    policy,
    policy_type,
    world_model,
    env_id,
    seed,
    max_steps,
    device,
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
    actions = []
    rewards = []

    for _ in range(max_steps):
        frame = normalize_frame(get_frame(env, obs))
        frames.append(frame)

        features = state.features

        if policy_type == "dreamer":
            action_t = dreamer_action(policy, features)
        elif policy_type == "evo":
            action_t = evo_action(policy, features)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        action_np = action_t.detach().cpu().numpy()[0].astype(np.float32)

        next_obs, reward, done, _ = step_env(env, action_np)

        imagine_out = world_model.imagine_step(state, action_t)
        predicted_next_state = get_next_state(imagine_out)

        state = encode_real_obs_into_state(
            world_model=world_model,
            obs=next_obs,
            previous_state=predicted_next_state,
            device=device,
        )

        actions.append(action_np)
        rewards.append(float(reward))

        obs = next_obs

        if done:
            break

    env.close()

    return {
        "policy": policy_name,
        "frames": np.asarray(frames, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "return": float(np.sum(rewards)),
    }


# ---------------------------------------------------------------------
# Combined GIF
# ---------------------------------------------------------------------

def resize_square(img, size):
    return img.resize((size, size), Image.Resampling.LANCZOS)


def draw_action_trace(values, current_t, width, height, name, y_min, y_max):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    left = 42
    right = 10
    top = 20
    bottom = 20

    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)

    x0 = left
    y0 = top + plot_h
    x1 = left + plot_w
    y1 = top

    draw.text((6, 4), name, fill=(0, 0, 0))

    draw.line((x0, y0, x1, y0), fill=(0, 0, 0), width=1)
    draw.line((x0, y0, x0, y1), fill=(0, 0, 0), width=1)

    draw.text((4, y1 - 6), f"{y_max:.1f}", fill=(80, 80, 80))
    draw.text((4, y0 - 8), f"{y_min:.1f}", fill=(80, 80, 80))

    if len(values) == 0:
        return img

    current_t = int(np.clip(current_t, 0, len(values) - 1))

    def to_xy(i, v):
        if len(values) <= 1:
            x = x0
        else:
            x = x0 + (i / (len(values) - 1)) * plot_w

        v = float(np.clip(v, y_min, y_max))
        y = y0 - ((v - y_min) / (y_max - y_min)) * plot_h
        return int(round(x)), int(round(y))

    if current_t >= 1:
        points = [to_xy(i, values[i]) for i in range(current_t + 1)]
        draw.line(points, fill=(20, 80, 180), width=2)

    cx, cy = to_xy(current_t, values[current_t])

    draw.line((cx, y1, cx, y0), fill=(180, 40, 40), width=2)
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(180, 40, 40))

    draw.text(
        (left, height - 16),
        f"t={current_t}, a={values[current_t]:.3f}",
        fill=(0, 0, 0),
    )

    return img


def draw_policy_action_panel(actions, current_t, width, policy_name):
    actions = np.asarray(actions, dtype=np.float32)

    title_h = 24
    plot_h = 72
    gap = 4

    panel_h = title_h + 3 * plot_h + 2 * gap

    panel = Image.new("RGB", (width, panel_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    draw.text((8, 5), f"{policy_name} actions", fill=(0, 0, 0))

    specs = [
        ("steer", actions[:, 0], -1.0, 1.0),
        ("gas", actions[:, 1], 0.0, 1.0),
        ("brake", actions[:, 2], 0.0, 1.0),
    ]

    y = title_h

    for name, values, y_min, y_max in specs:
        plot = draw_action_trace(
            values=values,
            current_t=current_t,
            width=width,
            height=plot_h,
            name=name,
            y_min=y_min,
            y_max=y_max,
        )
        panel.paste(plot, (0, y))
        y += plot_h + gap

    return panel

def draw_reward_panel(rewards, current_t, width, policy_name):
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

    plot_h = 72
    title_h = 24

    panel_h = title_h + plot_h

    panel = Image.new("RGB", (width, panel_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    draw.text((8, 5), f"{policy_name} rewards", fill=(0, 0, 0))

    finite_rewards = rewards[np.isfinite(rewards)]

    if len(finite_rewards) == 0:
        y_min, y_max = -1.0, 1.0
    else:
        y_min = float(finite_rewards.min())
        y_max = float(finite_rewards.max())

        if y_min == y_max:
            margin = 1.0 if y_min == 0 else 0.1 * abs(y_min)
        else:
            margin = 0.1 * (y_max - y_min)

        y_min -= margin
        y_max += margin

    plot = draw_action_trace(
        values=rewards,
        current_t=current_t,
        width=width,
        height=plot_h,
        name="reward",
        y_min=y_min,
        y_max=y_max,
    )

    panel.paste(plot, (0, title_h))

    return panel

def save_combined_real_env_gif(
    dreamer_ep,
    evo_ep,
    out_path,
    fps=20,
    gif_skip=2,
    column_width=360,
    frame_size=256,
):
    dreamer_frames = dreamer_ep["frames"]
    evo_frames = evo_ep["frames"]
    dreamer_actions = dreamer_ep["actions"]
    evo_actions = evo_ep["actions"]
    dreamer_rewards = dreamer_ep["rewards"]
    evo_rewards = evo_ep["rewards"]

    T = min(
        len(dreamer_frames),
        len(evo_frames),
        len(dreamer_actions),
        len(evo_actions),
        len(dreamer_rewards),
        len(evo_rewards),
    )

    if T <= 0:
        raise RuntimeError("No frames/actions available for combined GIF.")

    indices = list(range(0, T, max(1, gif_skip)))

    gap = 12
    title_h = 30
    action_gap = 10

    canvas_w = 2 * column_width + gap

    gif_frames = []

    for t in indices:
        dreamer_img = resize_square(to_rgb_pil(dreamer_frames[t]), frame_size)
        evo_img = resize_square(to_rgb_pil(evo_frames[t]), frame_size)

        dreamer_col = Image.new("RGB", (column_width, frame_size), color=(255, 255, 255))
        evo_col = Image.new("RGB", (column_width, frame_size), color=(255, 255, 255))

        x_offset = (column_width - frame_size) // 2

        dreamer_col.paste(dreamer_img, (x_offset, 0))
        evo_col.paste(evo_img, (x_offset, 0))

        dreamer_panel = draw_policy_action_panel(
            actions=dreamer_actions,
            current_t=t,
            width=column_width,
            policy_name="Dreamer",
        )

        evo_panel = draw_policy_action_panel(
            actions=evo_actions,
            current_t=t,
            width=column_width,
            policy_name="Evo",
        )

        dreamer_reward_panel = draw_reward_panel(
            rewards=dreamer_rewards,
            current_t=t,
            width=column_width,
            policy_name="Dreamer",
        )

        evo_reward_panel = draw_reward_panel(
            rewards=evo_rewards,
            current_t=t,
            width=column_width,
            policy_name="EVO",
        )

        canvas_w = 2 * column_width + gap
        action_panel_h = max(
            dreamer_panel.size[1],
            evo_panel.size[1],
        )

        reward_panel_h = max(
            dreamer_reward_panel.size[1],
            evo_reward_panel.size[1],
        )

        canvas_h = (
            title_h
            + gap
            + action_panel_h
            + gap
            + reward_panel_h
        )

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        draw.text(
            (8, 7),
            f"Dreamer real env | t={t} | return={dreamer_ep['return']:.2f}",
            fill=(0, 0, 0),
        )

        draw.text(
            (column_width + gap + 8, 7),
            f"Evo real env | t={t} | return={evo_ep['return']:.2f}",
            fill=(0, 0, 0),
        )

        canvas.paste(dreamer_col, (0, title_h))
        canvas.paste(evo_col, (column_width + gap, title_h))

        y_panel = title_h + frame_size + action_gap

        canvas.paste(dreamer_panel, (0, y_panel))
        canvas.paste(evo_panel, (column_width + gap, y_panel))

        y_rewards = y_panel + action_panel_h + gap
        
        canvas.paste(
            dreamer_reward_panel,
            (0, y_rewards),
        )

        canvas.paste(
            evo_reward_panel,
            (column_width + gap, y_rewards),
        )

        gif_frames.append(canvas)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = int(1000 / fps)

    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--actor-ckpt", type=str, required=True)
    parser.add_argument("--controller-json", type=str, required=True)

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)

    # Serve al loader del world model. Nel real env il reward usato per la valutazione
    # viene dall'ambiente, non dal reward model.
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dreamer-seed", type=int, default=None)
    parser.add_argument("--evo-seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--gif-skip", type=int, default=2)

    args = parser.parse_args()

    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dreamer_seed = args.seed if args.dreamer_seed is None else args.dreamer_seed
    evo_seed = args.seed if args.evo_seed is None else args.evo_seed

    print("Loading world model...")
    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )

    print("Loading Dreamer actor...")
    actor = load_actor(Path(args.actor_ckpt), device)

    print("Loading Evo controller...")
    evo = load_evo_controller(Path(args.controller_json), device)

    print("Collecting Dreamer real-env episode...")
    dreamer_ep = collect_real_env_episode(
        policy_name="Dreamer",
        policy=actor,
        policy_type="dreamer",
        world_model=world_model,
        env_id=args.env_id,
        seed=dreamer_seed,
        max_steps=args.max_steps,
        device=device,
    )

    print("Collecting Evo real-env episode...")
    evo_ep = collect_real_env_episode(
        policy_name="Evo",
        policy=evo,
        policy_type="evo",
        world_model=world_model,
        env_id=args.env_id,
        seed=evo_seed,
        max_steps=args.max_steps,
        device=device,
    )

    out_gif = out_dir / "dreamer_vs_evo_real_env_actions.gif"

    save_combined_real_env_gif(
        dreamer_ep=dreamer_ep,
        evo_ep=evo_ep,
        out_path=out_gif,
        fps=args.gif_fps,
        gif_skip=args.gif_skip,
    )

    print()
    print("Dreamer return:", dreamer_ep["return"])
    print("Evo return:", evo_ep["return"])
    print()
    print("Saved:")
    print(f"  {out_gif}")


if __name__ == "__main__":
    main()
