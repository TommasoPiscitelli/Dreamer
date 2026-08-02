import argparse
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from world_model.load_legacy import load_legacy_world_model
from world_model.api import LatentState
from dreamer.actor_critic import Actor
from dreamer.evo_controller import EvoController

from PIL import Image, ImageDraw
from io import BytesIO
# ---------------------------------------------------------------------
# Robust actor / controller loading
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
        "Could not build Actor. Check actor_critic.py.\n"
        + "\n".join(errors)
    )


def load_actor(actor_ckpt: Path, device):
    actor = build_actor_robust(device)
    ckpt = torch.load(actor_ckpt, map_location=device)

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

    if len(missing) > 0:
        print("Warning: missing actor keys:")
        for k in missing:
            print("  ", k)

    if len(unexpected) > 0:
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
# Action utilities
# ---------------------------------------------------------------------

def extract_action_tensor_from_output(out):
    if isinstance(out, torch.Tensor):
        return out

    if hasattr(out, "mean") and isinstance(out.mean, torch.Tensor):
        return out.mean

    if hasattr(out, "loc") and isinstance(out.loc, torch.Tensor):
        return out.loc

    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
            if hasattr(item, "mean") and isinstance(item.mean, torch.Tensor):
                return item.mean
            if hasattr(item, "loc") and isinstance(item.loc, torch.Tensor):
                return item.loc

    if isinstance(out, dict):
        for key in ["action", "mean", "loc", "raw_action"]:
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


def dreamer_action(actor, features):
    with torch.no_grad():
        out = actor(features)
        action = extract_action_tensor_from_output(out)
        return ensure_valid_action(action)


def evo_action(controller, features):
    with torch.no_grad():
        out = controller(features)
        action = extract_action_tensor_from_output(out)
        return ensure_valid_action(action)


# ---------------------------------------------------------------------
# World model utilities
# ---------------------------------------------------------------------

def get_next_state(imagine_out):
    if hasattr(imagine_out, "next_state"):
        return imagine_out.next_state
    if hasattr(imagine_out, "state"):
        return imagine_out.state
    if isinstance(imagine_out, tuple):
        return imagine_out[0]
    return imagine_out


def get_reward(imagine_out):
    for name in ["reward", "r", "pred_reward"]:
        if hasattr(imagine_out, name):
            value = getattr(imagine_out, name)
            if isinstance(value, torch.Tensor):
                return value

    if isinstance(imagine_out, tuple):
        for item in imagine_out:
            if isinstance(item, torch.Tensor):
                if item.numel() == 1 or item.shape[-1] == 1:
                    return item

    return None


def make_state(z, h, c):
    try:
        return LatentState(z=z, h=h, c=c, extra=None)
    except TypeError:
        return LatentState(z=z, h=h, c=c)


def load_initial_state_from_npz(path: Path, t: int, device: torch.device):
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
    path = random.choice(latent_files)
    data = np.load(path)
    T = len(data["z"])

    if T <= min_t + 2:
        return sample_start(latent_files, min_t=min_t)

    t = random.randint(min_t, T - 2)
    return path, t


# ---------------------------------------------------------------------
# Decoding z -> frame
# ---------------------------------------------------------------------

def decode_z_to_frame(world_model, z):
    """
    Try several possible VAE decoding APIs.
    Returns one RGB uint8 frame.
    """
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
                decoded = None

        if decoded is None:
            raise RuntimeError(
                "Could not decode z into image. "
                "No compatible decode method found. "
                f"Last error: {last_error}"
            )

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

    # Remove batch
    if x.ndim == 4:
        x = x[0]

    # CHW -> HWC
    if x.ndim == 3 and x.shape[0] in [1, 3, 4]:
        x = x.permute(1, 2, 0)

    arr = x.numpy()

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
# Rollout collection
# ---------------------------------------------------------------------

def run_world_model_episode(
    policy_name,
    policy,
    policy_type,
    world_model,
    initial_state,
    source_file,
    source_t,
    horizon,
    device,
):
    state = make_state(
        z=initial_state.z.clone(),
        h=initial_state.h.clone(),
        c=initial_state.c.clone(),
    )

    z_list = []
    h_list = []
    c_list = []
    action_list = []
    reward_list = []
    frame_list = []

    for t in range(horizon):
        z_list.append(state.z.detach().cpu().numpy()[0])
        h_list.append(state.h.detach().cpu().numpy())
        c_list.append(state.c.detach().cpu().numpy())

        try:
            frame = decode_z_to_frame(world_model, state.z)
            frame_list.append(frame)
        except Exception as e:
            if t == 0:
                print(f"Warning: could not decode frames for {policy_name}: {e}")
            frame_list = []

        features = state.features

        if policy_type == "dreamer":
            action = dreamer_action(policy, features)
        elif policy_type == "evo":
            action = evo_action(policy, features)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        action_np = action.detach().cpu().numpy()[0].astype(np.float32)
        action_list.append(action_np)

        imagine_out = world_model.imagine_step(state, action)

        reward = get_reward(imagine_out)
        if reward is None:
            reward_list.append(np.nan)
        else:
            reward_list.append(float(reward.detach().cpu().reshape(-1)[0]))

        state = get_next_state(imagine_out)

    return {
        "policy": policy_name,
        "source_file": source_file,
        "source_t": source_t,
        "z": np.asarray(z_list, dtype=np.float32),
        "h": np.asarray(h_list, dtype=np.float32),
        "c": np.asarray(c_list, dtype=np.float32),
        "action": np.asarray(action_list, dtype=np.float32),
        "reward_pred": np.asarray(reward_list, dtype=np.float32),
        "frames": np.asarray(frame_list) if len(frame_list) > 0 else None,
    }


def save_episode_npz(ep, out_npz):
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "policy": ep["policy"],
        "source_file": ep["source_file"],
        "source_t": ep["source_t"],
        "z": ep["z"],
        "h": ep["h"],
        "c": ep["c"],
        "action": ep["action"],
        "reward_pred": ep["reward_pred"],
        "predicted_return": float(np.nansum(ep["reward_pred"])),
    }

    if ep["frames"] is not None:
        payload["frames"] = ep["frames"]

    np.savez_compressed(out_npz, **payload)


# ---------------------------------------------------------------------
# CSV and plots
# ---------------------------------------------------------------------

def make_action_csv(dreamer_npz, evo_npz, out_csv):
    rows = []

    for policy, path in [("Dreamer", dreamer_npz), ("Evo", evo_npz)]:
        data = np.load(path)
        actions = data["action"]
        rewards = data["reward_pred"] if "reward_pred" in data else None

        for t, a in enumerate(actions):
            rows.append({
                "policy": policy,
                "source_file": Path(path).name,
                "timestep": t,
                "steer": float(a[0]),
                "gas": float(a[1]),
                "brake": float(a[2]),
                "reward_pred": float(rewards[t]) if rewards is not None else np.nan,
            })

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def plot_action_comparison(df, out_png):
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    for ax, action_name in zip(axes[:3], ["steer", "gas", "brake"]):
        for policy in ["Dreamer", "Evo"]:
            sub = df[df["policy"] == policy].sort_values("timestep")
            ax.plot(
                sub["timestep"],
                sub[action_name],
                label=f"{policy} {action_name}",
                linewidth=2,
            )

        ax.set_ylabel(action_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    dreamer = df[df["policy"] == "Dreamer"].sort_values("timestep")
    evo = df[df["policy"] == "Evo"].sort_values("timestep")

    T = min(len(dreamer), len(evo))

    for action_name in ["steer", "gas", "brake"]:
        diff = np.abs(
            dreamer[action_name].to_numpy()[:T]
            - evo[action_name].to_numpy()[:T]
        )
        axes[3].plot(range(T), diff, label=f"|{action_name} diff|", linewidth=2)

    axes[3].set_ylabel("absolute diff")
    axes[3].set_xlabel("World model timestep")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.suptitle("World model closed-loop actions: Dreamer vs Evo")
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)




def read_gif_frames(path):
    path = Path(path)
    img = Image.open(path)

    frames = []
    try:
        while True:
            frames.append(img.convert("RGB").copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    if len(frames) == 0:
        raise RuntimeError(f"No frames found in GIF: {path}")

    return frames


def resize_keep_aspect_pil(img, target_width):
    w, h = img.size
    target_height = int(round(h * target_width / w))
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)


def draw_single_action_plot(values, current_t, width, height, title, y_min, y_max):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n = len(values)

    left = 42
    right = 10
    top = 20
    bottom = 22

    plot_w = width - left - right
    plot_h = height - top - bottom

    draw.text((6, 4), title, fill=(0, 0, 0))

    # Assi
    x0 = left
    y0 = top + plot_h
    x1 = left + plot_w
    y1 = top

    draw.line((x0, y0, x1, y0), fill=(0, 0, 0), width=1)
    draw.line((x0, y0, x0, y1), fill=(0, 0, 0), width=1)

    draw.text((4, y1 - 6), f"{y_max:.1f}", fill=(80, 80, 80))
    draw.text((4, y0 - 8), f"{y_min:.1f}", fill=(80, 80, 80))

    if n <= 1:
        return img

    def to_xy(i, v):
        x = x0 + (i / (n - 1)) * plot_w
        v = float(np.clip(v, y_min, y_max))
        y = y0 - ((v - y_min) / (y_max - y_min)) * plot_h
        return int(round(x)), int(round(y))

    points = [to_xy(i, v) for i, v in enumerate(values)]
    draw.line(points, fill=(20, 80, 180), width=2)

    current_t = int(np.clip(current_t, 0, n - 1))
    cx, cy = to_xy(current_t, values[current_t])

    # Linea verticale tempo corrente
    draw.line((cx, y1, cx, y0), fill=(180, 40, 40), width=2)

    # Punto corrente
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(180, 40, 40))

    draw.text(
        (left, height - 17),
        f"t={current_t}, a={values[current_t]:.3f}",
        fill=(0, 0, 0),
    )

    return img


def draw_action_panel(actions, current_t, width, policy_name):
    actions = np.asarray(actions, dtype=np.float32)

    plot_h = 72
    gap = 4
    title_h = 24

    panel_h = title_h + 3 * plot_h + 2 * gap

    panel = Image.new("RGB", (width, panel_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    draw.text((8, 5), f"{policy_name} actions", fill=(0, 0, 0))

    action_specs = [
        ("steer", actions[:, 0], -1.0, 1.0),
        ("gas", actions[:, 1], 0.0, 1.0),
        ("brake", actions[:, 2], 0.0, 1.0),
    ]

    y = title_h

    for name, values, y_min, y_max in action_specs:
        plot = draw_single_action_plot(
            values=values,
            current_t=current_t,
            width=width,
            height=plot_h,
            title=name,
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

    plot = draw_single_action_plot(
        values=rewards,
        current_t=current_t,
        width=width,
        height=plot_h,
        title="reward",
        y_min=y_min,
        y_max=y_max,
    )

    panel.paste(plot, (0, title_h))

    return panel


def make_combined_dreamer_evo_action_gif(
    dreamer_gif_path,
    evo_gif_path,
    dreamer_npz_path,
    evo_npz_path,
    out_path,
    fps=20,
    column_width=360,
):
    dreamer_frames = read_gif_frames(dreamer_gif_path)
    evo_frames = read_gif_frames(evo_gif_path)

    dreamer_data = np.load(dreamer_npz_path)
    evo_data = np.load(evo_npz_path)

    dreamer_actions = dreamer_data["action"]
    evo_actions = evo_data["action"]

    dreamer_rewards = dreamer_data["reward_pred"]
    evo_rewards = evo_data["reward_pred"]

    n_frames = min(
        len(dreamer_frames),
        len(evo_frames),
        len(dreamer_actions),
        len(evo_actions),
        len(dreamer_rewards),
        len(evo_rewards),
    )

    combined_frames = []

    gap = 12
    top_title_h = 28

    for t in range(n_frames):
        dreamer_img = resize_keep_aspect_pil(dreamer_frames[t], column_width)
        evo_img = resize_keep_aspect_pil(evo_frames[t], column_width)

        top_h = max(dreamer_img.size[1], evo_img.size[1])

        dreamer_canvas = Image.new("RGB", (column_width, top_h), color=(255, 255, 255))
        evo_canvas = Image.new("RGB", (column_width, top_h), color=(255, 255, 255))

        dreamer_canvas.paste(dreamer_img, (0, 0))
        evo_canvas.paste(evo_img, (0, 0))

        action_t_dreamer = min(t, len(dreamer_actions) - 1)
        action_t_evo = min(t, len(evo_actions) - 1)

        reward_t_dreamer = min(t, len(dreamer_rewards) - 1)
        reward_t_evo = min(t, len(evo_rewards) - 1)

        dreamer_action_panel = draw_action_panel(
            actions=dreamer_actions,
            current_t=action_t_dreamer,
            width=column_width,
            policy_name="Dreamer",
        )

        evo_action_panel = draw_action_panel(
            actions=evo_actions,
            current_t=action_t_evo,
            width=column_width,
            policy_name="EVO",
        )

        dreamer_reward_panel = draw_reward_panel(
            rewards=dreamer_rewards,
            current_t=reward_t_dreamer,
            width=column_width,
            policy_name="Dreamer",
        )

        evo_reward_panel = draw_reward_panel(
            rewards=evo_rewards,
            current_t=reward_t_evo,
            width=column_width,
            policy_name="EVO",
        )

        canvas_w = 2 * column_width + gap
        action_panel_h = max(
            dreamer_action_panel.size[1],
            evo_action_panel.size[1],
        )

        reward_panel_h = max(
            dreamer_reward_panel.size[1],
            evo_reward_panel.size[1],
        )

        canvas_h = (
            top_title_h
            + top_h
            + gap
            + action_panel_h
            + gap
            + reward_panel_h
        )

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        draw.text((8, 6), f"Dreamer world model rollout | frame {t}", fill=(0, 0, 0))
        draw.text((column_width + gap + 8, 6), f"EVO world model rollout | frame {t}", fill=(0, 0, 0))

        canvas.paste(dreamer_canvas, (0, top_title_h))
        canvas.paste(evo_canvas, (column_width + gap, top_title_h))

        y_actions = top_title_h + top_h + gap

        canvas.paste(dreamer_action_panel, (0, y_actions))
        canvas.paste(evo_action_panel, (column_width + gap, y_actions))

        y_rewards = y_actions + action_panel_h + gap

        canvas.paste(
            dreamer_reward_panel,
            (0, y_rewards),
        )

        canvas.paste(
            evo_reward_panel,
            (column_width + gap, y_rewards),
        )

        combined_frames.append(canvas)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = int(1000 / fps)

    combined_frames[0].save(
        out_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=duration_ms,
        loop=0,
    )

    print("saved:", out_path)
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--latent-pattern", type=str, default="*.npz")
    parser.add_argument("--source-file", type=str, default=None)
    parser.add_argument("--source-t", type=int, default=None)

    parser.add_argument("--actor-ckpt", type=str, required=True)
    parser.add_argument("--controller-json", type=str, required=True)

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gif-fps", type=int, default=20)

    parser.add_argument("--out-dir", type=str, required=True)

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

    if args.source_file is not None:
        matches = [p for p in latent_files if p.name == args.source_file]
        if len(matches) == 0:
            raise FileNotFoundError(f"source_file {args.source_file} not found.")
        source_path = matches[0]
        source_t = args.source_t if args.source_t is not None else 1
    else:
        source_path, source_t = sample_start(latent_files)

    print(f"Initial latent state: {source_path.name}, t={source_t}")

    world_model = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=args.reward_calibration,
        device=device,
    )

    actor = load_actor(Path(args.actor_ckpt), device)
    evo = load_evo_controller(Path(args.controller_json), device)

    initial_state = load_initial_state_from_npz(source_path, source_t, device)

    dreamer_ep = run_world_model_episode(
        policy_name="Dreamer",
        policy=actor,
        policy_type="dreamer",
        world_model=world_model,
        initial_state=initial_state,
        source_file=source_path.name,
        source_t=source_t,
        horizon=args.horizon,
        device=device,
    )

    evo_ep = run_world_model_episode(
        policy_name="Evo",
        policy=evo,
        policy_type="evo",
        world_model=world_model,
        initial_state=initial_state,
        source_file=source_path.name,
        source_t=source_t,
        horizon=args.horizon,
        device=device,
    )

    dreamer_npz = out_dir / "dreamer_world_model_full_episode.npz"
    evo_npz = out_dir / "evo_world_model_full_episode.npz"

    save_episode_npz(dreamer_ep, dreamer_npz)
    save_episode_npz(evo_ep, evo_npz)

    if dreamer_ep["frames"] is not None:
        save_gif(dreamer_ep["frames"], out_dir / "dreamer_world_model_full_episode.gif", fps=args.gif_fps)
    else:
        print("Dreamer GIF not saved because decoding frames failed.")

    if evo_ep["frames"] is not None:
        save_gif(evo_ep["frames"], out_dir / "evo_world_model_full_episode.gif", fps=args.gif_fps)
    else:
        print("Evo GIF not saved because decoding frames failed.")

    combined_gif_path = out_dir / "dreamer_vs_evo_world_model_actions.gif"

    make_combined_dreamer_evo_action_gif(
    dreamer_gif_path=out_dir / "dreamer_world_model_full_episode.gif",
    evo_gif_path=out_dir / "evo_world_model_full_episode.gif",
    dreamer_npz_path=out_dir / "dreamer_world_model_full_episode.npz",
    evo_npz_path=out_dir / "evo_world_model_full_episode.npz",
    out_path=combined_gif_path,
    fps=args.gif_fps,
    column_width=360,
    )

    # action_csv = out_dir / "world_model_single_episode_actions.csv"
    # df = make_action_csv(dreamer_npz, evo_npz, action_csv)

    # action_png = out_dir / "world_model_single_episode_action_comparison.png"
    # plot_action_comparison(df, action_png)

    # summary = []
    # for policy, npz_path in [("Dreamer", dreamer_npz), ("Evo", evo_npz)]:
    #     data = np.load(npz_path)
    #     a = data["action"]
    #     r = data["reward_pred"]
    #     summary.append({
    #         "policy": policy,
    #         "source_file": str(data["source_file"]),
    #         "source_t": int(data["source_t"]),
    #         "horizon": len(a),
    #         "predicted_return": float(np.nansum(r)),
    #         "mean_pred_reward": float(np.nanmean(r)),
    #         "steer_mean": float(a[:, 0].mean()),
    #         "gas_mean": float(a[:, 1].mean()),
    #         "brake_mean": float(a[:, 2].mean()),
    #         "steer_abs_gt_09_frac": float((np.abs(a[:, 0]) > 0.9).mean()),
    #         "gas_gt_095_frac": float((a[:, 1] > 0.95).mean()),
    #         "brake_lt_005_frac": float((a[:, 2] < 0.005).mean()),
    #     })

    # summary_df = pd.DataFrame(summary)
    # summary_csv = out_dir / "world_model_single_episode_summary.csv"
    # summary_df.to_csv(summary_csv, index=False)

    print("\nSaved:")
    print(f"  {dreamer_npz}")
    print(f"  {evo_npz}")
    # print(f"  {action_csv}")
    # print(f"  {action_png}")
    # print(f"  {summary_csv}")
    if dreamer_ep["frames"] is not None:
        print(f"  {out_dir / 'dreamer_world_model_full_episode.gif'}")
    if evo_ep["frames"] is not None:
        print(f"  {out_dir / 'evo_world_model_full_episode.gif'}")

    # print()
    # print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
