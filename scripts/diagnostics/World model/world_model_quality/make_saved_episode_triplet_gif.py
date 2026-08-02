from pathlib import Path
import argparse
import math
import random
from PIL import Image, ImageDraw
from types import SimpleNamespace

import numpy as np
import torch

from world_model.load_legacy import load_legacy_world_model





FRAME_KEYS = [
    "render_obs",
    "render_frames",
    "rgb_frames",
    "frames",
    "real_frames",
    "real_raw_frames",
    "images",
    "obs",
    "observations",
]

ACTION_KEYS = [
    "actions",
    "action",
    "a",
]

REWARD_KEYS = [
    "rewards",
    "reward",
    "real_rewards",
    "real_env_rewards",
    "r",
]


def preprocess_obs(obs, size=64):
    if isinstance(obs, tuple):
        obs = obs[0]

    obs = np.asarray(obs)

    if obs.dtype != np.uint8:
        obs = np.clip(obs, 0, 255).astype(np.uint8)

    img = Image.fromarray(obs)
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0

    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return x, (arr * 255).astype(np.uint8)

def pick_array(data, candidate_keys, npz_path):
    for key in candidate_keys:
        if key in data:
            return data[key], key

    raise KeyError(
        f"Non trovo nessuna delle chiavi {candidate_keys} dentro {npz_path}. "
        f"Chiavi disponibili: {list(data.keys())}"
    )

def to_hwc_uint8(frame):
    frame = np.asarray(frame)

    if frame.ndim == 3 and frame.shape[0] in [1, 3, 4] and frame.shape[-1] not in [1, 3, 4]:
        frame = np.transpose(frame, (1, 2, 0))

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)

    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    return frame

def load_saved_episode(npz_path: Path):
    data = np.load(npz_path)

    frames, frame_key = pick_array(data, FRAME_KEYS, npz_path)
    actions, action_key = pick_array(data, ACTION_KEYS, npz_path)
    rewards, reward_key = pick_array(data, REWARD_KEYS, npz_path)

    frames = np.asarray(frames)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

    if frames.ndim != 4:
        raise ValueError(
            f"Mi aspettavo frames con shape [T,H,W,C] o [T,C,H,W], "
            f"ma ho trovato {frames.shape} in {npz_path}"
        )

    if actions.ndim != 2 or actions.shape[-1] != 3:
        raise ValueError(
            f"Mi aspettavo actions con shape [T,3], "
            f"ma ho trovato {actions.shape} in {npz_path}"
        )

    print(f"Loaded episode: {npz_path.name}")
    print(f"  frames key : {frame_key}, shape={frames.shape}")
    print(f"  actions key: {action_key}, shape={actions.shape}")
    print(f"  rewards key: {reward_key}, shape={rewards.shape}")

    return frames, actions, rewards

def align_real_rewards(rewards, num_frames):
    """
    Converte i reward per-transizione in reward per-frame.

    Convenzione:
        rewards[t] è il reward ottenuto dopo aver applicato action[t],
        cioè nella transizione:

            frame[t] -- action[t] --> frame[t+1]

    Quindi nella GIF:
        frame[0] non ha reward associato -> NaN
        frame[1] mostra rewards[0]
        frame[2] mostra rewards[1]
        ...
    """

    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

    frame_rewards = np.full(num_frames, np.nan, dtype=np.float32)

    num_rewards_to_copy = min(len(rewards), num_frames - 1)

    frame_rewards[1 : num_rewards_to_copy + 1] = rewards[:num_rewards_to_copy]

    return frame_rewards

@torch.no_grad()
def preprocess_frame_sequence(raw_frames, image_size=64):
    """
    raw_frames:
        array [T, H, W, C] proveniente direttamente dal true env.

    Restituisce:
        xs:
            lista di tensori preprocessati per il VAE / world model
        display_frames:
            frame originali del true env NON processati
    """
    raw_frames = np.asarray(raw_frames, dtype=np.uint8)

    if raw_frames.ndim != 4:
        raise ValueError(f"Expected raw_frames with shape [T,H,W,C], got {raw_frames.shape}")

    # ------------------------------------------------------------
    # Parte da usare per la GIF di sinistra:
    # frame originali, non normalizzati, non ridimensionati.
    # ------------------------------------------------------------
    display_frames = raw_frames.copy()

    # ------------------------------------------------------------
    # Parte da usare per il VAE:
    # resize a image_size x image_size + normalizzazione [0,1]
    # ------------------------------------------------------------
    xs = []

    for frame in raw_frames:
        img = Image.fromarray(frame)

        # solo per il VAE
        img_vae = img.resize((image_size, image_size), Image.BILINEAR)

        x = np.asarray(img_vae, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

        xs.append(x)

    return xs, display_frames

@torch.no_grad()
def encode_obs(wm, x, device):
    """
    Codifica una osservazione reale nel latente z del VAE.

    Input:
        x: frame preprocessato, shape [1, 3, 64, 64]

    Output:
        z: latente VAE, shape [1, 32]

    Nota:
        nel nostro ConvVAE il metodo disponibile è vae.encode(x).
        Se encode restituisce una tupla, prendiamo il primo elemento.
    """

    x = x.to(device)

    if x.ndim == 3:
        x = x.unsqueeze(0)

    out = wm.vae.encode(x)

    if isinstance(out, tuple):
        z = out[0]
    else:
        z = out

    return z.float()

def init_world_model_state(wm, z, device):
    """
    Crea lo stato iniziale del world model.

    Lo stato contiene:
        z: latente VAE iniziale, shape [B, 32]
        h: hidden state iniziale MDN-RNN, shape [1, B, 256]
        c: cell state iniziale MDN-RNN, shape [1, B, 256]

    All'inizio h e c sono nulli.
    """

    z = z.to(device).float()

    if z.ndim == 1:
        z = z.unsqueeze(0)

    batch_size = z.shape[0]

    h_dim = 256
    num_layers = 1

    h = torch.zeros(num_layers, batch_size, h_dim, device=device)
    c = torch.zeros(num_layers, batch_size, h_dim, device=device)

    return SimpleNamespace(z=z, h=h, c=c)

def tensor_to_uint8_image(x):
    x = x.detach().cpu().float()

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 3 and x.shape[0] in [1, 3]:
        x = x.permute(1, 2, 0)

    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 3)

    xmin = float(x.min())
    xmax = float(x.max())

    if xmin < -0.05 and xmax <= 1.2:
        x = (x + 1.0) / 2.0
    elif xmin < 0.0 or xmax > 1.0:
        x = torch.sigmoid(x)

    x = x.clamp(0.0, 1.0)
    arr = (x.numpy() * 255.0).round().astype(np.uint8)
    return arr

@torch.no_grad()
def decode_z(wm, z):
    z = z.float()

    if z.ndim == 1:
        z = z.unsqueeze(0)

    x_rec = wm.vae.decode(z)

    return tensor_to_uint8_image(x_rec)

def get_next_state_from_imagine_output(out):
    """
    Estrae il prossimo stato dall'output di wm.imagine_step(...).

    Nel nostro codice ci aspettiamo che imagine_step restituisca
    un oggetto con attributo .next_state.
    """

    return out.next_state

@torch.no_grad()
def world_model_step_with_reward(wm, state, action, device):
    """
    Esegue un passo del world model.

    1. Usa il transition model per ottenere il prossimo stato latente.
    2. Usa il reward model/classifier per assegnare un reward allo stato raggiunto.
    """

    action = torch.as_tensor(
        action,
        dtype=torch.float32,
        device=device,
    ).view(1, 3)

    out = wm.imagine_step(state, action)

    next_state = get_next_state_from_imagine_output(out)

    reward_pred = predict_reward_from_state(wm, next_state)

    return next_state, reward_pred

def set_state_z(state, z):
    """
    Sostituisce il latente z di uno stato del world model,
    mantenendo invariati h e c.

    Serve per la traiettoria VAE filtrata:
        h, c vengono aggiornati dal transition model
        z viene sostituito con il latente del vero frame osservato
    """

    return SimpleNamespace(
        z=z,
        h=state.h,
        c=state.c,
    )

@torch.no_grad()
def predict_reward_from_state(wm, state):
    """
    Calcola il reward predetto dal reward model del world model.

    Nel nostro caso wm.reward_model è il reward classifier
    trasformato in expected reward.

    Input:
        state.z: latente VAE, shape [1, 32]
        state.h: hidden state MDN-RNN, shape [1, 1, 256]

    Output:
        reward scalare float
    """

    z = state.z.reshape(1, -1)
    h = state.h[-1].reshape(1, -1)

    features = torch.cat([z, h], dim=-1)

    reward = wm.reward_model(features)

    return float(reward.detach().cpu().reshape(-1)[0])

@torch.no_grad()
def build_triplet_rollout(
    wm,
    raw_frames,
    actions,
    real_rewards,
    horizon,
    start_t,
    image_size,
    device,
):
    """
    Crea le tre sequenze della GIF:

    1) real_raw_frames:
       osservazioni reali salvate nell'episodio.

    2) vae_frames:
       VAE(obs reale) -> z reale -> decoder VAE.

    3) wm_frames:
       partenza dallo stesso z iniziale, poi rollout open-loop:
       z_{t+1} = transition_model(z_t, h_t, action_t)
       frame = decoder VAE(z_{t+1})
    """

    max_horizon = min(
        horizon,
        len(actions) - start_t,
        len(raw_frames) - start_t - 1,
    )

    frames_slice = raw_frames[start_t : start_t + max_horizon + 1]
    actions_slice = actions[start_t : start_t + max_horizon]
    rewards_slice = real_rewards[start_t : start_t + max_horizon]

    xs, real_raw_display = preprocess_frame_sequence(frames_slice, image_size=image_size)

    z0 = encode_obs(wm, xs[0], device=device)

    real_state = init_world_model_state(wm, z0, device)
    model_state = init_world_model_state(wm, z0.clone(), device)

    vae_frames = [decode_z(wm, z0)]
    wm_frames = [decode_z(wm, model_state.z)]

    # Reward allineati ai frame.
    real_env_rewards = align_real_rewards(rewards_slice, num_frames=max_horizon + 1)
    vae_rewards = [np.nan]
    wm_rewards = [np.nan]

    for t in range(max_horizon):
        action = actions_slice[t]

        # ------------------------------------------------------------
        # Parte centrale:
        # osservazione reale -> VAE encoder -> z reale -> VAE decoder.
        # Il reward model viene valutato sulla traiettoria reale codificata.
        # ------------------------------------------------------------
        z_real_next = encode_obs(wm, xs[t + 1], device=device)

        real_update_state, legacy_vae_reward = world_model_step_with_reward(
            wm,
            real_state,
            action,
            device,
        )

        real_state = set_state_z(real_update_state, z_real_next)

        vae_frame = decode_z(wm, z_real_next)

        vae_reward = predict_reward_from_state(wm, real_state)

        vae_frames.append(vae_frame)
        vae_rewards.append(float(vae_reward))

        # ------------------------------------------------------------
        # Parte destra:
        # world model open-loop.
        # Usa la stessa azione salvata, ma aggiorna lo stato solo tramite
        # transition model, senza correggersi con l'osservazione reale.
        # ------------------------------------------------------------
        model_state, wm_reward = world_model_step_with_reward(
            wm,
            model_state,
            action,
            device,
        )

        wm_frame = decode_z(wm, get_state_z(model_state))

        wm_reward = predict_reward_from_state(wm, model_state)

        wm_frames.append(wm_frame)
        wm_rewards.append(float(wm_reward))

    return {
        "real_raw_frames": np.asarray(real_raw_display, dtype=np.uint8),
        "vae_frames": np.asarray(vae_frames, dtype=np.uint8),
        "wm_frames": np.asarray(wm_frames, dtype=np.uint8),
        "real_env_rewards": np.asarray(real_env_rewards, dtype=np.float32),
        "vae_rewards": np.asarray(vae_rewards, dtype=np.float32),
        "wm_rewards": np.asarray(wm_rewards, dtype=np.float32),
        "actions": np.asarray(actions_slice, dtype=np.float32),
        "effective_horizon": max_horizon,
    }

def list_episode_files(args):
    if args.episode_npz is not None:
        return [Path(args.episode_npz)]

    episode_dir = Path(args.episodes_dir)
    files = sorted(episode_dir.glob(args.pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            f"Nessun file trovato in {episode_dir} con pattern {args.pattern}"
        )

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(files)

    return files[: args.num_episodes]

def save_true_truevae_model_gif(
    real_raw_frames,
    real_vae_frames,
    model_frames,
    out_path,
    fps=5,
    stride=1,
    real_env_rewards=None,
    real_vae_rewards=None,
    world_model_rewards=None,
):
    T = min(len(real_raw_frames), len(real_vae_frames), len(model_frames))
    frames = []

    for t in range(0, T, stride):
        frame = make_labeled_triplet_frame(
            real_raw_frames[t],
            real_vae_frames[t],
            model_frames[t],
            t,
            real_env_rewards=real_env_rewards,
            real_vae_rewards=real_vae_rewards,
            world_model_rewards=world_model_rewards,
        )
        frames.append(frame)

    if not frames:
        return

    duration_ms = int(1000 / fps)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

def resize_square(img, size=256):
    return img.resize((size, size), Image.Resampling.LANCZOS)

def resize_keep_aspect(img, target_h, resample=Image.NEAREST):
    w, h = img.size
    target_w = int(round(w * target_h / h))
    return img.resize((target_w, target_h), resample)

def draw_reward_plot(values, current_t, width, height=120, title="Reward"):
    img = Image.new("RGB", (int(width), int(height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n = len(values)

    left = 38
    right = 8
    top = 22
    bottom = 22

    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)

    # Titolo
    draw.text((4, 4), title, fill=(0, 0, 0))

    finite = np.isfinite(values)

    if n == 0 or not finite.any():
        draw.text((left, top + 20), "no reward", fill=(120, 120, 120))
        return img

    vals = values[finite]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    if abs(vmax - vmin) < 1e-8:
        vmin -= 1.0
        vmax += 1.0

    # Assi
    x0 = left
    y0 = top + plot_h
    x1 = left + plot_w
    y1 = top

    draw.line((x0, y0, x1, y0), fill=(0, 0, 0), width=1)
    draw.line((x0, y0, x0, y1), fill=(0, 0, 0), width=1)

    # Min/max labels
    draw.text((2, y1 - 5), f"{vmax:.2f}", fill=(80, 80, 80))
    draw.text((2, y0 - 8), f"{vmin:.2f}", fill=(80, 80, 80))

    def to_xy(i, v):
        if n <= 1:
            x = x0
        else:
            x = x0 + (i / (n - 1)) * plot_w

        y = y0 - ((float(v) - vmin) / (vmax - vmin)) * plot_h
        return int(round(x)), int(round(y))

    # Disegna la curva, spezzando in presenza di NaN.
    segment = []
    for i, v in enumerate(values):
        if np.isfinite(v):
            segment.append(to_xy(i, v))
        else:
            if len(segment) >= 2:
                draw.line(segment, fill=(20, 80, 180), width=2)
            segment = []

    if len(segment) >= 2:
        draw.line(segment, fill=(20, 80, 180), width=2)

    for i, v in enumerate(values):
        if np.isfinite(v):
            x, y = to_xy(i, v)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(20, 80, 180))

    # Linea verticale sul frame corrente
    current_t = int(min(max(current_t, 0), n - 1))
    cx, _ = to_xy(current_t, values[current_t] if np.isfinite(values[current_t]) else vmin)
    draw.line((cx, y1, cx, y0), fill=(180, 40, 40), width=1)

    # Valore corrente
    cur = values[current_t]
    if np.isfinite(cur):
        draw.text((left, height - 17), f"t={current_t}, r={cur:.3f}", fill=(0, 0, 0))
    else:
        draw.text((left, height - 17), f"t={current_t}, r=nan", fill=(0, 0, 0))

    return img

def make_labeled_triplet_frame(
    real_raw,
    real_vae,
    model,
    t,
    real_env_rewards=None,
    real_vae_rewards=None,
    world_model_rewards=None,
):
    panel_size = 256

    # True env no VAE:
    # frame renderizzato dall'ambiente, solo ridimensionato per il layout.
    real_raw_img = Image.fromarray(real_raw).convert("RGB")
    real_raw_img = real_raw_img.resize(
        (panel_size, panel_size),
        resample=Image.NEAREST,
    )

    # True env VAE e World model:
    # frame decodificati dal VAE/world model, mostrati nello stesso spazio.
    real_vae_img = Image.fromarray(real_vae).convert("RGB")
    model_img = Image.fromarray(model).convert("RGB")

    real_vae_img = real_vae_img.resize(
        (panel_size, panel_size),
        resample=Image.NEAREST,
    )
    model_img = model_img.resize(
        (panel_size, panel_size),
        resample=Image.NEAREST,
    )

    panel_gap = 8
    top_h = 34
    img_h = panel_size
    plot_h = 120

    canvas_w = panel_size + panel_gap + panel_size + panel_gap + panel_size
    canvas_h = top_h + img_h + plot_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    x0 = 0
    x1 = panel_size + panel_gap
    x2 = panel_size + panel_gap + panel_size + panel_gap

    canvas.paste(real_raw_img, (x0, top_h))
    canvas.paste(real_vae_img, (x1, top_h))
    canvas.paste(model_img, (x2, top_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((x0 + 8, 8), f"True env raw | t={t}", fill=(0, 0, 0))
    draw.text((x1 + 8, 8), f"True env VAE | t={t}", fill=(0, 0, 0))
    draw.text((x2 + 8, 8), f"World model | t={t}", fill=(0, 0, 0))

    plot_y = top_h + img_h

    if real_env_rewards is not None:
        p0 = draw_reward_plot(
            real_env_rewards,
            current_t=t,
            width=panel_size,
            height=plot_h,
            title="Real env reward",
        )
        canvas.paste(p0, (x0, plot_y))

    if real_vae_rewards is not None:
        p1 = draw_reward_plot(
            real_vae_rewards,
            current_t=t,
            width=panel_size,
            height=plot_h,
            title="Reward model on true VAE state",
        )
        canvas.paste(p1, (x1, plot_y))

    if world_model_rewards is not None:
        p2 = draw_reward_plot(
            world_model_rewards,
            current_t=t,
            width=panel_size,
            height=plot_h,
            title="Reward model on WM rollout",
        )
        canvas.paste(p2, (x2, plot_y))

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera GIF true env / VAE reconstruction / world-model open-loop "
            "partendo da episodi reali già salvati in .npz."
        )
    )

    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, default=None)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--episode-npz", type=str, default=None)
    parser.add_argument("--episodes-dir", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--start-t", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)

    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-stride", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="episode")

    args = parser.parse_args()

    if args.episode_npz is None and args.episodes_dir is None:
        raise ValueError("Devi passare --episode-npz oppure --episodes-dir.")

    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    gifs_dir = out_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading world model...")
    wm = load_legacy_world_model(
        vae_ckpt=args.vae_ckpt,
        mdn_rnn_ckpt=args.mdn_rnn_ckpt,
        reward_ckpt=args.reward_ckpt,
        reward_calibration=None,
        device=device,
    )

    print("Reward source: classifier expected reward")

    episode_files = list_episode_files(args)

    for idx, npz_path in enumerate(episode_files):
        raw_frames, actions, real_rewards = load_saved_episode(npz_path)

        rollout = build_triplet_rollout(
            wm=wm,
            raw_frames=raw_frames,
            actions=actions,
            real_rewards=real_rewards,
            horizon=args.horizon,
            start_t=args.start_t,
            image_size=args.image_size,
            device=device,
        )

        stem = f"{args.prefix}_{idx:03d}_{npz_path.stem}"
        gif_path = gifs_dir / f"{stem}_true_vae_wm.gif"

        save_true_truevae_model_gif(
            real_raw_frames=rollout["real_raw_frames"],
            real_vae_frames=rollout["vae_frames"],
            model_frames=rollout["wm_frames"],
            out_path=gif_path,
            fps=args.gif_fps,
            stride=args.gif_stride,
            real_env_rewards=rollout["real_env_rewards"],
            real_vae_rewards=rollout["vae_rewards"],
            world_model_rewards=rollout["wm_rewards"],
        )

        print(
            f"Saved {gif_path} | "
            f"horizon={rollout['effective_horizon']} | "
            f"start_t={args.start_t}"
        )


if __name__ == "__main__":
    main()
