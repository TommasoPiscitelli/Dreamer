import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from world_model.api import LatentState
from world_model.load_legacy import load_legacy_world_model
from dreamer.actor_critic import Actor


def state_features(state):
    return torch.cat([state.z, state.h[-1]], dim=-1)


def load_actor(actor_ckpt, device, feature_dim=288):
    actor = Actor(feature_dim=feature_dim).to(device)
    ckpt = torch.load(actor_ckpt, map_location=device)

    actor_state = (
        ckpt.get("actor_state_dict")
        or ckpt.get("actor")
        or ckpt.get("actor_state")
    )

    if actor_state is None:
        raise KeyError(f"Could not find actor state. Available keys: {ckpt.keys()}")

    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def encode_obs(world_model, obs, device):
    """
    Encode CarRacing observation.

    CarRacing gives 96x96x3 frames, but the legacy VAE expects 64x64x3,
    passed as torch tensor [B, C, 64, 64] in [0, 1].
    """
    obs_np = np.asarray(obs)

    if obs_np.ndim != 3:
        raise ValueError(f"Expected obs with shape [H, W, C], got {obs_np.shape}")

    if obs_np.shape[-1] > 3:
        obs_np = obs_np[..., :3]

    obs_bchw = (
        torch.tensor(obs_np, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    obs_bchw = F.interpolate(
        obs_bchw,
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )

    z = world_model.encode_obs(obs_bchw)

    if isinstance(z, tuple):
        z = z[0]

    return z.to(device)


def initial_state(world_model, z, device):
    state = world_model.initial_state(batch_size=1, device=device)
    return LatentState(z=z, h=state.h, c=state.c, extra={})


def extract_next_state(imagine_output):
    """
    Extract LatentState from the output of world_model.imagine_step.
    Some adapters return LatentState directly, others return ImagineOutput.
    """
    if isinstance(imagine_output, LatentState):
        return imagine_output

    for attr in ["next_state", "state", "latent_state"]:
        if hasattr(imagine_output, attr):
            state = getattr(imagine_output, attr)
            if isinstance(state, LatentState):
                return state

    print("Available ImagineOutput attributes:")
    print(dir(imagine_output))
    raise AttributeError("Could not extract LatentState from imagine_step output.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--actor-ckpt", type=str, required=True)
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--mdn-rnn-ckpt", type=str, required=True)
    parser.add_argument("--reward-ckpt", type=str, required=True)
    parser.add_argument("--reward-calibration", type=str, default=None)

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="data/latent/actor_real_entropy_3e-3")

    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
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

    actor = load_actor(args.actor_ckpt, device=device)

    env = gym.make("CarRacing-v3", render_mode=None, continuous=True)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)

        with torch.no_grad():
            z = encode_obs(world_model, obs, device)
            state = initial_state(world_model, z, device)

        z_list = [z.squeeze(0).detach().cpu().numpy()]
        h_next_list = []
        c_next_list = []
        action_list = []
        reward_list = []

        total_reward = 0.0

        for t in range(args.max_steps):
            with torch.no_grad():
                features = state_features(state)
                action_t, _ = actor.sample(features, deterministic=args.deterministic)

                action_np = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            total_reward += float(reward)

            with torch.no_grad():
                imagine_out = world_model.imagine_step(state, action_t)
                model_next_state = extract_next_state(imagine_out)

                # Encode the real next observation: teacher forcing on z.
                next_z_real = encode_obs(world_model, next_obs, device)

                h_next = model_next_state.h[-1].squeeze(0).detach().cpu().numpy()
                c_next = model_next_state.c[-1].squeeze(0).detach().cpu().numpy()

                state = LatentState(
                    z=next_z_real,
                    h=model_next_state.h,
                    c=model_next_state.c,
                    extra={},
                )

            action_list.append(action_np)
            reward_list.append(float(reward))
            h_next_list.append(h_next)
            c_next_list.append(c_next)
            z_list.append(next_z_real.squeeze(0).detach().cpu().numpy())

            if done:
                break

        path = out_dir / f"actor_real_ep{ep:04d}.npz"

        np.savez_compressed(
            path,
            z=np.asarray(z_list, dtype=np.float32),
            h_next=np.asarray(h_next_list, dtype=np.float32),
            c_next=np.asarray(c_next_list, dtype=np.float32),
            action=np.asarray(action_list, dtype=np.float32),
            reward=np.asarray(reward_list, dtype=np.float32),
        )

        print(f"episode {ep:04d} | steps={len(reward_list)} | return={total_reward:.2f} | saved={path}")

    env.close()


if __name__ == "__main__":
    main()