from pathlib import Path
import argparse
import time

import numpy as np

try:
    import gymnasium as gym
    GYMNASIUM_API = True
except ModuleNotFoundError:
    import gym
    GYMNASIUM_API = False


def make_carracing_env(env_id: str):
    """
    Supports both gymnasium and old gym.
    """
    try:
        return gym.make(env_id, continuous=True)
    except TypeError:
        return gym.make(env_id)
    except Exception:
        # Fallback for older gym versions.
        if env_id != "CarRacing-v0":
            print(f"Could not create {env_id}, trying CarRacing-v0...")
            return gym.make("CarRacing-v0")
        raise


def reset_env(env):
    out = env.reset()

    # gymnasium: obs, info
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs

    # old gym: obs
    return out


def step_env(env, action):
    out = env.step(action)

    # gymnasium: obs, reward, terminated, truncated, info
    if isinstance(out, tuple) and len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return next_obs, reward, terminated, truncated, done

    # old gym: obs, reward, done, info
    next_obs, reward, done, info = out
    terminated = done
    truncated = False
    return next_obs, reward, terminated, truncated, done


def sample_random_action(env):
    return env.action_space.sample().astype(np.float32)


def collect_episode(env, max_steps: int):
    obs_list = []
    action_list = []
    reward_list = []
    terminated_list = []
    truncated_list = []

    obs = reset_env(env)

    for _ in range(max_steps):
        action = sample_random_action(env)

        next_obs, reward, terminated, truncated, done = step_env(env, action)

        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(float(reward))
        terminated_list.append(bool(terminated))
        truncated_list.append(bool(truncated))

        obs = next_obs

        if done:
            break

    # Save final observation too, so obs has length T + 1.
    obs_list.append(obs)

    return {
        "obs": np.asarray(obs_list, dtype=np.uint8),
        "action": np.asarray(action_list, dtype=np.float32),
        "reward": np.asarray(reward_list, dtype=np.float32),
        "terminated": np.asarray(terminated_list, dtype=np.bool_),
        "truncated": np.asarray(truncated_list, dtype=np.bool_),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/raw/train")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--env-id", type=str, default="CarRacing-v2")
    args = parser.parse_args()

    print("Using backend:", "gymnasium" if GYMNASIUM_API else "gym")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_carracing_env(args.env_id)

    for ep in range(args.episodes):
        episode = collect_episode(env, max_steps=args.max_steps)

        filename = out_dir / f"{int(time.time() * 1000)}_{ep:05d}.npz"
        np.savez_compressed(filename, **episode)

        T = episode["action"].shape[0]
        ret = float(episode["reward"].sum())

        print(
            f"episode {ep + 1:04d}/{args.episodes} | "
            f"steps={T:4d} | return={ret:8.2f} | {filename}"
        )

    env.close()


if __name__ == "__main__":
    main()
