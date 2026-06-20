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
    try:
        return gym.make(env_id, continuous=True)
    except TypeError:
        return gym.make(env_id)
    except Exception:
        if env_id != "CarRacing-v0":
            print(f"Could not create {env_id}, trying CarRacing-v0...")
            return gym.make("CarRacing-v0")
        raise


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs
    return out


def step_env(env, action):
    out = env.step(action)

    if isinstance(out, tuple) and len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return next_obs, reward, terminated, truncated, done

    next_obs, reward, done, info = out
    terminated = done
    truncated = False
    return next_obs, reward, terminated, truncated, done


class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def reset(self):
        pass

    def __call__(self, obs):
        return self.env.action_space.sample().astype(np.float32)


class ForwardPolicy:
    """
    Simple non-random baseline:
    mostly goes forward, with small steering noise.
    Useful to diversify from fully random actions.
    """

    def __init__(self, noise: float = 0.05):
        self.noise = noise

    def reset(self):
        pass

    def __call__(self, obs):
        steer = np.random.normal(0.0, self.noise)
        gas = 0.55
        brake = 0.0
        return np.array(
            [
                np.clip(steer, -1.0, 1.0),
                np.clip(gas, 0.0, 1.0),
                np.clip(brake, 0.0, 1.0),
            ],
            dtype=np.float32,
        )


class RoadFollowingHeuristicPolicy:
    """
    Simple visual heuristic for CarRacing.

    It looks for grey road pixels in front of the car and steers toward
    the horizontal centroid of the detected road region.

    Action format:
        [steering, gas, brake]
    """

    def __init__(
        self,
        steer_gain: float = 1.7,
        gas_straight: float = 0.65,
        gas_curve: float = 0.35,
        brake_curve: float = 0.05,
        action_noise: float = 0.05,
        epsilon_random: float = 0.02,
        smooth: float = 0.70,
    ):
        self.steer_gain = steer_gain
        self.gas_straight = gas_straight
        self.gas_curve = gas_curve
        self.brake_curve = brake_curve
        self.action_noise = action_noise
        self.epsilon_random = epsilon_random
        self.smooth = smooth
        self.prev_steer = 0.0

    def reset(self):
        self.prev_steer = 0.0

    def __call__(self, obs):
        if np.random.rand() < self.epsilon_random:
            return np.array(
                [
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(0.0, 1.0),
                    np.random.uniform(0.0, 0.3),
                ],
                dtype=np.float32,
            )

        obs_f = obs.astype(np.float32)

        # Region in front of the car. CarRacing observations are 96x96.
        # The car is near the lower center; the road ahead is mostly above it.
        roi = obs_f[35:80, :, :]  # [H_roi, 96, 3]

        r = roi[:, :, 0]
        g = roi[:, :, 1]
        b = roi[:, :, 2]
        mean = (r + g + b) / 3.0

        # Road is approximately grey: channels close to each other,
        # not too dark and not too bright.
        greyish = (
            (np.abs(r - g) < 35.0)
            & (np.abs(g - b) < 35.0)
            & (np.abs(r - b) < 35.0)
        )
        brightness = (mean > 45.0) & (mean < 230.0)

        road_mask = greyish & brightness

        ys, xs = np.nonzero(road_mask)

        if len(xs) < 20:
            # If road detection fails, keep previous direction and slow down.
            steer = self.prev_steer
            gas = 0.25
            brake = 0.05
        else:
            # Weight lower rows more: closer road matters more.
            weights = 1.0 + ys.astype(np.float32) / max(1.0, roi.shape[0] - 1)
            target_x = np.average(xs.astype(np.float32), weights=weights)

            center_x = obs.shape[1] / 2.0
            error = (target_x - center_x) / center_x

            raw_steer = self.steer_gain * error

            steer = self.smooth * self.prev_steer + (1.0 - self.smooth) * raw_steer
            steer = float(np.clip(steer, -1.0, 1.0))

            abs_steer = abs(steer)
            gas = self.gas_straight if abs_steer < 0.35 else self.gas_curve
            brake = 0.0 if abs_steer < 0.65 else self.brake_curve

        steer += np.random.normal(0.0, self.action_noise)

        action = np.array(
            [
                np.clip(steer, -1.0, 1.0),
                np.clip(gas, 0.0, 1.0),
                np.clip(brake, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        self.prev_steer = float(action[0])
        return action


def build_policy(name: str, env, args):
    if name == "random":
        return RandomPolicy(env)

    if name == "forward":
        return ForwardPolicy(noise=args.action_noise)

    if name == "heuristic":
        return RoadFollowingHeuristicPolicy(
            steer_gain=args.steer_gain,
            action_noise=args.action_noise,
            epsilon_random=args.epsilon_random,
        )

    raise ValueError(f"Unknown policy: {name}")


def collect_episode(env, policy, max_steps: int):
    obs_list = []
    action_list = []
    reward_list = []
    terminated_list = []
    truncated_list = []

    obs = reset_env(env)
    policy.reset()

    for _ in range(max_steps):
        action = policy(obs)

        next_obs, reward, terminated, truncated, done = step_env(env, action)

        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(float(reward))
        terminated_list.append(bool(terminated))
        truncated_list.append(bool(truncated))

        obs = next_obs

        if done:
            break

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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--env-id", type=str, default="CarRacing-v2")

    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "forward", "heuristic"],
        default="heuristic",
    )

    parser.add_argument("--steer-gain", type=float, default=1.7)
    parser.add_argument("--action-noise", type=float, default=0.05)
    parser.add_argument("--epsilon-random", type=float, default=0.02)

    args = parser.parse_args()

    print("Using backend:", "gymnasium" if GYMNASIUM_API else "gym")
    print("Policy:", args.policy)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_carracing_env(args.env_id)
    policy = build_policy(args.policy, env, args)

    returns = []

    for ep in range(args.episodes):
        episode = collect_episode(env, policy, max_steps=args.max_steps)

        filename = out_dir / f"{int(time.time() * 1000)}_{args.policy}_{ep:05d}.npz"
        np.savez_compressed(filename, **episode)

        T = episode["action"].shape[0]
        ret = float(episode["reward"].sum())
        returns.append(ret)

        print(
            f"episode {ep + 1:04d}/{args.episodes} | "
            f"steps={T:4d} | return={ret:8.2f} | "
            f"mean_return={np.mean(returns):8.2f} | {filename}",
            flush=True,
        )

    env.close()

    print()
    print(f"Collected {args.episodes} episodes.")
    print(f"Mean return: {np.mean(returns):.2f}")
    print(f"Std return:  {np.std(returns):.2f}")
    print(f"Min return:  {np.min(returns):.2f}")
    print(f"Max return:  {np.max(returns):.2f}")


if __name__ == "__main__":
    main()
