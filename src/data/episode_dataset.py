from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class CarRacingEpisodeDataset(Dataset):
    """
    Loads CarRacing .npz episodes and returns fixed-length subsequences.

    Expected .npz format:
        obs:    [T + 1, 96, 96, 3], uint8
        action: [T, 3]
        reward: [T]

    Returned batch item:
        obs:     [seq_len + 1, 3, 64, 64], float32 in [0, 1]
        actions: [seq_len, 3], float32
        rewards: [seq_len], float32
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = 50,
        image_size: int = 64,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.image_size = image_size

        self.files = sorted(self.data_dir.glob("*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        self.index: list[tuple[int, int]] = []

        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                T = data["action"].shape[0]

            # Need obs[start : start + seq_len + 1]
            # and actions/rewards[start : start + seq_len].
            max_start = T - seq_len
            for start in range(max_start + 1):
                self.index.append((file_idx, start))

        if not self.index:
            raise ValueError(
                f"No valid subsequences. Try smaller seq_len. Got seq_len={seq_len}"
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]
        path = self.files[file_idx]

        with np.load(path) as data:
            obs = data["obs"][start : start + self.seq_len + 1]
            actions = data["action"][start : start + self.seq_len]
            rewards = data["reward"][start : start + self.seq_len]

        # obs: [T+1, H, W, C] uint8 -> [T+1, C, H, W] float32 in [0, 1]
        obs = torch.from_numpy(obs).float() / 255.0
        obs = obs.permute(0, 3, 1, 2).contiguous()

        if obs.shape[-1] != self.image_size or obs.shape[-2] != self.image_size:
            obs = F.interpolate(
                obs,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        actions = torch.from_numpy(actions.astype(np.float32))
        rewards = torch.from_numpy(rewards.astype(np.float32))

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
        }
