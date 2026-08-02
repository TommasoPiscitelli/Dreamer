from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CarRacingEpisodeDataset(Dataset):
    """
    Dataset di episodi CarRacing.

    Ogni file .npz deve contenere:
        obs:    [T + 1, 96, 96, 3]
        action: [T, 3]
        reward: [T]

    Restituisce sottosequenze di lunghezza seq_len.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = 50,
        image_size: int = 64,
    ):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        self.seq_len = seq_len
        self.image_size = image_size

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        self.index = []

        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                T = len(data["action"])

            for start in range(T - seq_len + 1):
                self.index.append((file_idx, start))

        if not self.index:
            raise ValueError(f"No valid subsequences with seq_len={seq_len}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]
        path = self.files[file_idx]

        end = start + self.seq_len

        with np.load(path) as data:
            obs = data["obs"][start : end + 1]
            actions = data["action"][start:end]
            rewards = data["reward"][start:end]

        obs = torch.from_numpy(obs).float() / 255.0
        obs = obs.permute(0, 3, 1, 2)

        if self.image_size is not None:
            obs = F.interpolate(
                obs,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "obs": obs,
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "rewards": torch.from_numpy(rewards.astype(np.float32)),
        }