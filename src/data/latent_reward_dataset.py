from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentRewardDataset(Dataset):
    """
    Dataset over cached latent features.

    Expected .npz format:
        features: [T, feature_dim]
        reward:   [T]

    Returns:
        features: [feature_dim]
        reward:   [1]
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        self.index: list[tuple[int, int]] = []

        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                T = data["reward"].shape[0]

            for t in range(T):
                self.index.append((file_idx, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, t = self.index[idx]
        path = self.files[file_idx]

        with np.load(path) as data:
            features = data["features"][t].astype(np.float32)
            reward = np.asarray([data["reward"][t]], dtype=np.float32)

        return {
            "features": torch.from_numpy(features),
            "reward": torch.from_numpy(reward),
        }
