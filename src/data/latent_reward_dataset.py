from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentRewardDataset(Dataset):
    """
    Dataset di feature latenti e reward.

    Ogni file .npz deve contenere:
        features: [T, feature_dim]
        reward:   [T]
    """

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.npz"))

        if not files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        all_features = []
        all_rewards = []

        for path in files:
            with np.load(path) as data:
                features = data["features"].astype(np.float32)
                rewards = data["reward"].astype(np.float32).reshape(-1, 1)

            all_features.append(features)
            all_rewards.append(rewards)

        self.features = torch.from_numpy(np.concatenate(all_features, axis=0))
        self.rewards = torch.from_numpy(np.concatenate(all_rewards, axis=0))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "reward": self.rewards[idx],
        }