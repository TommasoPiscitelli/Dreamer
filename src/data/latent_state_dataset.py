from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentStateDataset(Dataset):
    """
    Dataset di stati latenti salvati in file .npz.

    Ogni file deve contenere:
        z        [T+1, z_dim]
        h_next   [T, h_dim]
        c_next   [T, h_dim]

    Per ogni t >= 1:
        z_t = z[t]
        h_t = h_next[t - 1]
        c_t = c_next[t - 1]
    """

    def __init__(
        self,
        data_dir: str | Path,
        min_t: int = 1,
        max_t: Optional[int] = None,
        stride: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        self.index = []

        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                T = data["h_next"].shape[0]

            start = max(1, min_t)
            end = T if max_t is None else min(T, max_t)

            for t in range(start, end + 1, stride):
                self.index.append((file_idx, t))

        if not self.index:
            raise RuntimeError(f"No latent states found in {self.data_dir}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, t = self.index[idx]
        path = self.files[file_idx]

        with np.load(path) as data:
            z = data["z"][t].astype(np.float32)
            h = data["h_next"][t - 1].astype(np.float32)
            c = data["c_next"][t - 1].astype(np.float32)

        return {
            "z": torch.from_numpy(z.reshape(-1)),
            "h": torch.from_numpy(h.reshape(-1)),
            "c": torch.from_numpy(c.reshape(-1)),
            "file": path.name,
            "t": torch.tensor(t, dtype=torch.long),
        }