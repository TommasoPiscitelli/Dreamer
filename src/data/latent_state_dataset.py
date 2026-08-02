from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentStateDataset(Dataset):
    """
    Dataset of latent states sampled from cached CarRacing episodes.

    Each cached .npz file is expected to contain:
        z        [T+1, z_dim]
        h_next   [T, h_dim]
        c_next   [T, h_dim]

    A sample corresponds to a real latent state visited in the dataset:
        z_t      [z_dim]
        h_t      [h_dim]
        c_t      [h_dim]

    For t >= 1:
        z_t = z[t]
        h_t = h_next[t - 1]
        c_t = c_next[t - 1]

    We usually skip t=0 because h_0 and c_0 are just zeros.
    """

    def __init__(
        self,
        data_dir: str | Path,
        min_t: int = 1,
        max_t: Optional[int] = None,
        stride: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.min_t = min_t
        self.max_t = max_t
        self.stride = stride

        self.files = sorted(self.data_dir.glob("*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        self.index = []

        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                if "z" not in data.files:
                    raise KeyError(f"{path} does not contain 'z'. Keys: {data.files}")
                if "h_next" not in data.files:
                    raise KeyError(f"{path} does not contain 'h_next'. Keys: {data.files}")
                if "c_next" not in data.files:
                    raise KeyError(f"{path} does not contain 'c_next'. Keys: {data.files}")

                z = data["z"]
                h_next = data["h_next"]
                c_next = data["c_next"]

                T = h_next.shape[0]

                if z.shape[0] != T + 1:
                    raise ValueError(
                        f"Inconsistent shapes in {path}: "
                        f"z has length {z.shape[0]}, h_next has length {T}."
                    )

                if c_next.shape[0] != T:
                    raise ValueError(
                        f"Inconsistent shapes in {path}: "
                        f"c_next has length {c_next.shape[0]}, h_next has length {T}."
                    )

            start = max(1, self.min_t)
            end = T if self.max_t is None else min(T, self.max_t)

            for t in range(start, end + 1, self.stride):
                self.index.append((file_idx, t))

        if not self.index:
            raise RuntimeError(
                f"No latent states found in {self.data_dir} "
                f"with min_t={self.min_t}, max_t={self.max_t}, stride={self.stride}."
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, t = self.index[idx]
        path = self.files[file_idx]

        with np.load(path) as data:
            z = data["z"][t].astype(np.float32).reshape(-1)
            h = data["h_next"][t - 1].astype(np.float32).reshape(-1)
            c = data["c_next"][t - 1].astype(np.float32).reshape(-1)

        return {
            "z": torch.from_numpy(z),
            "h": torch.from_numpy(h),
            "c": torch.from_numpy(c),
            "file": path.name,
            "t": torch.tensor(t, dtype=torch.long),
        }