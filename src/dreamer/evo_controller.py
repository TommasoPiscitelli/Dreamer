import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def load_controller_weights(json_path, input_dim=288, action_dim=3):
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        obj = json.load(f)

    theta = np.asarray(obj["params"], dtype=np.float32)

    needed = input_dim * action_dim + action_dim

    if len(theta) != needed:
        raise ValueError(
            f"Unexpected controller size: got {len(theta)}, expected {needed}"
        )

    W = theta[: input_dim * action_dim].reshape(input_dim, action_dim)
    b = theta[input_dim * action_dim :]

    return W, b

class EvoController(nn.Module):
    """
    Controller evolutivo lineare.

    Input:
        features = concat(z, h), shape [B, 288]

    Output:
        action = [steer, gas, brake], shape [B, 3]
    """

    def __init__(self, W, b):
        super().__init__()

        self.register_buffer("W", torch.as_tensor(W, dtype=torch.float32))
        self.register_buffer("b", torch.as_tensor(b, dtype=torch.float32))

    @classmethod
    def from_json(cls, json_path, input_dim=288, action_dim=3):
        W, b = load_controller_weights(
            json_path=json_path,
            input_dim=input_dim,
            action_dim=action_dim,
        )
        return cls(W, b)

    def forward(self, features, deterministic=True):
        raw = features @ self.W + self.b

        steer = torch.tanh(raw[:, 0])
        gas = (torch.tanh(raw[:, 1]) + 1.0) / 2.0
        brake = torch.clamp(torch.tanh(raw[:, 2]), 0.0, 1.0)

        return torch.stack([steer, gas, brake], dim=-1)

    def act(self, features, deterministic=True):
        return self.forward(features, deterministic=deterministic)