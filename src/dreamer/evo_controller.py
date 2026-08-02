import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _is_numeric_sequence(x):
    if not isinstance(x, list):
        return False
    if len(x) == 0:
        return False
    return all(isinstance(v, (int, float)) for v in x)


def _find_numeric_vector(obj):
    """
    Cerca ricorsivamente il vettore dei parametri dentro il JSON.
    Supporta sia JSON come lista diretta, sia dizionari con chiavi tipo
    params, theta, weights, solution, best, controller.
    """
    if _is_numeric_sequence(obj):
        return np.asarray(obj, dtype=np.float32)

    if isinstance(obj, dict):
        preferred_keys = [
            "params",
            "theta",
            "weights",
            "solution",
            "best",
            "best_params",
            "controller",
            "controller_params",
        ]

        for k in preferred_keys:
            if k in obj:
                try:
                    v = _find_numeric_vector(obj[k])
                    if v is not None:
                        return v
                except Exception:
                    pass

        candidates = []
        for v in obj.values():
            try:
                vv = _find_numeric_vector(v)
                if vv is not None:
                    candidates.append(vv)
            except Exception:
                pass

        if candidates:
            return max(candidates, key=len)

    if isinstance(obj, list):
        candidates = []
        for v in obj:
            try:
                vv = _find_numeric_vector(v)
                if vv is not None:
                    candidates.append(vv)
            except Exception:
                pass

        if candidates:
            return max(candidates, key=len)

    return None


def load_controller_weights(json_path, input_dim=288, action_dim=3):
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        obj = json.load(f)

    # Caso 1: JSON con W/b espliciti.
    if isinstance(obj, dict):
        w_key = None
        b_key = None

        for k in ["W", "w", "weight", "weights", "matrix"]:
            if k in obj:
                w_key = k
                break

        for k in ["b", "bias", "biases"]:
            if k in obj:
                b_key = k
                break

        if w_key is not None and b_key is not None:
            W = np.asarray(obj[w_key], dtype=np.float32)
            b = np.asarray(obj[b_key], dtype=np.float32)

            if W.shape == (action_dim, input_dim):
                W = W.T

            if W.shape != (input_dim, action_dim):
                raise ValueError(f"Unexpected W shape: {W.shape}")

            if b.shape != (action_dim,):
                b = b.reshape(action_dim)

            return W, b

    # Caso 2: JSON come vettore piatto.
    theta = _find_numeric_vector(obj)

    if theta is None:
        raise ValueError(f"Could not find numeric parameter vector in {json_path}")

    needed = input_dim * action_dim + action_dim

    if len(theta) < needed:
        raise ValueError(
            f"Controller vector too short: len={len(theta)}, required at least {needed}"
        )

    if len(theta) > needed:
        print(
            f"[warning] controller vector has len={len(theta)}, "
            f"using first {needed} parameters"
        )
        theta = theta[:needed]

    W = theta[: input_dim * action_dim].reshape(input_dim, action_dim)
    b = theta[input_dim * action_dim : input_dim * action_dim + action_dim]

    return W.astype(np.float32), b.astype(np.float32)


class EvoController(nn.Module):
    """
    Wrapper PyTorch per il controller evolutivo del vecchio progetto World Models.

    Input:
        features = concat(z, h), shape [B, 288]

    Output:
        action = [steer, gas, brake], shape [B, 3]
    """

    def __init__(self, W, b):
        super().__init__()

        W = torch.as_tensor(W, dtype=torch.float32)
        b = torch.as_tensor(b, dtype=torch.float32)

        self.register_buffer("W", W)
        self.register_buffer("b", b)

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
