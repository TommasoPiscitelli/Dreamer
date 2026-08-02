import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardClassifierNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        super().__init__()

        layers = []
        dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim

        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RewardClassifierExpectedReward(nn.Module):
    """
    Wrapper usato dentro Dreamer.

    Input:
        features = concat(z, h), shape [..., 288]

    Output:
        expected reward, shape [..., 1]

    Il reward è:
        E[r] = sum_k P(class k | features) * class_value_k
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        input_dim = int(ckpt.get("feature_dim", ckpt.get("input_dim", 288)))
        hidden_dim = int(ckpt.get("hidden_dim", 256))
        num_layers = int(ckpt.get("num_layers", 3))
        num_classes = int(ckpt.get("num_classes", 5))

        self.class_names = ckpt.get("class_names", None)

        self.model = RewardClassifierNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )

        self.model.load_state_dict(ckpt["model_state_dict"])

        feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
        class_values = np.asarray(ckpt["class_values"], dtype=np.float32)

        self.register_buffer("feature_mean", torch.tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.tensor(feature_std, dtype=torch.float32))
        self.register_buffer("class_values", torch.tensor(class_values, dtype=torch.float32))

        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

    def forward(self, features):
        original_shape = features.shape[:-1]

        x = features.reshape(-1, features.shape[-1])
        x = (x - self.feature_mean) / self.feature_std

        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)

        expected_reward = (probs * self.class_values[None, :]).sum(dim=-1, keepdim=True)

        return expected_reward.reshape(*original_shape, 1)
