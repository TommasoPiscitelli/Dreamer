import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
    ):
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
    Wrapper used as reward model inside the world model.

    Input:
        features = concat(z, h), shape [..., feature_dim]

    Output:
        expected reward, shape [..., 1]
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        input_dim = int(ckpt.get("feature_dim", ckpt.get("input_dim", 288)))
        hidden_dim = int(ckpt.get("hidden_dim", 256))
        num_layers = int(ckpt.get("num_layers", 3))
        num_classes = int(ckpt.get("num_classes", len(ckpt["class_values"])))

        self.classifier = RewardClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )

        self.classifier.load_state_dict(ckpt["model_state_dict"])

        self.register_buffer(
            "feature_mean",
            torch.as_tensor(ckpt["feature_mean"], dtype=torch.float32).view(1, -1),
        )

        self.register_buffer(
            "feature_std",
            torch.clamp(
                torch.as_tensor(ckpt["feature_std"], dtype=torch.float32).view(1, -1),
                min=1e-6,
            ),
        )

        self.register_buffer(
            "class_values",
            torch.as_tensor(ckpt["class_values"], dtype=torch.float32).view(1, -1),
        )

        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

    def forward(self, features):
        original_shape = features.shape[:-1]

        x = features.reshape(-1, features.shape[-1])
        x = (x - self.feature_mean) / self.feature_std

        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)

        expected_reward = (probs * self.class_values).sum(dim=-1, keepdim=True)

        return expected_reward.reshape(*original_shape, 1)