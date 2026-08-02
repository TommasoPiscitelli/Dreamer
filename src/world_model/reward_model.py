import torch
import torch.nn as nn


class RewardClassifier(nn.Module):
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
    Reward model usato dal world model.

    Input:
        features = concat(z, h), shape [B, 288]

    Output:
        expected reward, shape [B, 1]

    Il checkpoint del classifier contiene:
        - model_state_dict
        - feature_mean
        - feature_std
        - class_values
    """

    def __init__(self, classifier_ckpt, device):
        super().__init__()

        ckpt = torch.load(classifier_ckpt, map_location=device, weights_only=False)

        input_dim = int(ckpt.get("feature_dim", ckpt.get("input_dim", 288)))
        hidden_dim = int(ckpt.get("hidden_dim", 256))
        num_layers = int(ckpt.get("num_layers", 3))

        if "class_values" not in ckpt:
            raise KeyError(
                "Il checkpoint del reward classifier deve contenere la chiave 'class_values'."
            )

        class_values = torch.as_tensor(
            ckpt["class_values"],
            dtype=torch.float32,
            device=device,
        ).view(1, -1)

        num_classes = int(ckpt.get("num_classes", class_values.shape[-1]))

        self.classifier = RewardClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        ).to(device)

        if "model_state_dict" not in ckpt:
            raise KeyError(
                "Il checkpoint del reward classifier deve contenere la chiave 'model_state_dict'."
            )

        self.classifier.load_state_dict(ckpt["model_state_dict"])

        self.register_buffer(
            "feature_mean",
            torch.as_tensor(
                ckpt["feature_mean"],
                dtype=torch.float32,
                device=device,
            ).view(1, -1),
        )

        self.register_buffer(
            "feature_std",
            torch.clamp(
                torch.as_tensor(
                    ckpt["feature_std"],
                    dtype=torch.float32,
                    device=device,
                ).view(1, -1),
                min=1e-6,
            ),
        )

        self.register_buffer("class_values", class_values)

    def forward(self, features):
        if features.ndim == 1:
            features = features.unsqueeze(0)

        x = (features - self.feature_mean) / self.feature_std

        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)

        expected_reward = (probs * self.class_values).sum(dim=-1, keepdim=True)

        return expected_reward