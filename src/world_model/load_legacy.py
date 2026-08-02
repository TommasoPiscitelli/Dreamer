from pathlib import Path
import json

import torch
import torch.nn as nn

from world_model.vae import ConvVAE
from world_model.mdn_rnn import MDNRNN, MDNRNNConfig
from world_model import RewardModel, HaWorldModelAdapter


def _load_state_dict_flexible(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            f"Unsupported checkpoint type in {checkpoint_path}: {type(checkpoint)}"
        )

    model.load_state_dict(state_dict)
    return checkpoint



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



def load_legacy_world_model(
    vae_ckpt,
    mdn_rnn_ckpt,
    reward_ckpt=None,
    reward_calibration=None,
    device="cpu",
) -> HaWorldModelAdapter:
    """
    Carica il World Model legacy:

        VAE + MDN-RNN + RewardClassifierExpectedReward

    Nota:
        In questa versione reward_ckpt deve essere il checkpoint del reward classifier,
        ad esempio:

            checkpoints/reward_classifier/reward_classifier.pt

        Non deve essere il vecchio reward_model.pt regressivo.
    """

    device = torch.device(device)

    vae = ConvVAE(z_size=32, kl_tolerance=0.5).to(device)

    mdn_config = MDNRNNConfig(
        input_size=35,
        output_size=32,
        hidden_size=256,
        num_layers=1,
        num_mixtures=5,
        dropout=0.0,
    )

    mdn_rnn = MDNRNN(mdn_config).to(device)

    _load_state_dict_flexible(vae, vae_ckpt, device)
    _load_state_dict_flexible(mdn_rnn, mdn_rnn_ckpt, device)

    if reward_ckpt is None:
        raise ValueError(
            "reward_ckpt è obbligatorio: passa il checkpoint del reward classifier, "
            "per esempio checkpoints/reward_classifier/reward_classifier.pt"
        )

    print(f"Using reward classifier: {reward_ckpt}")

    reward_model = RewardClassifierExpectedReward(
        classifier_ckpt=reward_ckpt,
        device=device,
    ).to(device)

    if reward_calibration is not None:
        print(
            "Nota: reward_calibration viene ignorato perché stai usando "
            "il reward classifier."
        )

    world_model = HaWorldModelAdapter(
        vae=vae,
        mdn_rnn=mdn_rnn,
        reward_model=reward_model,
        z_dim=32,
        h_dim=256,
        action_dim=3,
        num_layers=1,
        discount=0.99,
        freeze_vae=True,
        freeze_mdn_rnn=True,
        reward_scale=1.0,
        reward_bias=0.0,
    ).to(device)

    world_model.eval()

    return world_model
