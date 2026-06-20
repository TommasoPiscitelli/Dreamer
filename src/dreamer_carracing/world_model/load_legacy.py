from pathlib import Path
from typing import Any

import torch

from dreamer_carracing.legacy_world_models.vae import ConvVAE
from dreamer_carracing.legacy_world_models.mdn_rnn import MDNRNN, MDNRNNConfig
from dreamer_carracing.world_model import RewardModel, HaWorldModelAdapter


def _load_state_dict_flexible(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device):
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
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    model.load_state_dict(state_dict)
    return checkpoint


def load_legacy_world_model(
    vae_ckpt: str | Path,
    mdn_rnn_ckpt: str | Path,
    reward_ckpt: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> HaWorldModelAdapter:
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

    reward_model = RewardModel(feature_dim=32 + 256).to(device)

    _load_state_dict_flexible(vae, vae_ckpt, device)
    _load_state_dict_flexible(mdn_rnn, mdn_rnn_ckpt, device)

    if reward_ckpt is not None:
        _load_state_dict_flexible(reward_model, reward_ckpt, device)

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
    ).to(device)

    world_model.eval()
    return world_model
