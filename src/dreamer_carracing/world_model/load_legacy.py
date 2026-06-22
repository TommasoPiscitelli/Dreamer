from pathlib import Path
import json

import torch

from dreamer_carracing.legacy_world_models.vae import ConvVAE
from dreamer_carracing.legacy_world_models.mdn_rnn import MDNRNN, MDNRNNConfig
from dreamer_carracing.world_model import RewardModel, HaWorldModelAdapter


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


def load_legacy_world_model(
    vae_ckpt,
    mdn_rnn_ckpt,
    reward_ckpt=None,
    reward_calibration=None,
    device="cpu",
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

    reward_scale = 1.0
    reward_bias = 0.0

    if reward_calibration is not None:
        reward_calibration = Path(reward_calibration)

        if not reward_calibration.exists():
            raise FileNotFoundError(
                f"Reward calibration file not found: {reward_calibration}"
            )

        with open(reward_calibration, "r") as f:
            calibration = json.load(f)

        reward_scale = float(calibration["scale"])
        reward_bias = float(calibration["bias"])

        print(
            f"Using reward calibration: "
            f"scale={reward_scale:.6f}, bias={reward_bias:.6f}"
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
        reward_scale=reward_scale,
        reward_bias=reward_bias,
    ).to(device)

    world_model.eval()

    return world_model
