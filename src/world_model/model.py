from pathlib import Path

import torch

from world_model.world_model import HaWorldModelAdapter
from world_model.vae import ConvVAE
from world_model.mdn_rnn import MDNRNN, MDNRNNConfig
from world_model.reward_model import RewardClassifierExpectedReward


def load_state_dict(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict)

    return checkpoint


def load_legacy_world_model(
    vae_ckpt,
    mdn_rnn_ckpt,
    reward_ckpt,
    reward_calibration=None,
    device="cpu",
) -> HaWorldModelAdapter:
    """
    Load the frozen legacy world model:

        VAE + MDN-RNN + RewardClassifierExpectedReward
    """

    # Kept only for backward compatibility.
    # Reward calibration is ignored because we use the reward classifier.
    _ = reward_calibration

    device = torch.device(device)

    vae = ConvVAE(
        z_size=32,
        kl_tolerance=0.5,
    ).to(device)

    mdn_rnn = MDNRNN(
        MDNRNNConfig(
            input_size=35,      # z_dim + action_dim = 32 + 3
            output_size=32,
            hidden_size=256,
            num_layers=1,
            num_mixtures=5,
            dropout=0.0,
        )
    ).to(device)

    load_state_dict(vae, vae_ckpt, device)
    load_state_dict(mdn_rnn, mdn_rnn_ckpt, device)

    print(f"Using reward classifier: {reward_ckpt}")

    reward_model = RewardClassifierExpectedReward(
        checkpoint_path=reward_ckpt,
    ).to(device)

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