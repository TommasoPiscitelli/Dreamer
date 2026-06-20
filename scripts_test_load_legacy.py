import torch

from dreamer_carracing.world_model.load_legacy import load_legacy_world_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wm = load_legacy_world_model(
        vae_ckpt="checkpoints/legacy/vae/vae.pt",
        mdn_rnn_ckpt="checkpoints/legacy/mdn_rnn/mdn_rnn.pt",
        device=device,
    )

    B = 2
    obs = torch.randn(B, 3, 64, 64, device=device)
    action = torch.zeros(B, 3, device=device)

    state = wm.initial_state(B, torch.device(device))
    next_state = wm.observe_step(state, action, obs)
    out = wm.imagine_step(next_state, action)

    print("Loaded legacy world model.")
    print("z:", next_state.z.shape)
    print("h:", next_state.h.shape)
    print("reward:", out.reward.shape)


if __name__ == "__main__":
    main()
