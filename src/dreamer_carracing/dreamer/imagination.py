from dataclasses import dataclass

import torch

from dreamer_carracing.world_model.api import LatentState


@dataclass
class ImaginedRollout:
    """
    Container for imagined trajectories.

    Shapes:
        features:  [H + 1, B, feature_dim]
        actions:   [H, B, action_dim]
        rewards:   [H, B, 1]
        discounts: [H, B, 1]
        values:    [H + 1, B, 1]
        entropies: [H, B, 1]
    """

    start_state: LatentState
    final_state: LatentState

    features: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    values: torch.Tensor
    entropies: torch.Tensor


def imagine_rollout(
    world_model,
    actor,
    value,
    start_state: LatentState,
    horizon: int,
    deterministic: bool = False,
) -> ImaginedRollout:
    """
    Generate an imagined rollout starting from a real latent state.

    The computation is differentiable. Do not wrap this function in torch.no_grad()
    during actor training, otherwise gradients cannot flow from imagined rewards
    back to the actor through the world model.

    Args:
        world_model:
            Frozen world model adapter with imagine_step(state, action).

        actor:
            Actor network. Takes state.features and returns actions.

        value:
            Value network. Takes state.features and returns V(s).

        start_state:
            Initial latent state sampled from real cached latent data.

        horizon:
            Number of imagined steps H.

        deterministic:
            If True, use mean actor action. Useful for debugging/evaluation.
            If False, use reparameterized stochastic action.

    Returns:
        ImaginedRollout with tensors stacked over time.
    """

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    state = start_state

    features = [state.features]
    values = [value(state.features)]

    actions = []
    rewards = []
    discounts = []
    entropies = []

    for _ in range(horizon):
        action, entropy = actor.sample(
            state.features,
            deterministic=deterministic,
        )

        imagined = world_model.imagine_step(state, action)

        actions.append(action)
        rewards.append(imagined.reward)
        discounts.append(imagined.discount)
        entropies.append(entropy)

        state = imagined.next_state

        features.append(state.features)
        values.append(value(state.features))

    return ImaginedRollout(
        start_state=start_state,
        final_state=state,
        features=torch.stack(features, dim=0),
        actions=torch.stack(actions, dim=0),
        rewards=torch.stack(rewards, dim=0),
        discounts=torch.stack(discounts, dim=0),
        values=torch.stack(values, dim=0),
        entropies=torch.stack(entropies, dim=0),
    )