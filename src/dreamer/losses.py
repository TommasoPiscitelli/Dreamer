from dataclasses import dataclass
import torch
import torch.nn.functional as F
from dreamer.imagination import ImaginedRollout


@dataclass
class BehaviorLosses:
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    lambda_returns: torch.Tensor

    mean_return: torch.Tensor
    mean_reward: torch.Tensor
    mean_value: torch.Tensor
    mean_entropy: torch.Tensor


def lambda_returns(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    values: torch.Tensor,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """
    Compute lambda returns for imagined trajectories.

    Args:
        rewards:
            Tensor of shape [H, B, 1].
            rewards[t] is the predicted reward after transitioning
            from state s_t to state s_{t+1}.

        discounts:
            Tensor of shape [H, B, 1].
            Usually gamma = 0.99.

        values:
            Tensor of shape [H + 1, B, 1].
            values[t] = V(s_t).
            The final value values[H] is used as bootstrap.

        lambda_:
            Lambda parameter for TD(lambda), usually 0.95.

    Returns:
        returns:
            Tensor of shape [H, B, 1].
            returns[t] is the target return for state s_t.
    """

    if rewards.ndim != 3:
        raise ValueError(f"rewards must have shape [H, B, 1], got {rewards.shape}")

    if discounts.shape != rewards.shape:
        raise ValueError(
            f"discounts must have same shape as rewards: "
            f"got discounts={discounts.shape}, rewards={rewards.shape}"
        )

    if values.ndim != 3:
        raise ValueError(f"values must have shape [H + 1, B, 1], got {values.shape}")

    horizon = rewards.shape[0]

    if values.shape[0] != horizon + 1:
        raise ValueError(
            f"values must have time dimension H + 1. "
            f"got values.shape[0]={values.shape[0]}, horizon={horizon}"
        )

    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")

    next_return = values[-1]
    returns = []

    for t in reversed(range(horizon)):
        next_value = values[t + 1]

        target = rewards[t] + discounts[t] * (
            (1.0 - lambda_) * next_value + lambda_ * next_return
        )

        returns.append(target)
        next_return = target

    returns.reverse()

    return torch.stack(returns, dim=0)

def compute_behavior_losses(
    rollout: ImaginedRollout,
    lambda_: float = 0.95,
    entropy_scale: float = 1e-4,
) -> BehaviorLosses:
    """
    Connect imagined rollout with lambda returns.

    rollout.rewards:   [H, B, 1]
    rollout.discounts: [H, B, 1]
    rollout.values:    [H + 1, B, 1]
    rollout.entropies: [H, B, 1]

    Lambda returns:
        targets[t] = r_t + gamma * ((1-lambda) V(s_{t+1}) + lambda targets[t+1])

    Actor objective:
        maximize lambda returns and encourage some entropy.

    Value objective:
        fit V(s_t) to the lambda return target.
    """

    targets = lambda_returns(
        rewards=rollout.rewards,
        discounts=rollout.discounts,
        values=rollout.values,
        lambda_=lambda_,
    )

    # values[:-1] corresponds to V(s_0), ..., V(s_{H-1})
    value_pred = rollout.values[:-1]

    # Actor: maximize imagined returns.
    # We minimize the negative objective.
    actor_loss = -targets.mean() - entropy_scale * rollout.entropies.mean()

    # Value: supervised regression toward lambda-return targets.
    # Detach targets so the value update does not backprop through the world model.
    value_loss = F.mse_loss(value_pred, targets.detach())

    return BehaviorLosses(
        actor_loss=actor_loss,
        value_loss=value_loss,
        lambda_returns=targets,
        mean_return=targets.mean().detach(),
        mean_reward=rollout.rewards.mean().detach(),
        mean_value=value_pred.mean().detach(),
        mean_entropy=rollout.entropies.mean().detach(),
    )