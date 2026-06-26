from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dreamer_carracing.dreamer.returns import lambda_returns
from dreamer_carracing.dreamer.imagination import ImaginedRollout


@dataclass
class BehaviorLosses:
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    lambda_returns: torch.Tensor

    mean_return: torch.Tensor
    mean_reward: torch.Tensor
    mean_value: torch.Tensor
    mean_entropy: torch.Tensor


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