import torch


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
            Usually gamma, for example 0.99.

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