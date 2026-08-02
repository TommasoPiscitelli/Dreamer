import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--title", type=str, default="Real Environment Policy Summary")
    parser.add_argument("--ma-window", type=int, default=5)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    required_cols = ["file", "steps", "return", "mean_reward", "min_reward", "max_reward"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano le colonne: {missing}")

    returns = df["return"].to_numpy()
    steps = df["steps"].to_numpy()
    mean_rewards = df["mean_reward"].to_numpy()

    n = len(df)
    mean_ret = returns.mean()
    median_ret = np.median(returns)
    std_ret = returns.std()
    min_ret = returns.min()
    max_ret = returns.max()

    mean_steps = steps.mean()
    min_steps = steps.min()
    max_steps = steps.max()

    frac_pos = (returns > 0).mean()
    frac_50 = (returns > 50).mean()

    if args.out is None:
        out_path = csv_path.parent / "policy_real_env_summary.png"
    else:
        out_path = Path(args.out)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) Return per episodio
    ax = axes[0, 0]
    ax.plot(np.arange(n), returns, marker="o", linestyle="-", linewidth=1)
    ax.axhline(mean_ret, linestyle="--", linewidth=1, label=f"mean = {mean_ret:.2f}")
    if n >= args.ma_window:
        ma = moving_average(returns, args.ma_window)
        ax.plot(
            np.arange(len(ma)) + args.ma_window - 1,
            ma,
            linewidth=2,
            label=f"moving avg ({args.ma_window})"
        )
    ax.set_title("Episode returns")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()

    # 2) Istogramma dei return
    ax = axes[0, 1]
    ax.hist(returns, bins=min(15, max(5, n)), alpha=0.8)
    ax.axvline(mean_ret, linestyle="--", linewidth=1, label=f"mean = {mean_ret:.2f}")
    ax.axvline(median_ret, linestyle=":", linewidth=1, label=f"median = {median_ret:.2f}")
    ax.set_title("Return distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Count")
    ax.legend()

    # 3) Steps per episodio
    ax = axes[1, 0]
    ax.plot(np.arange(n), steps, marker="o", linestyle="-", linewidth=1, label="steps")
    ax.axhline(mean_steps, linestyle="--", linewidth=1, label=f"mean = {mean_steps:.1f}")
    ax.set_title("Episode length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()

    # 4) Riquadro statistiche
    ax = axes[1, 1]
    ax.axis("off")
    text = (
        f"Policy summary\n"
        f"{'-'*30}\n"
        f"Episodes: {n}\n\n"
        f"Return mean:    {mean_ret:.2f}\n"
        f"Return median:  {median_ret:.2f}\n"
        f"Return std:     {std_ret:.2f}\n"
        f"Return min:     {min_ret:.2f}\n"
        f"Return max:     {max_ret:.2f}\n\n"
        f"Mean steps:     {mean_steps:.1f}\n"
        f"Min steps:      {min_steps}\n"
        f"Max steps:      {max_steps}\n\n"
        f"Frac(return > 0):   {frac_pos:.2%}\n"
        f"Frac(return > 50):  {frac_50:.2%}\n\n"
        f"Mean of mean_reward: {mean_rewards.mean():.4f}\n"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, family="monospace")

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot to:", out_path)


if __name__ == "__main__":
    main()