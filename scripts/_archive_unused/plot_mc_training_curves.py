import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def pick_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"None of the candidate columns found: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Dreamer MC training without value model",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    step_col = pick_column(df, ["step"])
    actor_loss_col = pick_column(df, ["actor_loss"])
    return_col = pick_column(df, ["mean_return", "mc_return", "return"])
    reward_col = pick_column(df, ["mean_reward", "reward_sum"], required=False)
    entropy_col = pick_column(df, ["mean_entropy", "entropy"], required=False)

    print("Loaded:", csv_path)
    print("Columns:", list(df.columns))
    print()
    print(df.head())
    print()
    print(df.tail())

    # ---------------------------------------------------------
    # 1. MC return e reward
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        df[step_col],
        df[return_col],
        label=return_col,
        linewidth=2,
    )
    ax1.set_xlabel("Training step")
    ax1.set_ylabel(return_col)
    ax1.grid(True, alpha=0.3)

    if reward_col is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            df[step_col],
            df[reward_col],
            label=reward_col,
            linewidth=2,
            linestyle="--",
        )
        ax2.set_ylabel(reward_col)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    else:
        ax1.legend(loc="best")

    fig.suptitle(f"{args.title_prefix}: imagined return and reward")
    fig.tight_layout()

    out_path = out_dir / "mc_training_return_reward.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

    # ---------------------------------------------------------
    # 2. Actor loss
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df[step_col],
        df[actor_loss_col],
        label=actor_loss_col,
        linewidth=2,
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Actor loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle(f"{args.title_prefix}: actor loss")
    fig.tight_layout()

    out_path = out_dir / "mc_training_actor_loss.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

    # ---------------------------------------------------------
    # 3. Entropy
    # ---------------------------------------------------------
    if entropy_col is not None:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            df[step_col],
            df[entropy_col],
            label=entropy_col,
            linewidth=2,
        )
        ax.set_xlabel("Training step")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.suptitle(f"{args.title_prefix}: policy entropy")
        fig.tight_layout()

        out_path = out_dir / "mc_training_entropy.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print("Saved:", out_path)

    # ---------------------------------------------------------
    # 4. Azioni medie, se presenti
    # ---------------------------------------------------------
    action_cols = [c for c in ["steer", "gas", "brake"] if c in df.columns]

    if len(action_cols) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        for c in action_cols:
            ax.plot(
                df[step_col],
                df[c],
                label=c,
                linewidth=2,
            )

        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean action value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.suptitle(f"{args.title_prefix}: mean actions")
        fig.tight_layout()

        out_path = out_dir / "mc_training_mean_actions.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print("Saved:", out_path)

    print("\nFinal row:")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
