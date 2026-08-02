import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Dreamer actor-value training",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = [
        "step",
        "actor_loss",
        "value_loss",
        "mean_return",
        "mean_reward",
        "mean_value",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ---------------------------------------------------------
    # 1. mean_return, mean_value, mean_reward
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df["step"], df["mean_return"], label="mean_return", linewidth=2)
    ax1.plot(df["step"], df["mean_value"], label="mean_value", linewidth=2)

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Mean return / mean value")
    ax1.grid(True, alpha=0.3)

    # mean_reward ha scala molto più piccola, quindi lo mettiamo su secondo asse
    ax2 = ax1.twinx()
    ax2.plot(df["step"], df["mean_reward"], label="mean_reward", linewidth=2, linestyle="--")
    ax2.set_ylabel("Mean reward")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.suptitle(f"{args.title_prefix}: imagined return/value/reward")
    fig.tight_layout()

    out_path = out_dir / "training_mean_return_value_reward.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")

    # ---------------------------------------------------------
    # 2. actor_loss e value_loss
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df["step"], df["actor_loss"], label="actor_loss", linewidth=2)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Actor loss")
    ax1.grid(True, alpha=0.3)

    # value_loss può avere scala diversa, quindi secondo asse
    ax2 = ax1.twinx()
    ax2.plot(df["step"], df["value_loss"], label="value_loss", linewidth=2, linestyle="--")
    ax2.set_ylabel("Value loss")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.suptitle(f"{args.title_prefix}: actor and value losses")
    fig.tight_layout()

    out_path = out_dir / "training_actor_value_losses.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")

    # ---------------------------------------------------------
    # Summary testuale
    # ---------------------------------------------------------
    print("\nFinal row:")
    print(df.tail(1).to_string(index=False))

    print("\nColumns summary:")
    print(
        df[
            [
                "actor_loss",
                "value_loss",
                "mean_return",
                "mean_reward",
                "mean_value",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()
