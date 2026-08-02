import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PATTERN = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+"
    r"actor_loss=\s*(?P<actor_loss>[-+]?\d*\.?\d+)\s+\|\s+"
    r"value_loss=\s*(?P<value_loss>[-+]?\d*\.?\d+)\s+\|\s+"
    r"mean_return=\s*(?P<mean_return>[-+]?\d*\.?\d+)\s+\|\s+"
    r"mean_reward=\s*(?P<mean_reward>[-+]?\d*\.?\d+)\s+\|\s+"
    r"mean_value=\s*(?P<mean_value>[-+]?\d*\.?\d+)\s+\|\s+"
    r"mean_entropy=\s*(?P<mean_entropy>[-+]?\d*\.?\d+)"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="logs/training_log_analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    with open(args.log, "r") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                row = {k: float(v) for k, v in m.groupdict().items()}
                row["step"] = int(row["step"])
                rows.append(row)

    if not rows:
        raise RuntimeError("No training rows found in log.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "training_log_parsed.csv", index=False)

    print(df.tail())
    print()
    print("Summary:")
    print(df.describe())

    for col in ["actor_loss", "value_loss", "mean_return", "mean_reward", "mean_value", "mean_entropy"]:
        plt.figure(figsize=(8, 5))
        plt.plot(df["step"], df[col])
        plt.xlabel("Training step")
        plt.ylabel(col)
        plt.title(col)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["mean_return"], label="mean_return")
    plt.plot(df["step"], df["mean_value"], label="mean_value")
    plt.xlabel("Training step")
    plt.ylabel("Value")
    plt.title("Mean return vs mean value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_return_vs_mean_value.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["mean_reward"], label="mean_reward")
    plt.xlabel("Training step")
    plt.ylabel("Predicted reward")
    plt.title("Mean predicted reward during training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_reward.png", dpi=200)
    plt.close()

    print()
    print("Saved plots in:", out_dir)


if __name__ == "__main__":
    main()