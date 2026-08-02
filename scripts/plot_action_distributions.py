from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


ACTION_COLS = ["steer", "gas", "brake"]


def check_columns(df: pd.DataFrame):
    required = ["rollout_id", "t", "reward", "steer", "gas", "brake"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Mancano le colonne richieste: {missing}. "
            f"Colonne disponibili: {list(df.columns)}"
        )


def action_range(action_name: str):
    if action_name == "steer":
        return (-1.0, 1.0)
    if action_name in ["gas", "brake"]:
        return (0.0, 1.0)
    return None


def plot_combined_histograms(df: pd.DataFrame, out_path: Path, bins: int, title: str | None):
    n_rollouts = df["rollout_id"].nunique()
    n_steps = len(df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, action_name in zip(axes, ACTION_COLS):
        values = df[action_name].dropna()

        ax.hist(
            values,
            bins=bins,
            range=action_range(action_name),
            edgecolor="black",
            alpha=0.8,
        )

        ax.set_title(action_name)
        ax.set_xlabel("Action value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        mean = values.mean()
        std = values.std()

        ax.text(
            0.03,
            0.97,
            f"mean = {mean:.3f}\nstd = {std:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", alpha=0.15),
        )

    full_title = title if title is not None else "Action distributions"
    fig.suptitle(
        f"{full_title}\nrollouts = {n_rollouts} | total steps = {n_steps}",
        fontsize=13,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a single figure with steer/gas/brake histograms."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV con colonne rollout_id,t,reward,steer,gas,brake",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path del file PNG finale",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Numero di bin",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Titolo del grafico",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    check_columns(df)

    plot_combined_histograms(
        df=df,
        out_path=args.out,
        bins=args.bins,
        title=args.title,
    )


if __name__ == "__main__":
    main()
