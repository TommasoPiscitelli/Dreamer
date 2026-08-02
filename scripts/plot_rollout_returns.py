from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def format_horizon_info(df: pd.DataFrame) -> str:
    if "horizon" not in df.columns:
        return "horizon: non disponibile"

    horizons = df["horizon"].dropna()

    if len(horizons) == 0:
        return "horizon: non disponibile"

    unique_horizons = sorted(horizons.unique())

    if len(unique_horizons) == 1:
        return f"horizon: {int(unique_horizons[0])} steps"

    return (
        f"horizon medio: {horizons.mean():.1f} steps\n"
        f"min/max horizon: {int(horizons.min())}/{int(horizons.max())}"
    )


def plot_returns(csv_path: Path, out_path: Path | None, title: str | None):
    df = pd.read_csv(csv_path)

    required_cols = ["discounted_return", "undiscounted_return"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Nel CSV mancano le colonne richieste: {missing}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    n_episodes = len(df)

    discounted_mean = df["discounted_return"].mean()
    undiscounted_mean = df["undiscounted_return"].mean()

    labels = ["Discounted return", "Undiscounted return"]
    values = [discounted_mean, undiscounted_mean]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(labels, values)

    ax.set_ylabel("Return medio")
    ax.set_title(title if title is not None else csv_path.stem)

    horizon_info = format_horizon_info(df)

    info_text = (
        f"episodi: {n_episodes}\n"
        f"{horizon_info}"
    )

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    for bar, value in zip(bars, values):
        height = bar.get_height()

        if height >= 0:
            va = "bottom"
            y = height
        else:
            va = "top"
            y = height

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.2f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    ax.axhline(0, linewidth=0.8)
    fig.tight_layout()

    if out_path is None:
        out_path = csv_path.with_suffix(".png")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Salvato grafico in: {out_path}")
    print(f"Episodi: {n_episodes}")
    print(f"Discounted return medio: {discounted_mean:.4f}")
    print(f"Undiscounted return medio: {undiscounted_mean:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera un grafico con discounted e undiscounted return medi da un CSV di rollout."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path del CSV con discounted_return e undiscounted_return.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path dell'immagine di output. Se omesso, usa lo stesso nome del CSV con estensione .png.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Titolo del grafico.",
    )

    args = parser.parse_args()

    plot_returns(
        csv_path=args.csv,
        out_path=args.out,
        title=args.title,
    )


if __name__ == "__main__":
    main()
