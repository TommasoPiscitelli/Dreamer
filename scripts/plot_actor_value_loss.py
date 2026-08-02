import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def plot_actor_value_loss(log_path, output_dir):
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    actor_losses = []
    value_losses = []

    pattern = re.compile(
        r"step\s+(\d+).*?"
        r"actor_loss=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?"
        r"value_loss=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            match = pattern.search(line)

            if match is None:
                continue

            steps.append(int(match.group(1)))
            actor_losses.append(float(match.group(2)))
            value_losses.append(float(match.group(3)))

    if not steps:
        raise ValueError(f"Nessun dato trovato in {log_path}")

    actor_output = output_dir / "actor_loss.png"
    value_output = output_dir / "value_loss.png"

    plt.figure(figsize=(10, 6))
    plt.plot(steps, actor_losses)
    plt.title("Actor loss")
    plt.xlabel("Training step")
    plt.ylabel("Actor loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(actor_output, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, value_losses)
    plt.title("Value loss")
    plt.xlabel("Training step")
    plt.ylabel("Value loss")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(value_output, dpi=200)
    plt.close()

    print(f"Salvato: {actor_output}")
    print(f"Salvato: {value_output}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    plot_actor_value_loss(
        log_path=args.log_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
