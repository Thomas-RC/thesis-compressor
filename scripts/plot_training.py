#!/usr/bin/env python3
"""Plot training curves from one or more training CSVs.

Each CSV must have columns: step, phase, loss_nat, bpc, lr, wall_s
Produces a 2x1 figure: BPC over steps (top) and learning rate (bottom).

Usage:
  python3 scripts/plot_training.py results/long_medium_20k.csv
  python3 scripts/plot_training.py \\
      --label medium results/long_medium_20k.csv \\
      --label large  results/long_large_20k.csv \\
      --output results/plots/long_compare.png
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"step", "phase", "loss_nat", "bpc", "lr", "wall_s"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: brakujące kolumny: {missing}")
    return df


def plot_runs(runs: list[tuple[str, pd.DataFrame]], output: Path, gzip_baseline: float | None) -> None:
    fig, (ax_bpc, ax_lr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    for label, df in runs:
        train = df[df.phase == "train"]
        val = df[df.phase == "val"]
        ax_bpc.plot(train.step, train.bpc, alpha=0.35, linewidth=1.0, label=f"{label} (train)")
        ax_bpc.plot(val.step, val.bpc, marker="o", markersize=3.5, linewidth=1.5, label=f"{label} (val)")
        ax_lr.plot(train.step, train.lr, linewidth=1.0, label=label)

    if gzip_baseline is not None:
        ax_bpc.axhline(gzip_baseline, color="red", linestyle="--", linewidth=1.0,
                       label=f"GZIP baseline ({gzip_baseline} BPC)")

    ax_bpc.axhline(8.0, color="gray", linestyle=":", linewidth=0.8, label="uniform (8.0 BPC)")
    ax_bpc.set_ylabel("BPC (bits per byte)")
    ax_bpc.set_title("Training and validation BPC")
    ax_bpc.grid(True, alpha=0.3)
    ax_bpc.legend(loc="upper right", fontsize=9)

    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("step")
    ax_lr.set_yscale("log")
    ax_lr.grid(True, alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved: {output}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="*", help="CSV file(s) to plot")
    p.add_argument("--label", action="append", nargs=2, metavar=("LABEL", "CSV"),
                   help="labelled CSV (repeat for multiple)")
    p.add_argument("--output", default="results/plots/training.png", help="output PNG path")
    p.add_argument("--gzip-baseline", type=float, default=2.92,
                   help="GZIP BPC reference line (default 2.92 for enwik8); set to 0 to disable")
    args = p.parse_args()

    runs: list[tuple[str, pd.DataFrame]] = []
    for csv in args.csv:
        path = Path(csv)
        runs.append((path.stem, load(path)))
    for label, csv in args.label or []:
        runs.append((label, load(Path(csv))))

    if not runs:
        p.error("podaj przynajmniej jeden CSV (jako pozycyjny argument lub --label)")

    baseline = args.gzip_baseline if args.gzip_baseline > 0 else None
    plot_runs(runs, Path(args.output), baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
