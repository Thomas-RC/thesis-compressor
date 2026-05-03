#!/usr/bin/env python3
"""Plot heatmaps and scaling scatter from grid search results.

Reads `results/grid.csv` produced by the `grid` binary with columns:
  d_model, n_layers, seq_len, n_heads, n_params, final_val_bpc, best_val_bpc, wall_s

Generates two figures:
  1. Heatmap of best_val_bpc over (d_model x n_layers), faceted by seq_len.
  2. Scatter BPC vs n_params (log-x), markers colored by seq_len.

Usage:
  python3 scripts/plot_grid.py
  python3 scripts/plot_grid.py --input results/grid.csv --output-dir results/plots
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {
        "d_model", "n_layers", "seq_len", "n_heads",
        "n_params", "final_val_bpc", "best_val_bpc", "wall_s",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: brakujące kolumny: {missing}")
    for col in ["n_params", "final_val_bpc", "best_val_bpc", "wall_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def heatmap_per_seq_len(df: pd.DataFrame, output: Path, gzip_baseline: float | None) -> None:
    seq_lens = sorted(df.seq_len.unique())
    d_models = sorted(df.d_model.unique())
    n_layers_vals = sorted(df.n_layers.unique())

    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5 * len(seq_lens), 4.5), squeeze=False)
    vmin = df.best_val_bpc.min()
    vmax = df.best_val_bpc.max()

    for ax, sl in zip(axes[0], seq_lens):
        sub = df[df.seq_len == sl]
        grid = np.full((len(d_models), len(n_layers_vals)), np.nan)
        for _, row in sub.iterrows():
            i = d_models.index(int(row.d_model))
            j = n_layers_vals.index(int(row.n_layers))
            grid[i, j] = row.best_val_bpc
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(n_layers_vals)))
        ax.set_xticklabels(n_layers_vals)
        ax.set_yticks(range(len(d_models)))
        ax.set_yticklabels(d_models)
        ax.set_xlabel("n_layers")
        ax.set_ylabel("d_model")
        ax.set_title(f"seq_len = {sl}")
        for i in range(len(d_models)):
            for j in range(len(n_layers_vals)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > (vmin + vmax) / 2 else "black",
                            fontsize=9)

    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85)
    cbar.set_label("best val BPC (lower = better)")
    if gzip_baseline is not None:
        cbar.ax.axhline(gzip_baseline, color="red", linewidth=1.5)
    fig.suptitle("Grid search: best validation BPC per configuration", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved: {output}")


def scatter_params(df: pd.DataFrame, output: Path, gzip_baseline: float | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    seq_lens = sorted(df.seq_len.unique())
    cmap = plt.get_cmap("tab10")

    for k, sl in enumerate(seq_lens):
        sub = df[df.seq_len == sl].dropna(subset=["best_val_bpc"])
        ax.scatter(sub.n_params, sub.best_val_bpc, s=60,
                   color=cmap(k), label=f"seq_len={sl}", edgecolor="black", linewidth=0.5)
        for _, row in sub.iterrows():
            ax.annotate(f"L={int(row.n_layers)}\nd={int(row.d_model)}",
                        (row.n_params, row.best_val_bpc),
                        fontsize=7, alpha=0.6, xytext=(4, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("liczba parametrów")
    ax.set_ylabel("best val BPC")
    ax.set_title("BPC vs rozmiar modelu (skalowanie)")
    ax.grid(True, alpha=0.3)

    if gzip_baseline is not None:
        ax.axhline(gzip_baseline, color="red", linestyle="--", linewidth=1.0,
                   label=f"GZIP baseline ({gzip_baseline} BPC)")
    ax.axhline(8.0, color="gray", linestyle=":", linewidth=0.8, label="uniform (8.0 BPC)")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved: {output}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="results/grid.csv", help="grid CSV path")
    p.add_argument("--output-dir", default="results/plots", help="directory for output PNGs")
    p.add_argument("--gzip-baseline", type=float, default=2.92,
                   help="GZIP BPC reference line; set to 0 to disable")
    args = p.parse_args()

    df = load(Path(args.input))
    if df.empty:
        print("brak danych w CSV", file=sys.stderr)
        return 1

    baseline = args.gzip_baseline if args.gzip_baseline > 0 else None
    out_dir = Path(args.output_dir)
    heatmap_per_seq_len(df, out_dir / "grid_heatmap.png", baseline)
    scatter_params(df, out_dir / "grid_scaling.png", baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
