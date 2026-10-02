#!/usr/bin/env python3
"""Fit Chinchilla scaling law to byte-level Transformer experiments on enwik8.

Form (Hoffmann et al. 2022, arXiv:2203.15556):
    L(N, D) = E + A / N^alpha + B / D^beta
where
    L     loss in nats per token (= BPC * ln 2)
    N     number of parameters
    D     number of training tokens (steps * batch * seq_len)
    E     irreducible loss (lower bound, ~corpus entropy)
    A, B  scaling coefficients (positive)
    alpha, beta  scaling exponents on params, tokens

Inputs:
    results/grid.csv         (27 baseline points from `bin/grid`)
    plus 2 long-run baselines hardcoded
    plus 1 llama point used for validation only

Outputs:
    fitted parameters with std errors and R^2,
    extrapolations to xxlarge and beyond,
    plot results/plots/scaling_law_fit.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

LN2 = float(np.log(2))


def chinchilla(X, E, A, B, alpha, beta):
    N, D = X
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def fmt_n(n):
    if n >= 1e9:
        return f"{n/1e9:.2f} G"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{n/1e3:.2f} k"
    return f"{n:.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="results/grid.csv")
    ap.add_argument("--output-dir", default="results/plots")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Grid points
    grid = pd.read_csv(args.grid)
    grid["D"] = 2000 * 16 * grid["seq_len"]
    grid["N"] = grid["n_params"]
    grid["bpc"] = grid["best_val_bpc"]
    grid["arch"] = "baseline"
    grid["label"] = grid.apply(
        lambda r: f"d{r['d_model']}_L{r['n_layers']}_s{r['seq_len']}", axis=1
    )

    # 2) Long-run baselines (best_val_bpc from training loops, server-ai 4080S)
    long_baseline = pd.DataFrame(
        [
            {"label": "large_40k_b16", "N": 19_673_088, "D": 40000 * 16 * 1024, "bpc": 3.97, "arch": "baseline"},
            {"label": "xlarge_20k_b8", "N": 57_827_328, "D": 20000 * 8 * 1024, "bpc": 3.93, "arch": "baseline"},
        ]
    )

    # 3) Llama point (separate arch, used only for validation)
    llama = pd.DataFrame(
        [
            {"label": "xlarge_llama_20k_b8", "N": 57_029_376, "D": 20000 * 8 * 1024, "bpc": 3.85, "arch": "llama"},
        ]
    )

    baseline = pd.concat(
        [grid[["label", "arch", "N", "D", "bpc"]], long_baseline], ignore_index=True
    )
    baseline["L_nat"] = baseline["bpc"] * LN2
    llama["L_nat"] = llama["bpc"] * LN2

    print(f"baseline points: {len(baseline)}    llama val points: {len(llama)}")
    print(baseline[["label", "N", "D", "bpc"]].to_string(index=False))

    # ----- Fit -----
    Ns = baseline["N"].to_numpy()
    Ds = baseline["D"].to_numpy()
    Ls = baseline["L_nat"].to_numpy()

    p0 = [1.0, 100.0, 100.0, 0.3, 0.3]
    bounds = ([0.0, 0.0, 0.0, 0.01, 0.01], [10.0, 1e8, 1e8, 1.0, 1.0])
    popt, pcov = curve_fit(chinchilla, (Ns, Ds), Ls, p0=p0, bounds=bounds, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    E, A, B, a, b = popt
    Ls_pred = chinchilla((Ns, Ds), *popt)
    r2 = 1.0 - np.sum((Ls - Ls_pred) ** 2) / np.sum((Ls - Ls.mean()) ** 2)

    print("\n=== Chinchilla fit (baseline arch) ===")
    print(f"  E      = {E:.4f} ± {perr[0]:.4f} nat   ({E / LN2:.4f} BPC)")
    print(f"  A      = {A:.4f} ± {perr[1]:.4f}")
    print(f"  B      = {B:.4f} ± {perr[2]:.4f}")
    print(f"  alpha  = {a:.4f} ± {perr[3]:.4f}   (params exponent)")
    print(f"  beta   = {b:.4f} ± {perr[4]:.4f}   (tokens exponent)")
    print(f"  R^2    = {r2:.4f}")

    # ----- Llama predictions -----
    print("\n=== Llama val predictions using baseline law ===")
    for _, r in llama.iterrows():
        L_pred = chinchilla((r["N"], r["D"]), *popt)
        bpc_pred = L_pred / LN2
        print(
            f"  {r['label']}: predicted={bpc_pred:.4f} BPC, actual={r['bpc']:.4f} BPC, "
            f"Δ={r['bpc'] - bpc_pred:+.4f}"
        )

    # ----- Extrapolations -----
    print("\n=== Extrapolations (baseline law) ===")
    extrapolations = [
        ("xxlarge @20k b=8", 102_000_000, 20000 * 8 * 1024),
        ("xxlarge @40k b=8", 102_000_000, 40000 * 8 * 1024),
        ("xxlarge @80k b=8", 102_000_000, 80000 * 8 * 1024),
        ("200M params @ 1.6B tokens", 200_000_000, 1_600_000_000),
        ("500M params @ 10B tokens", 500_000_000, 10_000_000_000),
    ]
    for label, N, D in extrapolations:
        L_pred = chinchilla((N, D), *popt)
        bpc_pred = L_pred / LN2
        print(f"  {label}: N={fmt_n(N)} D={fmt_n(D)} -> {bpc_pred:.4f} BPC")

    # ----- Solve for GZIP target with Chinchilla-optimal D/N=20 -----
    print("\n=== Reaching GZIP (2.92 BPC) at Chinchilla-optimal D/N=20 ===")
    L_target = 2.92 * LN2

    def gap(N_):
        return chinchilla((N_, 20.0 * N_), *popt) - L_target

    try:
        if gap(1e6) > 0 and gap(1e12) < 0:
            N_star = brentq(gap, 1e6, 1e12, xtol=1e3)
            D_star = 20.0 * N_star
            steps_star = D_star / (8 * 1024)
            wall_h = steps_star / 9.0 / 3600.0
            print(f"  Required N = {fmt_n(N_star)} params")
            print(f"  Required D = {fmt_n(D_star)} tokens")
            print(f"  Equivalent: {fmt_n(steps_star)} steps at b=8, seq=1024")
            print(f"  Wall estimate (4080S, ~9 step/s, ignoring N>>57M slowdown): {wall_h:.1f} h")
        else:
            print("  Target unreachable in fitted bounds (E too high or law extrapolates oddly).")
    except ValueError as e:
        print(f"  brentq failed: {e}")

    # ----- Plot -----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sc = ax.scatter(Ls, Ls_pred, c=baseline["D"], cmap="viridis", s=40, alpha=0.7, label="baseline (grid+long)")
    minL, maxL = float(min(Ls.min(), Ls_pred.min())), float(max(Ls.max(), Ls_pred.max()))
    ax.plot([minL, maxL], [minL, maxL], "k--", alpha=0.3, label="y=x")
    for _, r in llama.iterrows():
        L_pred = chinchilla((r["N"], r["D"]), *popt)
        ax.scatter([r["L_nat"]], [L_pred], c="red", s=140, marker="*", label=f"{r['label']}", zorder=10)
    plt.colorbar(sc, ax=ax, label="tokens D")
    ax.set_xlabel("actual loss (nat)")
    ax.set_ylabel("predicted loss (nat)")
    ax.set_title(f"Chinchilla fit  (R²={r2:.3f}, n={len(baseline)})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    Ns_range = np.logspace(5.5, 9.0, 200)
    for D_fixed, color, label in [
        (2000 * 16 * 256, "tab:blue", "32 M tok (grid s=256)"),
        (2000 * 16 * 1024, "tab:green", "131 M tok (grid s=1024)"),
        (164_000_000, "tab:orange", "164 M tok (xlarge run)"),
        (656_000_000, "tab:red", "656 M tok (large 40k)"),
        (1_640_000_000, "tab:purple", "1.64 G tok (multi-day)"),
    ]:
        Ls_pred = chinchilla((Ns_range, D_fixed), *popt)
        ax.plot(Ns_range, Ls_pred / LN2, color=color, label=label, alpha=0.8)
    ax.scatter(
        baseline["N"], baseline["bpc"], c=baseline["D"], cmap="viridis", s=30, alpha=0.6, edgecolors="k", linewidths=0.3
    )
    ax.scatter(
        llama["N"], llama["bpc"], c="red", s=140, marker="*", label="llama (separate arch)", zorder=10
    )
    ax.axhline(2.92, color="red", linestyle="--", alpha=0.5, label="GZIP (2.92 BPC)")
    ax.axhline(E / LN2, color="black", linestyle=":", alpha=0.4, label=f"irreducible E ({E/LN2:.2f} BPC)")
    ax.set_xscale("log")
    ax.set_xlabel("params N")
    ax.set_ylabel("BPC")
    ax.set_title("Predicted BPC vs N at fixed D (baseline law)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = out_dir / "scaling_law_fit.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
