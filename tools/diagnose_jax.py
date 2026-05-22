#!/usr/bin/env python3
"""
tools/diagnose_jax.py

Per-seed error-over-time diagnostic for the Much Longer (ATL→JAX) scenario.

Runs seeds 1–10 in nighttime mode, captures the full error time series for each
seed, and produces:
  results/jax_diagnostic/per_seed_errors.csv   — (t_s, seed, error_m) tidy CSV
  results/jax_diagnostic/divergence_points.csv — seed, final_error_m, divergence_t_s
  results/jax_diagnostic/jax_seed_curves.png   — overlaid error-vs-time plot

Divergence is defined as the first time the 60-second trailing mean error
crosses 50 m and does not recover below 50 m before route end.

Usage:
    conda run -n base python tools/diagnose_jax.py
"""
from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_nav.demo.run_scenario import run_scenario
from uav_nav.demo.scenario_configs import SCENARIO_MUCH_LONGER

OUTPUT_DIR = Path("results/jax_diagnostic")
N_SEEDS = 10
PASS_M = 50.0
# Trailing-mean window for divergence detection (seconds)
_DIV_WINDOW_S = 60.0


def _divergence_time(error_samples: list[float], duration_s: float) -> float | None:
    """Return first t_s where 60-s trailing mean > 50 m and never recovers."""
    n = len(error_samples)
    if n == 0:
        return None
    dt = duration_s / n
    window = max(1, int(_DIV_WINDOW_S / dt))

    for i in range(window, n):
        mean_w = sum(error_samples[i - window:i]) / window
        if mean_w > PASS_M:
            # Check it stays above 50 m for the rest of the run
            remaining = error_samples[i:]
            if all(e > PASS_M * 0.8 for e in remaining):
                return i * dt
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  ATL→JAX Per-Seed Divergence Diagnostic")
    print(f"  Seeds 1–{N_SEEDS}, nighttime mode")
    print("=" * 60)

    seed_results: list[dict] = []
    tidy_rows: list[dict] = []

    for seed in range(1, N_SEEDS + 1):
        cfg = dataclasses.replace(SCENARIO_MUCH_LONGER, seed=seed)
        print(f"\nSeed {seed} …", end=" ", flush=True)
        result = run_scenario(
            cfg,
            output_dir=str(OUTPUT_DIR),
            use_solar=False,
            use_vlm=True,
            use_celestial=True,
        )
        fe = result["final_error_m"]
        passed = fe < PASS_M
        samples = result["error_samples"]
        dur = result["duration_s"]
        div_t = _divergence_time(samples, dur)

        status = "PASS" if passed else f"FAIL  diverges≈{div_t:.0f}s" if div_t else "FAIL"
        print(f"{fe:.0f} m  {status}")

        seed_results.append({
            "seed": seed,
            "final_error_m": round(fe, 1),
            "passed": passed,
            "divergence_t_s": round(div_t, 0) if div_t else "",
            "vlm_ukf_updates": result.get("vlm_ukf_updates", 0),
            "vlm_obs_total": result.get("vlm_obs_total", 0),
            "vlm_rejected_bearing": result.get("vlm_rejected_bearing", 0),
            "vlm_rejected_ambiguity": result.get("vlm_rejected_ambiguity", 0),
            "celestial_updates": result.get("celestial_updates", 0),
            "nees_alarm_count": result.get("nees_alarm_count", 0),
        })

        dt = dur / max(len(samples), 1)
        for i, err in enumerate(samples):
            tidy_rows.append({"t_s": round(i * dt, 1), "seed": seed, "error_m": round(err, 1)})

    # --- write CSVs ---
    div_path = OUTPUT_DIR / "divergence_points.csv"
    with open(div_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_results[0].keys()))
        w.writeheader()
        w.writerows(seed_results)

    tidy_path = OUTPUT_DIR / "per_seed_errors.csv"
    with open(tidy_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_s", "seed", "error_m"])
        w.writeheader()
        w.writerows(tidy_rows)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(12, 5))
    palette_pass = "#3fb950"
    palette_fail = "#f85149"

    for sr in seed_results:
        s = sr["seed"]
        rows = [r for r in tidy_rows if r["seed"] == s]
        ts = [r["t_s"] / 60.0 for r in rows]
        errs = [r["error_m"] for r in rows]
        color = palette_pass if sr["passed"] else palette_fail
        lw = 1.8 if not sr["passed"] else 0.9
        alpha = 0.9 if not sr["passed"] else 0.5
        label = f"seed {s}" if not sr["passed"] else None
        ax.plot(ts, errs, color=color, lw=lw, alpha=alpha, label=label)
        if not sr["passed"] and sr["divergence_t_s"] != "":
            dt_min = float(sr["divergence_t_s"]) / 60.0
            ax.axvline(dt_min, color=palette_fail, lw=0.6, ls="--", alpha=0.5)

    ax.axhline(PASS_M, color="#e3b341", ls=":", lw=1.5, label="50 m criterion")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("ATL→JAX Much Longer — per-seed error (red = FAIL, green = PASS)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(200, max(r["final_error_m"] for r in seed_results) * 1.2))
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "jax_seed_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"  Seed | Final err | Pass | Diverges at")
    print(f"  -----|-----------|------|------------")
    for sr in seed_results:
        div = f"{sr['divergence_t_s']} s" if sr['divergence_t_s'] != "" else "—"
        flag = "✓" if sr["passed"] else "✗"
        print(f"  {sr['seed']:4d} | {sr['final_error_m']:7.1f} m | {flag}    | {div}")

    n_pass = sum(1 for r in seed_results if r["passed"])
    print(f"\n  Pass rate : {n_pass}/{N_SEEDS}")
    print(f"\n  divergence_points.csv → {div_path}")
    print(f"  per_seed_errors.csv   → {tidy_path}")
    print(f"  jax_seed_curves.png   → {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
