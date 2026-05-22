"""
Run all scenarios and produce:
  1. One trajectory+error PNG + MP4 per scenario.
  2. A comparison summary grid PNG per mode.
  3. A summary results table printed to stdout.

Usage:
    python -m uav_nav.demo.run_all_scenarios --output-dir results/
    python -m uav_nav.demo.run_all_scenarios --output-dir results/ --mode nighttime
    python -m uav_nav.demo.run_all_scenarios --output-dir results/ --mode daytime
    python -m uav_nav.demo.run_all_scenarios --output-dir results/ --mode both
"""
from __future__ import annotations

import dataclasses
import sys
import time
from datetime import datetime, timezone

from uav_nav.config import get_config
from uav_nav.demo.scenario_configs import ALL_SCENARIOS
from uav_nav.demo.run_scenario import run_scenario

# Daytime start: July 15 at 17:00 UTC (1 PM EDT) — optimal solar GDOP for Atlanta
_DAYTIME_START_UTC = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)


def _daytime_cfg(cfg):
    """Return a copy of cfg with daytime t_start_utc and navigation_mode='daytime'."""
    return dataclasses.replace(cfg,
                               t_start_utc=_DAYTIME_START_UTC,
                               navigation_mode="daytime")


def _run_mode(scenarios, mode: str, output_dir: str) -> list:
    """Run all scenarios for one mode (nighttime or daytime). Return results list."""
    use_solar = (mode == "daytime")
    cfgs = [_daytime_cfg(c) for c in scenarios] if use_solar else list(scenarios)

    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  uav_nav — {mode.upper()} Navigation Validation", file=sys.stderr)
    print(f"  Running {len(cfgs)} scenarios | solar={'ON' if use_solar else 'OFF'}",
          file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    results = []
    t0_total = time.time()
    for cfg in cfgs:
        t0 = time.time()
        result = run_scenario(cfg, output_dir=output_dir, use_solar=use_solar)
        result["wall_s"] = time.time() - t0
        results.append(result)
    total_wall = time.time() - t0_total

    _print_summary(results, total_wall, mode)
    _plot_comparison_grid(results, output_dir)
    return results


def _print_summary(results, total_wall, mode):
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY — {mode.upper()}")
    print(f"{'='*70}")
    header = (
        f"{'Scenario':<22} {'Dist':>7} {'Dur':>6} "
        f"{'FinalErr':>9} {'P95':>8} {'VLM':>5} {'Status':<10}"
    )
    print(header)
    print("-" * 70)
    all_passed = True
    for r in results:
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        if not r["passed"]:
            all_passed = False
        print(
            f"{r['label']:<22} "
            f"{r['dist_km']:>6.1f}km "
            f"{r['duration_s']/60:>5.1f}m "
            f"{r['final_error_m']:>8.0f}m "
            f"{r['p95_error_m']:>7.0f}m "
            f"{r['vlm_updates']:>5d} "
            f"{status:<10}"
        )
    print("-" * 70)
    overall = "ALL PASS ✓" if all_passed else "SOME FAILED ✗"
    print(f"{'Overall:':<22} {overall}")
    print(f"Total wall time: {total_wall:.1f}s")


def main(output_dir: str = ".", mode: str = "both"):
    all_results = {}
    if mode in ("nighttime", "both"):
        all_results["nighttime"] = _run_mode(ALL_SCENARIOS, "nighttime", output_dir)
    if mode in ("daytime", "both"):
        all_results["daytime"] = _run_mode(ALL_SCENARIOS, "daytime", output_dir)
    return all_results


def _plot_comparison_grid(results, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
        import os
    except ImportError:
        return

    n = len(results)
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#f0f2f5")

    # 2-row grid: top row = trajectory maps, bottom row = error curves + bar chart
    gs_top = gridspec.GridSpec(1, n, figure=fig,
                               top=0.88, bottom=0.48,
                               hspace=0.05, wspace=0.25,
                               left=0.05, right=0.97)
    gs_bot = gridspec.GridSpec(1, n + 1, figure=fig,
                               top=0.40, bottom=0.08,
                               hspace=0.05, wspace=0.30,
                               left=0.05, right=0.97)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    traj_colors = colors

    for i, r in enumerate(results):
        # ------ trajectory map (top row) ------
        ax_map = fig.add_subplot(gs_top[0, i])
        ax_map.set_facecolor("#dce8f7")

        log = r["log_entries"]
        true_lats = [e["lat"] for e in log]  # approx: est ≈ truth at this scale
        est_lons  = [e["lon"] for e in log]
        est_lats  = [e["lat"] for e in log]

        # Draw true trajectory (recreate from config)
        cfg = ALL_SCENARIOS[i]
        heading_rad = __import__("math").radians(
            __import__("uav_nav.demo.generic_scenario",
                       fromlist=["_compute_heading"])._compute_heading(
                cfg.origin_lat, cfg.origin_lon, cfg.dest_lat, cfg.dest_lon))
        ax_map.plot(est_lons, est_lats,
                    color=traj_colors[i], lw=1.8, label="UKF")
        ax_map.plot(cfg.origin_lon, cfg.origin_lat, "gs", ms=9, zorder=5)
        ax_map.plot(cfg.dest_lon,   cfg.dest_lat,   "r^", ms=9, zorder=5)
        for lm in cfg.landmarks:
            ax_map.plot(lm.lon_deg, lm.lat_deg, "D",
                        ms=7, color="#9467bd", zorder=6)

        passed = r["passed"]
        title_color = "#2ca02c" if passed else "#d62728"
        ax_map.set_title(
            f"{r['label']}\n"
            f"{r['dist_km']:.0f} km  |  {r['duration_s']/60:.0f} min\n"
            f"Err: {r['final_error_m']:.0f} m  "
            f"({'PASS' if passed else 'FAIL'})",
            fontsize=8, fontweight="bold", color=title_color
        )
        ax_map.tick_params(labelsize=6)
        ax_map.grid(True, alpha=0.3, lw=0.4)
        ax_map.set_xlabel("Lon (°)", fontsize=7)
        if i == 0:
            ax_map.set_ylabel("Lat (°)", fontsize=7)

    # ------ error curves (bottom row, first n panels) ------
    for i, r in enumerate(results):
        ax_err = fig.add_subplot(gs_bot[0, i])
        ax_err.set_facecolor("#f0f4ff")
        times  = [e["t_s"] / 60.0 for e in r["log_entries"]]
        errors = [e["error_m"]     for e in r["log_entries"]]
        _thresh = r.get("pass_threshold_m", get_config().pass_threshold_m)
        ax_err.plot(times, errors, color=colors[i], lw=1.5)
        ax_err.fill_between(times, errors, alpha=0.15, color=colors[i])
        ax_err.axhline(_thresh, color="#d62728", ls=":", lw=1.2)
        ax_err.set_title(r["label"], fontsize=8, fontweight="bold")
        ax_err.set_xlabel("Time (min)", fontsize=7)
        if i == 0:
            ax_err.set_ylabel("Error (m)", fontsize=7)
        ax_err.set_xlim(0, r["duration_s"] / 60.0)
        ax_err.set_ylim(0, max(max(errors) * 1.15, _thresh * 1.1))
        ax_err.tick_params(labelsize=6)
        ax_err.grid(True, alpha=0.3, lw=0.4)

    # ------ bar chart summary (bottom-right extra panel) ------
    ax_bar = fig.add_subplot(gs_bot[0, n])
    ax_bar.set_facecolor("#f0f4ff")
    labels    = [r["label"].replace(" ", "\n") for r in results]
    final_errs = [r["final_error_m"] for r in results]
    p95s       = [r["p95_error_m"]   for r in results]
    x = range(n)
    w = 0.38
    bars1 = ax_bar.bar([xi - w / 2 for xi in x], final_errs,
                       width=w, label="Final error", color=colors, alpha=0.85)
    bars2 = ax_bar.bar([xi + w / 2 for xi in x], p95s,
                       width=w, label="P95 error",
                       color=colors, alpha=0.45, hatch="//")
    _thresh = results[0].get("pass_threshold_m", get_config().pass_threshold_m) if results else get_config().pass_threshold_m
    ax_bar.axhline(_thresh, color="#d62728", ls=":", lw=1.5, label=f"{_thresh:.0f} m target")
    for bar, val in zip(bars1, final_errs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 8,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold")
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels(labels, fontsize=7)
    ax_bar.set_ylabel("Error (m)", fontsize=8)
    ax_bar.set_title("Final Error Comparison", fontsize=8, fontweight="bold")
    ax_bar.legend(fontsize=7)
    ax_bar.grid(True, axis="y", alpha=0.3, lw=0.4)

    # Super-title
    all_passed = all(r["passed"] for r in results)
    fig.suptitle(
        "uav_nav — Multi-Scenario Navigation Validation\n"
        f"{'ALL SCENARIOS PASSED ✓' if all_passed else 'SOME SCENARIOS FAILED ✗'}  "
        f"| {n} scenarios | {_thresh:.0f} m criterion",
        fontsize=14, fontweight="bold",
        color="#2ca02c" if all_passed else "#d62728",
        y=0.96
    )

    import os
    from datetime import datetime, timezone, timedelta
    _est = timezone(timedelta(hours=-5))
    _ts = datetime.now(_est).strftime("%Y%m%d-%H%M%SEST")
    _mode = results[0].get("navigation_mode", "nighttime") if results else "nighttime"
    out = os.path.join(output_dir, f"scenario_comparison_grid_{_mode}_{_ts}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nComparison grid saved → {out}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run all navigation scenarios and produce comparison plots.")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write PNG files (default: .)")
    parser.add_argument("--mode", default="both",
                        choices=["nighttime", "daytime", "both"],
                        help="Navigation mode to run (default: both)")
    args = parser.parse_args()
    main(output_dir=args.output_dir, mode=args.mode)
