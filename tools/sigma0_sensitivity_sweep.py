#!/usr/bin/env python3
"""
tools/sigma0_sensitivity_sweep.py

R11 — VLM bearing noise (σ₀) sensitivity sweep.

Sweeps vlm_bearing_deg ∈ {5°, 15°, 30°, 45°, 112°} on SCENARIO_BASELINE
(ATL→Macon, 128 km, daytime, seeds 1–10).  Shows that headline pass rates
are not brittle to the specific σ₀ value assumed in the UKF.

112° is included because the Stage 2 calibration measured RMSE = 112°
against Gemma 3 4B on real aerial corridor imagery (see
tools/vlm_calibration/bearing_calibration_results.json).

Results: results/sigma0_sensitivity_sweep.json
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uav_nav.config as _cfg_mod
from uav_nav.config import get_config
from uav_nav.demo.run_scenario import run_scenario
from uav_nav.demo.scenario_configs import SCENARIO_BASELINE

_DAYTIME_START_UTC = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)

SIGMA0_VALUES = [5.0, 15.0, 30.0, 45.0, 112.0]
N_SEEDS = 10
START_SEED = 1
OUTPUT_DIR = "/tmp/sigma0_sweep"
OUTPUT_JSON = "results/sigma0_sensitivity_sweep.json"


def _daytime_cfg(base):
    return dataclasses.replace(
        base,
        t_start_utc=_DAYTIME_START_UTC,
        navigation_mode="daytime",
    )


def run_sigma0(scenario_cfg, seeds, output_dir, sigma0: float):
    """Run 10 seeds with vlm_bearing_deg patched to sigma0."""
    base_settings = get_config()
    patched_noise = base_settings.measurement_noise.model_copy(
        update={"vlm_bearing_deg": sigma0}
    )
    patched = base_settings.model_copy(
        update={"measurement_noise": patched_noise}
    )
    _cfg_mod._config = patched
    rows = []
    try:
        for s in seeds:
            cfg = dataclasses.replace(scenario_cfg, seed=s)
            r = run_scenario(
                cfg,
                output_dir=output_dir,
                use_vlm=True,
                use_solar=True,
                use_celestial=False,
                use_barometer=True,
                use_magnetometer=True,
                use_airspeed=True,
            )
            rows.append({
                "seed": s,
                "final_error_m": r["final_error_m"],
                "vlm_updates": r.get("vlm_updates", 0),
                "solar_updates": r.get("solar_updates", 0),
                "nees_alarms": r["nees_alarm_count"],
                "true_nees_2d_max": r.get("true_nees_2d_max"),
                "passed": r["passed"],
            })
    finally:
        _cfg_mod._config = base_settings
    return rows


def summarize(rows, threshold):
    errs = [r["final_error_m"] for r in rows]
    passes = sum(1 for r in rows if r["passed"])
    nees_vals = [r["true_nees_2d_max"] for r in rows
                 if r.get("true_nees_2d_max") is not None]
    return {
        "n_seeds": len(rows),
        "threshold_m": threshold,
        "pass_count": passes,
        "pass_rate": passes / len(rows),
        "mean_m": statistics.mean(errs),
        "median_m": statistics.median(errs),
        "stdev_m": statistics.stdev(errs) if len(errs) > 1 else 0.0,
        "min_m": min(errs),
        "max_m": max(errs),
        "nis_alarm_seeds": sum(1 for r in rows if r["nees_alarms"] > 0),
        "true_nees_2d_max_mean": statistics.mean(nees_vals) if nees_vals else None,
    }


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    threshold = get_config().pass_threshold_m
    seeds = list(range(START_SEED, START_SEED + N_SEEDS))
    base_cfg = _daytime_cfg(SCENARIO_BASELINE)

    print(f"σ₀ sensitivity sweep — {base_cfg.label}, {N_SEEDS} seeds each")
    print(f"Threshold: {threshold} m  |  Seeds: {seeds[0]}–{seeds[-1]}")
    print(f"Default σ₀ in config: {get_config().measurement_noise.vlm_bearing_deg}°\n")
    print(f"{'σ₀ (°)':>8}  {'Pass':>6}  {'Mean':>7}  {'Med':>7}  "
          f"{'Min':>6}  {'Max':>6}  {'NIS':>4}  {'NEES_max':>9}")
    print("-" * 72)

    all_results = []
    for sigma0 in SIGMA0_VALUES:
        rows = run_sigma0(base_cfg, seeds, OUTPUT_DIR, sigma0)
        s = summarize(rows, threshold)
        default_marker = " *" if sigma0 == get_config().measurement_noise.vlm_bearing_deg else "  "
        nees_str = f"{s['true_nees_2d_max_mean']:.2f}" if s["true_nees_2d_max_mean"] else "  n/a"
        print(
            f"{sigma0:>7.1f}{default_marker}  "
            f"{s['pass_count']:>3}/{s['n_seeds']:<2}  "
            f"{s['mean_m']:>7.1f}  {s['median_m']:>7.1f}  "
            f"{s['min_m']:>6.1f}  {s['max_m']:>6.1f}  "
            f"{s['nis_alarm_seeds']:>4}  "
            f"{nees_str:>9}"
        )
        all_results.append({
            "sigma0_deg": sigma0,
            "is_default": (sigma0 == get_config().measurement_noise.vlm_bearing_deg),
            **s,
            "per_seed": rows,
        })

    out = Path(OUTPUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "scenario": base_cfg.label,
                "n_seeds": N_SEEDS,
                "sigma0_values": SIGMA0_VALUES,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults written to: {out.resolve()}")


if __name__ == "__main__":
    main()
