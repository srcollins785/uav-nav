#!/usr/bin/env python3
"""
tools/p2_4_threshold_sensitivity.py

P2.4 — VLM gating threshold sensitivity sweep.

Shows that the three data-association parameters are not cherry-picked:
pass rate remains acceptable across a range of values on either side of the
chosen defaults.

  vlm_assoc_min_margin_deg          :  5  |  8 (default)  | 12
  vlm_assoc_max_bearing_mismatch_deg: 10  | 15 (default)  | 25
  landmark_min_confidence           : 0.6 | 0.7 (default) | 0.8

Scenario: SCENARIO_MUCH_LONGER (ATL→JAX, 435 km, daytime), 10 seeds per cell.
All other parameters held at their default values from config.yaml.

Results: results/p2_4_threshold_sensitivity.json
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
from uav_nav.demo.scenario_configs import SCENARIO_BASELINE, SCENARIO_MUCH_LONGER

_DAYTIME_START_UTC = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)

SWEEPS = [
    {
        "param": "vlm_assoc_min_margin_deg",
        "label": "Ambiguity margin (deg)",
        "values": [5.0, 8.0, 12.0],
        "default": 8.0,
    },
    {
        "param": "vlm_assoc_max_bearing_mismatch_deg",
        "label": "Absolute gate (deg)",
        "values": [10.0, 15.0, 25.0],
        "default": 15.0,
    },
    {
        "param": "landmark_min_confidence",
        "label": "VLM confidence threshold",
        "values": [0.6, 0.7, 0.8],
        "default": 0.7,
    },
]

N_SEEDS = 10
START_SEED = 1
OUTPUT_DIR = "/tmp/p2_4_sweep"
OUTPUT_JSON = "results/p2_4_threshold_sensitivity_much_longer.json"


def _daytime_cfg(base):
    return dataclasses.replace(
        base,
        t_start_utc=_DAYTIME_START_UTC,
        navigation_mode="daytime",
    )


def run_cell(scenario_cfg, seeds, output_dir, **overrides):
    """Run one (parameter, value) cell, temporarily patching the config singleton."""
    base_settings = get_config()
    patched = base_settings.model_copy(update=overrides)
    _cfg_mod._config = patched
    rows = []
    try:
        for s in seeds:
            cfg = dataclasses.replace(scenario_cfg, seed=s)
            r = run_scenario(cfg, output_dir=output_dir, use_solar=True)
            rows.append({
                "seed": s,
                "final_error_m": r["final_error_m"],
                "vlm_updates": r["vlm_updates"],
                "nees_alarms": r["nees_alarm_count"],
                "passed": r["passed"],
            })
    finally:
        _cfg_mod._config = base_settings
    return rows


def summarize(rows, threshold):
    errs = [r["final_error_m"] for r in rows]
    passes = sum(1 for r in rows if r["passed"])
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
        "nees_alarm_seeds": sum(1 for r in rows if r["nees_alarms"] > 0),
    }


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    threshold = get_config().pass_threshold_m
    seeds = list(range(START_SEED, START_SEED + N_SEEDS))
    base_cfg = _daytime_cfg(SCENARIO_MUCH_LONGER)

    print(f"P2.4 threshold sensitivity sweep — {base_cfg.label}, {N_SEEDS} seeds each")
    print(f"Threshold: {threshold} m  |  Start seed: {START_SEED}\n")

    all_results = []

    for sweep in SWEEPS:
        param = sweep["param"]
        label = sweep["label"]
        default = sweep["default"]
        print(f"\n{'='*60}")
        print(f"Sweeping: {label}  (param: {param})")
        print(f"{'Value':>8}  {'Pass':>6}  {'Mean':>7}  {'Med':>7}  "
              f"{'Min':>6}  {'Max':>6}  {'NEES':>5}")
        print("-" * 60)

        sweep_rows = []
        for val in sweep["values"]:
            rows = run_cell(base_cfg, seeds, OUTPUT_DIR, **{param: val})
            s = summarize(rows, threshold)
            is_default = (val == default)
            marker = " *" if is_default else "  "
            print(
                f"{val:>8.1f}{marker}  "
                f"{s['pass_count']:>3}/{s['n_seeds']:<2}  "
                f"{s['mean_m']:>7.1f}  {s['median_m']:>7.1f}  "
                f"{s['min_m']:>6.1f}  {s['max_m']:>6.1f}  "
                f"{s['nees_alarm_seeds']:>5}"
            )
            sweep_rows.append({
                "param": param,
                "label": label,
                "value": val,
                "is_default": is_default,
                **s,
                "per_seed": rows,
            })
        all_results.append({"sweep": sweep, "cells": sweep_rows})

    out = Path(OUTPUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"scenario": base_cfg.label, "n_seeds": N_SEEDS,
                   "threshold_m": threshold, "sweeps": all_results}, f, indent=2)
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
