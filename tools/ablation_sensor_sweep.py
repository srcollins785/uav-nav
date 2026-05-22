#!/usr/bin/env python3
"""
tools/ablation_sensor_sweep.py

Sensor contribution ablation — answers Dr. Gupta's requests #1 and #2:
  "Add ablations showing each sensor's contribution."
  "VLM's actual contribution remains unclear."

Four configurations × 10 seeds on SCENARIO_BASELINE (ATL→Macon, 128 km,
daytime, seeds 1–10). All configs keep IMU + barometer + magnetometer + airspeed.

  A: Dead reckoning  — IMU + baro + mag          (no VLM, no solar)
  B: + Solar only    — IMU + baro + mag + solar   (no VLM)
  C: + VLM only      — IMU + baro + mag + VLM     (no solar)
  D: Full system     — IMU + baro + mag + VLM + solar

B→D delta = solar marginal contribution; C→D delta = VLM marginal.

Results: results/ablation_sensor_sweep.json
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uav_nav.config import get_config
from uav_nav.demo.run_scenario import run_scenario
from uav_nav.demo.scenario_configs import SCENARIO_BASELINE

_DAYTIME_START_UTC = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)

CONFIGS = [
    {
        "label": "Dead reckoning (IMU+baro+mag)",
        "key": "dead_reckoning",
        "use_vlm": False,
        "use_solar": False,
    },
    {
        "label": "+ Solar only",
        "key": "solar_only",
        "use_vlm": False,
        "use_solar": True,
    },
    {
        "label": "+ VLM only",
        "key": "vlm_only",
        "use_vlm": True,
        "use_solar": False,
    },
    {
        "label": "Full system (solar + VLM)",
        "key": "full_system",
        "use_vlm": True,
        "use_solar": True,
    },
]

N_SEEDS = 10
START_SEED = 1
OUTPUT_DIR = "/tmp/ablation_sweep"
OUTPUT_JSON = "results/ablation_sensor_sweep.json"


def _daytime_cfg(base):
    return dataclasses.replace(
        base,
        t_start_utc=_DAYTIME_START_UTC,
        navigation_mode="daytime",
    )


def run_config(scenario_cfg, seeds, output_dir, use_vlm, use_solar):
    rows = []
    for s in seeds:
        cfg = dataclasses.replace(scenario_cfg, seed=s)
        r = run_scenario(
            cfg,
            output_dir=output_dir,
            use_vlm=use_vlm,
            use_solar=use_solar,
            use_celestial=False,   # daytime mode: use solar not stars
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
            "passed": r["passed"],
        })
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
    base_cfg = _daytime_cfg(SCENARIO_BASELINE)

    print(f"Sensor ablation sweep — {base_cfg.label}, {N_SEEDS} seeds each")
    print(f"Threshold: {threshold} m  |  Seeds: {seeds[0]}–{seeds[-1]}\n")
    print(f"{'Config':<36}  {'Pass':>6}  {'Mean':>7}  {'Med':>7}  "
          f"{'Min':>6}  {'Max':>6}  {'NEES':>5}")
    print("-" * 80)

    all_results = []
    for cfg_spec in CONFIGS:
        rows = run_config(
            base_cfg, seeds, OUTPUT_DIR,
            use_vlm=cfg_spec["use_vlm"],
            use_solar=cfg_spec["use_solar"],
        )
        s = summarize(rows, threshold)
        print(
            f"{cfg_spec['label']:<36}  "
            f"{s['pass_count']:>3}/{s['n_seeds']:<2}  "
            f"{s['mean_m']:>7.1f}  {s['median_m']:>7.1f}  "
            f"{s['min_m']:>6.1f}  {s['max_m']:>6.1f}  "
            f"{s['nees_alarm_seeds']:>5}"
        )
        all_results.append({**cfg_spec, **s, "per_seed": rows})

    out = Path(OUTPUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "scenario": base_cfg.label,
            "n_seeds": N_SEEDS,
            "threshold_m": threshold,
            "configs": all_results,
        }, f, indent=2)
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
