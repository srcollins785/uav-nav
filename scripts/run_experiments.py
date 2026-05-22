#!/usr/bin/env python3
"""
scripts/run_experiments.py

Unified reproducible experiment runner.

Runs all selected scenarios × modes × seeds through the full navigation
pipeline and writes a CSV of per-run results.

Usage:
    python scripts/run_experiments.py \\
        --config configs/default_experiment.yaml \\
        --scenarios all \\
        --modes daytime,nighttime \\
        --seeds 10 \\
        --output results/latest

    # Quick smoke-test (1 scenario, 1 seed):
    python scripts/run_experiments.py --quick

All flags override values in the YAML config.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import statistics
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

# Allow importing uav_nav from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from uav_nav.demo.run_scenario import run_scenario
from uav_nav.demo.scenario_configs import ALL_SCENARIOS, ScenarioConfig

_DAYTIME_START_UTC = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)

# CSV columns produced by this runner (subset of the full ablation schema)
_CSV_COLUMNS = [
    "run_id", "timestamp", "scenario_id", "scenario_label", "mode", "seed",
    "imu_enabled", "barometer_enabled", "magnetometer_enabled",
    "airspeed_enabled", "vlm_enabled", "celestial_enabled", "solar_enabled",
    "final_error_m", "max_error_m", "mean_error_m", "median_error_m", "p90_error_m",
    "pass_10m", "pass_25m", "pass_50m", "pass_100m",
    "vlm_updates", "vlm_obs_total", "vlm_passed_confidence",
    "vlm_rejected_confidence", "vlm_rejected_bearing", "vlm_rejected_ambiguity",
    "vlm_ukf_updates",
    "celestial_updates", "solar_updates",
    "nees_alarm_count",
    "n_landmarks", "dist_km", "duration_s",
    "notes",
]


def _load_config(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_scenarios(spec) -> List[ScenarioConfig]:
    if spec == "all" or spec is None:
        return list(ALL_SCENARIOS)
    if isinstance(spec, str):
        spec = [spec]
    by_id = {s.scenario_id: s for s in ALL_SCENARIOS}
    missing = set(spec) - set(by_id)
    if missing:
        print(f"ERROR: unknown scenario_id(s): {sorted(missing)}", file=sys.stderr)
        print(f"  known: {list(by_id)}", file=sys.stderr)
        sys.exit(2)
    return [by_id[s] for s in spec]


def _daytime_cfg(cfg: ScenarioConfig) -> ScenarioConfig:
    return dataclasses.replace(cfg, t_start_utc=_DAYTIME_START_UTC, navigation_mode="daytime")


def _result_to_csv_row(result: dict, mode: str, seed: int, run_id: str) -> dict:
    ts = datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "run_id": run_id,
        "timestamp": ts,
        "scenario_id": result["scenario_id"],
        "scenario_label": result["label"],
        "mode": mode,
        "seed": seed,
        "imu_enabled": result.get("imu_enabled", True),
        "barometer_enabled": result.get("barometer_enabled", True),
        "magnetometer_enabled": result.get("magnetometer_enabled", True),
        "airspeed_enabled": result.get("airspeed_enabled", True),
        "vlm_enabled": result.get("vlm_enabled", True),
        "celestial_enabled": result.get("celestial_enabled", False),
        "solar_enabled": result.get("solar_enabled", False),
        "final_error_m": round(result["final_error_m"], 2),
        "max_error_m": round(result.get("max_error_m", float("nan")), 2),
        "mean_error_m": round(result["mean_error_m"], 2),
        "median_error_m": round(result.get("median_error_m", float("nan")), 2),
        "p90_error_m": round(result.get("p90_error_m", float("nan")), 2),
        "pass_10m": result.get("pass_10m", False),
        "pass_25m": result.get("pass_25m", False),
        "pass_50m": result.get("pass_50m", result.get("passed", False)),
        "pass_100m": result.get("pass_100m", False),
        "vlm_updates": result.get("vlm_updates", 0),
        "vlm_obs_total": result.get("vlm_obs_total", 0),
        "vlm_passed_confidence": result.get("vlm_passed_confidence", 0),
        "vlm_rejected_confidence": result.get("vlm_rejected_confidence", 0),
        "vlm_rejected_bearing": result.get("vlm_rejected_bearing", 0),
        "vlm_rejected_ambiguity": result.get("vlm_rejected_ambiguity", 0),
        "vlm_ukf_updates": result.get("vlm_ukf_updates", 0),
        "celestial_updates": result.get("celestial_updates", 0),
        "solar_updates": result.get("solar_updates", 0),
        "nees_alarm_count": result.get("nees_alarm_count", 0),
        "n_landmarks": result.get("n_landmarks", 0),
        "dist_km": round(result.get("dist_km", 0), 2),
        "duration_s": round(result.get("duration_s", 0), 1),
        "notes": "",
    }


def run_all(
    scenarios: List[ScenarioConfig],
    modes: List[str],
    seeds: List[int],
    output_dir: str,
    csv_path: str,
) -> List[dict]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        for mode in modes:
            use_solar = (mode == "daytime")
            for base_cfg in scenarios:
                cfg = _daytime_cfg(base_cfg) if use_solar else base_cfg
                for seed in seeds:
                    cfg_s = dataclasses.replace(cfg, seed=seed)
                    run_id = str(uuid.uuid4())[:8]
                    try:
                        result = run_scenario(
                            cfg_s,
                            output_dir=output_dir,
                            use_solar=use_solar,
                            use_vlm=True,
                            use_celestial=not use_solar,
                        )
                        row = _result_to_csv_row(result, mode, seed, run_id)
                    except Exception as exc:
                        print(f"  ERROR: {cfg_s.scenario_id} mode={mode} seed={seed}: {exc}",
                              file=sys.stderr)
                        row = {col: "" for col in _CSV_COLUMNS}
                        row.update({"run_id": run_id, "scenario_id": cfg_s.scenario_id,
                                    "mode": mode, "seed": seed, "notes": str(exc)})
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)

    return rows


def _print_summary(rows: List[dict]) -> None:
    print(f"\n{'='*72}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*72}")
    header = f"{'Scenario':<22} {'Mode':<10} {'Seeds':>5} {'Pass50':>7} {'Mean':>8} {'Median':>8}"
    print(header)
    print("-" * 72)

    from itertools import groupby
    key = lambda r: (r.get("scenario_id", ""), r.get("mode", ""))
    for (sid, mode), group in groupby(sorted(rows, key=key), key=key):
        grp = list(group)
        valid = [r for r in grp if r.get("final_error_m", "") != ""]
        if not valid:
            continue
        errs = [float(r["final_error_m"]) for r in valid]
        passes = sum(1 for r in valid if str(r.get("pass_50m", "False")).lower() == "true")
        label = valid[0].get("scenario_label", sid)
        print(
            f"{label:<22} {mode:<10} {len(valid):>5} "
            f"{passes}/{len(valid):>5} "
            f"{statistics.mean(errs):>7.1f}m "
            f"{statistics.median(errs):>7.1f}m"
        )

    total = len(rows)
    passed = sum(1 for r in rows if str(r.get("pass_50m", "False")).lower() == "true")
    print(f"\nTotal runs: {total}   Passed (50m): {passed}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Unified UAV navigation experiment runner")
    parser.add_argument("--config", default="configs/default_experiment.yaml",
                        help="Experiment YAML config (default: configs/default_experiment.yaml)")
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario IDs or 'all'")
    parser.add_argument("--modes", default=None,
                        help="Comma-separated modes: daytime,nighttime")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds")
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--output", default=None, dest="output_dir",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 scenario, 1 seed, nighttime only")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # CLI overrides YAML
    scenario_spec = args.scenarios or cfg.get("scenarios", "all")
    if isinstance(scenario_spec, str) and "," in scenario_spec:
        scenario_spec = scenario_spec.split(",")

    mode_spec = args.modes or cfg.get("modes", ["nighttime", "daytime"])
    if isinstance(mode_spec, str):
        mode_spec = mode_spec.split(",")

    seed_cfg = cfg.get("seeds", {})
    n_seeds = args.seeds or seed_cfg.get("n", 10)
    start_seed = args.start_seed or seed_cfg.get("start", 1)
    seeds = list(range(start_seed, start_seed + n_seeds))

    output_dir = args.output_dir or cfg.get("output_dir", "results/latest")

    if args.quick:
        scenario_spec = ["very_short"]
        mode_spec = ["nighttime"]
        seeds = [1]
        print("Quick mode: 1 scenario, 1 seed, nighttime only")

    scenarios = _resolve_scenarios(scenario_spec)
    csv_path = str(Path(output_dir) / "run_results.csv")

    print(f"Running {len(scenarios)} scenario(s) × {len(mode_spec)} mode(s) × {len(seeds)} seed(s)")
    print(f"  Scenarios : {[s.scenario_id for s in scenarios]}")
    print(f"  Modes     : {mode_spec}")
    print(f"  Seeds     : {seeds}")
    print(f"  Output    : {output_dir}")
    print(f"  CSV       : {csv_path}")
    print()

    rows = run_all(scenarios, mode_spec, seeds, output_dir, csv_path)
    _print_summary(rows)
    print(f"\nCSV written → {csv_path}")


if __name__ == "__main__":
    main()
