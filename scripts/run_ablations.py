#!/usr/bin/env python3
"""
scripts/run_ablations.py

Sensor ablation experiment runner.

Tests each sensing configuration (A0–A6) across all selected scenarios,
modes, and seeds, writing one CSV row per run.

Ablation conditions
-------------------
A0  IMU only
A1  IMU + barometer
A2  IMU + barometer + magnetometer
A3  IMU + barometer + magnetometer + airspeed
A4  IMU + barometer + magnetometer + airspeed + VLM landmark bearing
A5  IMU + barometer + magnetometer + airspeed + celestial/solar (no VLM)
A6  Full system (all sensors)

For A5 and A6, "celestial/solar" means celestial in nighttime mode and
solar in daytime mode — controlled automatically by the mode argument.

NOTE: The main route sweeps use geometry-derived VLM-like bearing
observations to isolate estimator and data-association behaviour.
They do NOT measure full image-to-VLM perception performance.

Usage:
    python scripts/run_ablations.py \\
        --config configs/default_experiment.yaml \\
        --scenarios all \\
        --modes daytime,nighttime \\
        --seeds 10 \\
        --output results/ablations

    # Quick smoke-test:
    python scripts/run_ablations.py --quick
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import statistics
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from uav_nav.demo.run_scenario import run_scenario
from uav_nav.demo.scenario_configs import ALL_SCENARIOS, ScenarioConfig

_DAYTIME_START_UTC = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Ablation condition definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationCondition:
    name: str
    label: str
    use_barometer: bool
    use_magnetometer: bool
    use_airspeed: bool
    use_vlm: bool
    use_celestial_solar: bool   # True → celestial (night) or solar (day)


ABLATION_CONDITIONS: Dict[str, AblationCondition] = {
    "A0": AblationCondition(
        name="A0", label="IMU only",
        use_barometer=False, use_magnetometer=False, use_airspeed=False,
        use_vlm=False, use_celestial_solar=False,
    ),
    "A1": AblationCondition(
        name="A1", label="IMU + barometer",
        use_barometer=True, use_magnetometer=False, use_airspeed=False,
        use_vlm=False, use_celestial_solar=False,
    ),
    "A2": AblationCondition(
        name="A2", label="IMU + baro + mag",
        use_barometer=True, use_magnetometer=True, use_airspeed=False,
        use_vlm=False, use_celestial_solar=False,
    ),
    "A3": AblationCondition(
        name="A3", label="IMU + baro + mag + airspeed",
        use_barometer=True, use_magnetometer=True, use_airspeed=True,
        use_vlm=False, use_celestial_solar=False,
    ),
    "A4": AblationCondition(
        name="A4", label="IMU + baro + mag + airspeed + VLM",
        use_barometer=True, use_magnetometer=True, use_airspeed=True,
        use_vlm=True, use_celestial_solar=False,
    ),
    "A5": AblationCondition(
        name="A5", label="IMU + baro + mag + airspeed + celestial/solar",
        use_barometer=True, use_magnetometer=True, use_airspeed=True,
        use_vlm=False, use_celestial_solar=True,
    ),
    "A6": AblationCondition(
        name="A6", label="Full system (all sensors)",
        use_barometer=True, use_magnetometer=True, use_airspeed=True,
        use_vlm=True, use_celestial_solar=True,
    ),
}

# CSV columns written by this runner
_CSV_COLUMNS = [
    "run_id", "timestamp",
    "scenario_id", "scenario_label", "mode", "seed",
    "ablation_name", "ablation_label",
    "imu_enabled", "barometer_enabled", "magnetometer_enabled",
    "airspeed_enabled", "vlm_enabled", "celestial_enabled", "solar_enabled",
    "final_error_m", "max_error_m", "mean_error_m", "median_error_m", "p90_error_m",
    "pass_10m", "pass_25m", "pass_50m", "pass_100m",
    "vlm_obs_total", "vlm_passed_confidence", "vlm_rejected_confidence",
    "vlm_rejected_bearing", "vlm_rejected_ambiguity", "vlm_ukf_updates",
    "celestial_updates", "solar_updates",
    "nis_alarm_count",
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
        sys.exit(2)
    return [by_id[s] for s in spec]


def _resolve_ablations(spec) -> List[AblationCondition]:
    if spec == "all" or spec is None:
        return list(ABLATION_CONDITIONS.values())
    if isinstance(spec, str):
        spec = spec.split(",")
    missing = set(spec) - set(ABLATION_CONDITIONS)
    if missing:
        print(f"ERROR: unknown ablation condition(s): {sorted(missing)}", file=sys.stderr)
        print(f"  known: {list(ABLATION_CONDITIONS)}", file=sys.stderr)
        sys.exit(2)
    return [ABLATION_CONDITIONS[a] for a in spec]


def _daytime_cfg(cfg: ScenarioConfig) -> ScenarioConfig:
    return dataclasses.replace(cfg, t_start_utc=_DAYTIME_START_UTC, navigation_mode="daytime")


def _make_row(
    result: dict,
    mode: str,
    seed: int,
    ablation: AblationCondition,
    run_id: str,
    notes: str = "",
) -> dict:
    ts = datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%dT%H:%M:%S")
    solar_on = mode == "daytime" and ablation.use_celestial_solar
    cel_on = mode == "nighttime" and ablation.use_celestial_solar
    return {
        "run_id": run_id,
        "timestamp": ts,
        "scenario_id": result.get("scenario_id", ""),
        "scenario_label": result.get("label", ""),
        "mode": mode,
        "seed": seed,
        "ablation_name": ablation.name,
        "ablation_label": ablation.label,
        "imu_enabled": True,
        "barometer_enabled": ablation.use_barometer,
        "magnetometer_enabled": ablation.use_magnetometer,
        "airspeed_enabled": ablation.use_airspeed,
        "vlm_enabled": ablation.use_vlm,
        "celestial_enabled": cel_on,
        "solar_enabled": solar_on,
        "final_error_m": round(result.get("final_error_m", float("nan")), 2),
        "max_error_m": round(result.get("max_error_m", float("nan")), 2),
        "mean_error_m": round(result.get("mean_error_m", float("nan")), 2),
        "median_error_m": round(result.get("median_error_m", float("nan")), 2),
        "p90_error_m": round(result.get("p90_error_m", float("nan")), 2),
        "pass_10m": result.get("pass_10m", False),
        "pass_25m": result.get("pass_25m", False),
        "pass_50m": result.get("pass_50m", result.get("passed", False)),
        "pass_100m": result.get("pass_100m", False),
        "vlm_obs_total": result.get("vlm_obs_total", 0),
        "vlm_passed_confidence": result.get("vlm_passed_confidence", 0),
        "vlm_rejected_confidence": result.get("vlm_rejected_confidence", 0),
        "vlm_rejected_bearing": result.get("vlm_rejected_bearing", 0),
        "vlm_rejected_ambiguity": result.get("vlm_rejected_ambiguity", 0),
        "vlm_ukf_updates": result.get("vlm_ukf_updates", 0),
        "celestial_updates": result.get("celestial_updates", 0),
        "solar_updates": result.get("solar_updates", 0),
        "nis_alarm_count": result.get("nees_alarm_count", 0),
        "n_landmarks": result.get("n_landmarks", 0),
        "dist_km": round(result.get("dist_km", 0), 2),
        "duration_s": round(result.get("duration_s", 0), 1),
        "notes": notes,
    }


def run_ablations(
    scenarios: List[ScenarioConfig],
    modes: List[str],
    seeds: List[int],
    ablations: List[AblationCondition],
    output_dir: str,
    csv_path: str,
) -> List[dict]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(scenarios) * len(modes) * len(seeds) * len(ablations)
    done = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        for ablation in ablations:
            for mode in modes:
                use_solar = (mode == "daytime") and ablation.use_celestial_solar
                use_celestial = (mode == "nighttime") and ablation.use_celestial_solar

                for base_cfg in scenarios:
                    cfg = _daytime_cfg(base_cfg) if (mode == "daytime") else base_cfg

                    for seed in seeds:
                        cfg_s = dataclasses.replace(cfg, seed=seed)
                        run_id = str(uuid.uuid4())[:8]
                        done += 1
                        print(
                            f"[{done}/{total}] {ablation.name} | {mode} | "
                            f"{cfg_s.scenario_id} | seed={seed}",
                            file=sys.stderr,
                        )
                        try:
                            result = run_scenario(
                                cfg_s,
                                output_dir=output_dir,
                                use_solar=use_solar,
                                use_vlm=ablation.use_vlm,
                                use_celestial=use_celestial,
                                use_barometer=ablation.use_barometer,
                                use_magnetometer=ablation.use_magnetometer,
                                use_airspeed=ablation.use_airspeed,
                            )
                            row = _make_row(result, mode, seed, ablation, run_id)
                        except Exception as exc:
                            print(f"  ERROR: {exc}", file=sys.stderr)
                            row = {col: "" for col in _CSV_COLUMNS}
                            row.update({
                                "run_id": run_id,
                                "scenario_id": cfg_s.scenario_id,
                                "mode": mode, "seed": seed,
                                "ablation_name": ablation.name,
                                "notes": str(exc),
                            })

                        writer.writerow(row)
                        f.flush()
                        rows.append(row)

    return rows


def _print_ablation_summary(rows: List[dict]) -> None:
    from itertools import groupby

    print(f"\n{'='*80}")
    print("  ABLATION SUMMARY (mean final error per condition, nighttime & daytime)")
    print(f"{'='*80}")
    print(f"{'Condition':<6} {'Label':<38} {'Mode':<10} {'Pass50':>8} {'Mean':>8} {'Max':>8}")
    print("-" * 80)

    key = lambda r: (r.get("ablation_name", ""), r.get("mode", ""))
    for (aname, mode), grp in groupby(sorted(rows, key=key), key=key):
        valid = [r for r in grp if r.get("final_error_m") not in ("", None)]
        if not valid:
            continue
        errs = [float(r["final_error_m"]) for r in valid]
        passes = sum(1 for r in valid if str(r.get("pass_50m", "False")).lower() == "true")
        label = ABLATION_CONDITIONS.get(aname, AblationCondition(
            aname, aname, False, False, False, False, False)).label
        print(
            f"{aname:<6} {label:<38} {mode:<10} "
            f"{passes}/{len(valid):>5} "
            f"{statistics.mean(errs):>7.1f}m "
            f"{max(errs):>7.1f}m"
        )

    total = len(rows)
    passed = sum(1 for r in rows if str(r.get("pass_50m", "False")).lower() == "true")
    print(f"\nTotal runs: {total}   Passed (50m): {passed}/{total}")


def main():
    parser = argparse.ArgumentParser(description="UAV navigation sensor ablation runner")
    parser.add_argument("--config", default="configs/default_experiment.yaml")
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--ablations", default=None,
                        help="Comma-separated ablation names (A0–A6) or 'all'")
    parser.add_argument("--output", default=None, dest="output_dir")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: very_short, 1 seed, A0+A6, nighttime only")
    args = parser.parse_args()

    cfg = _load_config(args.config)

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

    output_dir = args.output_dir or cfg.get("output_dir", "results/ablations")
    ablation_spec = args.ablations or cfg.get("ablation_conditions", "all")

    if args.quick:
        scenario_spec = ["very_short"]
        mode_spec = ["nighttime"]
        seeds = [1]
        ablation_spec = "A0,A6"
        output_dir = "results/smoke_test"
        print("Quick mode: very_short, seed=1, A0+A6, nighttime")

    scenarios = _resolve_scenarios(scenario_spec)
    ablations = _resolve_ablations(ablation_spec)
    csv_path = str(Path(output_dir) / "ablation_results.csv")

    total = len(scenarios) * len(mode_spec) * len(seeds) * len(ablations)
    print(f"Ablation sweep: {len(ablations)} condition(s) × {len(scenarios)} scenario(s)"
          f" × {len(mode_spec)} mode(s) × {len(seeds)} seed(s) = {total} runs")
    print(f"  Conditions : {[a.name for a in ablations]}")
    print(f"  Scenarios  : {[s.scenario_id for s in scenarios]}")
    print(f"  Modes      : {mode_spec}")
    print(f"  Seeds      : {seeds}")
    print(f"  Output     : {output_dir}")
    print(f"  CSV        : {csv_path}")
    print()

    rows = run_ablations(scenarios, mode_spec, seeds, ablations, output_dir, csv_path)
    _print_ablation_summary(rows)
    print(f"\nCSV written → {csv_path}")

    # Write a brief Markdown summary alongside the CSV
    md_path = str(Path(output_dir) / "ablation_summary.md")
    _write_markdown_summary(rows, md_path)
    print(f"Markdown summary → {md_path}")


def _write_markdown_summary(rows: List[dict], path: str) -> None:
    from itertools import groupby

    lines = [
        "# Sensor Ablation Summary",
        "",
        "> NOTE: VLM observations are geometry-derived (simulated bearings), "
        "not real VLM inference. These results characterise estimator and "
        "data-association behaviour, not end-to-end VLM perception performance.",
        "",
        "## Results by Ablation Condition",
        "",
        "| Condition | Label | Mode | Seeds | Pass@50m | Mean err (m) | Max err (m) |",
        "|---|---|---|---|---|---|---|",
    ]

    key = lambda r: (r.get("ablation_name", ""), r.get("mode", ""))
    for (aname, mode), grp in groupby(sorted(rows, key=key), key=key):
        valid = [r for r in grp if r.get("final_error_m") not in ("", None)]
        if not valid:
            continue
        errs = [float(r["final_error_m"]) for r in valid]
        passes = sum(1 for r in valid if str(r.get("pass_50m", "False")).lower() == "true")
        label = ABLATION_CONDITIONS.get(aname, AblationCondition(
            aname, aname, False, False, False, False, False)).label
        lines.append(
            f"| {aname} | {label} | {mode} | {len(valid)} "
            f"| {passes}/{len(valid)} "
            f"| {statistics.mean(errs):.1f} "
            f"| {max(errs):.1f} |"
        )

    lines += [
        "",
        "## Sensor Configuration Matrix",
        "",
        "| Condition | IMU | Baro | Mag | Airspeed | VLM | Cel/Solar |",
        "|---|---|---|---|---|---|---|",
    ]
    for abl in ABLATION_CONDITIONS.values():
        def yn(v):
            return "✓" if v else "✗"
        lines.append(
            f"| {abl.name} | ✓ | {yn(abl.use_barometer)} "
            f"| {yn(abl.use_magnetometer)} | {yn(abl.use_airspeed)} "
            f"| {yn(abl.use_vlm)} | {yn(abl.use_celestial_solar)} |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
