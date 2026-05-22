#!/usr/bin/env python3
"""
scripts/run_smoke_test.py

Quick smoke test: verifies the full pipeline executes without crashing.

Runs:
  - Scenario  : very_short (ATL → Stone Mountain, ~25 km, ~8 min)
  - Mode      : nighttime
  - Seed      : 1
  - Ablations : A0 (IMU only) and A6 (full system)

Produces:
  - results/smoke_test/smoke_test_results.csv
  - prints PASS / FAIL for each ablation condition

Usage:
    conda run -n base python scripts/run_smoke_test.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_ablations import (
    ABLATION_CONDITIONS,
    _make_row,
    _write_markdown_summary,
)
from uav_nav.demo.run_scenario import run_scenario
from uav_nav.demo.scenario_configs import SCENARIO_VERY_SHORT
import dataclasses
import uuid

OUTPUT_DIR = Path("results/smoke_test")
CSV_PATH = OUTPUT_DIR / "smoke_test_results.csv"

_CSV_COLUMNS = [
    "run_id", "timestamp", "scenario_id", "scenario_label", "mode", "seed",
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

SMOKE_CONDITIONS = ["A0", "A6"]
SMOKE_SEED = 1
SMOKE_MODE = "nighttime"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  UAV Navigation Smoke Test")
    print(f"  Scenario : {SCENARIO_VERY_SHORT.label}")
    print(f"  Mode     : {SMOKE_MODE}")
    print(f"  Seed     : {SMOKE_SEED}")
    print(f"  Ablations: {SMOKE_CONDITIONS}")
    print("=" * 60)

    rows = []
    crashed = []  # track runs that threw exceptions (pipeline failures)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        for aname in SMOKE_CONDITIONS:
            ablation = ABLATION_CONDITIONS[aname]
            cfg = dataclasses.replace(SCENARIO_VERY_SHORT, seed=SMOKE_SEED)
            run_id = str(uuid.uuid4())[:8]

            print(f"\nRunning {aname} ({ablation.label})...")
            t0 = time.time()
            try:
                result = run_scenario(
                    cfg,
                    output_dir=str(OUTPUT_DIR),
                    use_solar=False,
                    use_vlm=ablation.use_vlm,
                    use_celestial=ablation.use_celestial_solar,
                    use_barometer=ablation.use_barometer,
                    use_magnetometer=ablation.use_magnetometer,
                    use_airspeed=ablation.use_airspeed,
                )
                wall = time.time() - t0
                row = _make_row(result, SMOKE_MODE, SMOKE_SEED, ablation, run_id)
                err = result["final_error_m"]
                passed_50 = result.get("pass_50m", result.get("passed", False))
                # Report navigation performance for information only.
                # Ablation conditions without all sensors (e.g. A0 IMU-only)
                # are expected to fail the 50 m criterion — that is the point
                # of the ablation, not a pipeline failure.
                nav_status = "pass@50m" if passed_50 else "fail@50m (expected for degraded configs)"
                print(f"  Final error : {err:.1f} m   [{nav_status}]")
                print(f"  VLM updates : {result.get('vlm_updates', 0)}")
                print(f"  NIS alarms  : {result.get('nees_alarm_count', 0)}")
                print(f"  Wall time   : {wall:.1f}s")
                print(f"  Pipeline    : OK")
            except Exception as exc:
                wall = time.time() - t0
                crashed.append(aname)
                print(f"  Pipeline    : CRASHED — {exc}", file=sys.stderr)
                row = {col: "" for col in _CSV_COLUMNS}
                row.update({
                    "run_id": run_id, "ablation_name": aname,
                    "scenario_id": cfg.scenario_id, "mode": SMOKE_MODE,
                    "seed": SMOKE_SEED, "notes": f"CRASH: {exc}",
                })

            writer.writerow(row)
            rows.append(row)

    print(f"\nCSV → {CSV_PATH}")
    _write_markdown_summary(rows, str(OUTPUT_DIR / "smoke_test_summary.md"))
    print(f"MD  → {OUTPUT_DIR / 'smoke_test_summary.md'}")
    print()
    print("=" * 60)
    if crashed:
        print(f"  Smoke test: FAIL ✗  (pipeline crashed for: {crashed})")
        print("  NOTE: navigation performance failures are expected for")
        print("  degraded sensor configs (e.g. A0 IMU-only) — only")
        print("  exception/crash failures cause the smoke test to fail.")
    else:
        print("  Smoke test: PASS ✓  (all conditions ran without crashing)")
        print("  Navigation performance per condition shown above.")
    print("=" * 60)
    sys.exit(1 if crashed else 0)


if __name__ == "__main__":
    main()
