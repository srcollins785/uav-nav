#!/usr/bin/env python3
"""
tools/daytime_baseline_sweep.py

Control experiment: 10-seed robustness sweep using the ORIGINAL 2-landmark
daytime config (Macon Water Body + Warner Robins AF Substation only).

Purpose: establish what the real "before enrichment" distribution looks like
across seeds.  The previously cited 77 m was a single seed (501).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uav_nav.config import get_config
from uav_nav.demo import daytime_scenario as ds
from uav_nav.demo.scenario_configs import LandmarkConfig
from uav_nav.vlm.schemas import LandmarkType


def patched_make_config(seed: int = 501):
    cfg = ds._make_daytime_config.__wrapped__(seed=seed) if hasattr(
        ds._make_daytime_config, "__wrapped__"
    ) else _orig_make_config(seed=seed)
    cfg.landmarks = [
        LandmarkConfig("Macon Water Body", LandmarkType.water_body,
                       32.810, -83.730, 20.0),
        LandmarkConfig("Warner Robins AF Substation",
                       LandmarkType.transformer_substation,
                       32.630, -83.625, 15.0, 15.0),
    ]
    return cfg


_orig_make_config = ds._make_daytime_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/daytime_baseline_sweep")
    parser.add_argument("--output-json",
                        default="results/daytime_baseline_sweep.json")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ds._make_daytime_config = patched_make_config
    threshold = get_config().pass_threshold_m

    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    rows = []
    print(f"Daytime ATL→MCN BASELINE sweep (2 landmarks only): "
          f"{len(seeds)} seeds, threshold={threshold} m")
    print(f"{'seed':>5}  {'final_err':>10}  {'p95':>8}  "
          f"{'vlm':>5}  {'solar':>6}  {'nees':>5}  {'status':>8}")
    print("-" * 60)

    for s in seeds:
        r = ds.run_daytime_scenario(seed=s, use_solar=True, use_vlm=True,
                                     output_dir=args.output_dir)
        rows.append({
            "seed": s,
            "final_error_m": r["final_error_m"],
            "p95_error_m": r["p95_error_m"],
            "vlm_updates": r["vlm_updates"],
            "solar_updates": r["solar_updates"],
            "nees_alarms": r["nees_alarm_count"],
            "passed": r["passed"],
        })
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{s:>5}  {r['final_error_m']:>10.1f}  {r['p95_error_m']:>8.1f}  "
              f"{r['vlm_updates']:>5}  {r['solar_updates']:>6}  "
              f"{r['nees_alarm_count']:>5}  {status:>8}")

    final_errs = [r["final_error_m"] for r in rows]
    passes = sum(1 for r in rows if r["passed"])
    nees_any = sum(1 for r in rows if r["nees_alarms"] > 0)

    summary = {
        "config": "2 landmarks (baseline)",
        "n_seeds": len(seeds),
        "threshold_m": threshold,
        "pass_count": passes,
        "pass_rate": passes / len(seeds),
        "mean_m": statistics.mean(final_errs),
        "median_m": statistics.median(final_errs),
        "stdev_m": statistics.stdev(final_errs) if len(final_errs) > 1 else 0.0,
        "min_m": min(final_errs),
        "max_m": max(final_errs),
        "seeds_with_nees_alarm": nees_any,
    }
    print("-" * 60)
    print(f"Pass rate: {passes}/{len(seeds)}  ({100*passes/len(seeds):.0f}%)")
    print(f"Final error: mean={summary['mean_m']:.1f}  med={summary['median_m']:.1f}  "
          f"min={summary['min_m']:.1f}  max={summary['max_m']:.1f}  "
          f"stdev={summary['stdev_m']:.1f} m")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_seed": rows}, f, indent=2)
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
