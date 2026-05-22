#!/usr/bin/env python3
"""
tools/daytime_all5_10seed_sweep.py

10-seed robustness sweep on every scenario (Very Short, Short, Baseline,
Longer, Much Longer) run in DAYTIME mode — solar ephemeris replaces stars.

Mirrors nighttime_10seed_sweep.py but applies the daytime cfg transform
(t_start_utc = July 15 17:00 UTC, navigation_mode="daytime") and calls
run_scenario with use_solar=True.

Per research-standards rule: no seed cherry-picking. Reports pass rate,
mean/median/stdev, and NEES alarm totals per scenario.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uav_nav.config import get_config
from uav_nav.demo.run_scenario import run_scenario
from uav_nav.demo.scenario_configs import ALL_SCENARIOS

# Same daytime start used in uav_nav.demo.run_all_scenarios: 1 PM EDT,
# optimal solar GDOP for Atlanta (GDOP ≈ 0.165°).
_DAYTIME_START_UTC = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)


def _daytime_cfg(cfg):
    return dataclasses.replace(
        cfg,
        t_start_utc=_DAYTIME_START_UTC,
        navigation_mode="daytime",
    )


def sweep_scenario(base_cfg, seeds, output_dir: str):
    rows = []
    for s in seeds:
        cfg = dataclasses.replace(base_cfg, seed=s)
        r = run_scenario(cfg, output_dir=output_dir, use_solar=True)
        rows.append({
            "seed": s,
            "final_error_m": r["final_error_m"],
            "p95_error_m": r["p95_error_m"],
            "vlm_updates": r["vlm_updates"],
            "celestial_updates": r.get("celestial_updates", 0),
            "solar_updates": r.get("solar_updates", 0),
            "nees_alarms": r["nees_alarm_count"],
            "passed": r["passed"],
        })
    return rows


def summarize(name, rows, threshold):
    errs = [r["final_error_m"] for r in rows]
    passes = sum(1 for r in rows if r["passed"])
    return {
        "scenario": name,
        "n_seeds": len(rows),
        "threshold_m": threshold,
        "pass_count": passes,
        "pass_rate": passes / len(rows),
        "mean_m": statistics.mean(errs),
        "median_m": statistics.median(errs),
        "stdev_m": statistics.stdev(errs) if len(errs) > 1 else 0.0,
        "min_m": min(errs),
        "max_m": max(errs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/daytime_all5_sweep")
    parser.add_argument(
        "--output-json",
        default="results/daytime_all5_10seed_sweep.json",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Limit sweep to these scenario_id(s); repeat flag for multiple. "
             "Default: run all 5 scenarios in daytime mode.",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    threshold = get_config().pass_threshold_m
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))

    if args.scenario:
        selected_ids = set(args.scenario)
        scenarios = [c for c in ALL_SCENARIOS if c.scenario_id in selected_ids]
        missing = selected_ids - {c.scenario_id for c in scenarios}
        if missing:
            print(f"ERROR: unknown scenario_id(s): {sorted(missing)}", file=sys.stderr)
            print(f"  known: {[c.scenario_id for c in ALL_SCENARIOS]}", file=sys.stderr)
            sys.exit(2)
    else:
        scenarios = list(ALL_SCENARIOS)

    print(f"Daytime sweep over {len(scenarios)} scenario(s) "
          f"× {len(seeds)}-seed. threshold={threshold} m")
    print(f"Daytime start: {_DAYTIME_START_UTC.isoformat()} (1 PM EDT, optimal solar GDOP)")
    print()

    all_rows = {}
    summaries = []
    for base_cfg in scenarios:
        cfg = _daytime_cfg(base_cfg)
        print(f"\n=== {cfg.label} ({cfg.scenario_id}) — DAYTIME ===")
        print(f"{'seed':>5}  {'final_err':>10}  {'p95':>8}  "
              f"{'vlm':>5}  {'solar':>6}  {'nees':>5}  {'status':>6}")
        rows = sweep_scenario(cfg, seeds, args.output_dir)
        for r in rows:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"{r['seed']:>5}  {r['final_error_m']:>10.1f}  "
                  f"{r['p95_error_m']:>8.1f}  {r['vlm_updates']:>5}  "
                  f"{r['solar_updates']:>6}  "
                  f"{r['nees_alarms']:>5}  {status:>6}")
        summary = summarize(cfg.label, rows, threshold)
        summaries.append(summary)
        all_rows[cfg.scenario_id] = rows
        print(
            f"→ {cfg.label}: {summary['pass_count']}/{summary['n_seeds']} PASS  "
            f"mean={summary['mean_m']:.1f}  med={summary['median_m']:.1f}  "
            f"max={summary['max_m']:.1f}  stdev={summary['stdev_m']:.1f}"
        )

    # Overall table
    print()
    print("=" * 78)
    print(f"{'Scenario':<22}  {'Pass':>5}  {'Mean':>7}  {'Median':>7}  "
          f"{'Min':>6}  {'Max':>6}  {'Stdev':>7}")
    print("-" * 78)
    for s in summaries:
        print(f"{s['scenario']:<22}  {s['pass_count']:>3}/{s['n_seeds']:<2} "
              f"{s['mean_m']:>7.1f}  {s['median_m']:>7.1f}  "
              f"{s['min_m']:>6.1f}  {s['max_m']:>6.1f}  {s['stdev_m']:>7.1f}")
    total_pass = sum(s["pass_count"] for s in summaries)
    total_runs = sum(s["n_seeds"] for s in summaries)
    print("-" * 78)
    print(f"OVERALL (daytime): {total_pass}/{total_runs} PASS "
          f"({100 * total_pass / total_runs:.0f}%)")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summaries": summaries, "per_scenario": all_rows}, f, indent=2)
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
