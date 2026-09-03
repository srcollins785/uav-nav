#!/usr/bin/env python3
"""
tools/verify_reproduction.py

Checks that THIS checkout, in YOUR environment, reproduces the per-seed
numbers published in results/paper/ — the same numbers the MILCOM 2026 paper
reports.

Run it before trusting anything else:

    python tools/verify_reproduction.py                  # fast: very_short, 3 seeds
    python tools/verify_reproduction.py --scenario baseline --seeds 5
    python tools/verify_reproduction.py --all            # every scenario, 10 seeds (slow)

WHY THIS EXISTS
---------------
Two things silently change the numbers, and both have bitten this project:

1. A missing celestial solver. third_party/inverse_celnav must be present.
   If it is not, uav_nav.celestial.solver prints one warning line and
   continues with celestial fixes DISABLED -- easy to scroll past, and it
   moves very_short seed 1 from 27.3 m to 17.5 m. This script fails loudly
   instead.

2. Library versions. The published numbers were produced on the authors'
   machine; NumPy/SciPy build differences shift results by a fraction of a
   metre. See requirements-repro.txt and the tolerance note below.

TOLERANCE
---------
Default 1.0 m absolute. Drift between the authors' recorded values and a
clean container (Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1) grows with route
length, because the filter integrates for longer before the final fix:

    scenario      route     seeds   worst delta
    very_short     30 km      3        0.23 m
    much_longer   435 km     10        0.82 m

So on the long routes the 1.0 m default leaves only ~0.2 m of headroom. A
third party on a different BLAS may well exceed it and see a FAIL that means
nothing. If you are checking `longer` or `much_longer`, pass a wider bound:

    python tools/verify_reproduction.py --scenario much_longer --seeds 10 \
        --tolerance 2.0

The default is deliberately left at 1.0 m so that the short scenarios --
where drift really is sub-decimetre -- still fail loudly on a misconfigured
checkout. Judge a result by the reported worst delta, not only by PASS/FAIL:
a delta near 1 m on a 435 km route is ordinary numerical drift, while the
same delta on very_short is not.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PUBLISHED = REPO / "results" / "paper" / "nighttime_10seed_sweep.json"
DEFAULT_TOL_M = 1.0


def _fail(msg: str) -> "NoReturn":  # noqa: F821
    print(f"\nFAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_solver() -> None:
    from uav_nav.celestial.solver import is_available
    from uav_nav.config import get_config

    path = get_config().inverse_celnav_abs_path()
    if not is_available():
        _fail(
            "the celestial solver is not importable, so celestial fixes are "
            "DISABLED and the published numbers cannot be reproduced.\n"
            f"       expected at: {path}\n"
            "       config key:  inverse_celnav_path (uav_nav/config.yaml)\n"
            "       This repository ships the solver at third_party/inverse_celnav;\n"
            "       if you moved or removed it, restore it or repoint the config."
        )
    print(f"  celestial solver: OK  ({path})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="very_short")
    ap.add_argument("--seeds", type=int, default=3, help="how many seeds, from seed 1")
    ap.add_argument("--all", action="store_true", help="every scenario, 10 seeds")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOL_M, metavar="M",
                    help="absolute agreement bound in metres (default 1.0). "
                         "Drift grows with route length -- measured worst was "
                         "0.23 m on very_short but 0.82 m on much_longer -- so "
                         "use 2.0 for the long routes. See the TOLERANCE note "
                         "in this file's docstring.")
    args = ap.parse_args()

    if not PUBLISHED.exists():
        _fail(f"published reference not found: {PUBLISHED}")
    published = json.load(PUBLISHED.open())["per_scenario"]

    from uav_nav.demo.run_scenario import run_scenario
    from uav_nav.demo.scenario_configs import ALL_SCENARIOS

    by_id = {s.scenario_id: s for s in ALL_SCENARIOS}
    scenarios = (["very_short", "short", "baseline", "longer", "much_longer"]
                 if args.all else [args.scenario])
    n_seeds = 10 if args.all else args.seeds

    print("Reproduction check")
    print("=" * 62)
    check_solver()
    print(f"  reference:        {PUBLISHED.relative_to(REPO)}")
    print(f"  tolerance:        {args.tolerance} m absolute\n")

    worst = 0.0
    failures = []
    for sid in scenarios:
        if sid not in by_id:
            _fail(f"unknown scenario '{sid}'")
        if sid not in published:
            _fail(f"no published data for scenario '{sid}'")
        ref_rows = published[sid][:n_seeds]
        print(f"{sid}")
        print(f"  {'seed':>4s} {'published':>11s} {'yours':>11s} {'delta':>9s}")
        for row in ref_rows:
            seed = row["seed"]
            cfg = dataclasses.replace(by_id[sid], seed=seed, navigation_mode="nighttime")
            res = run_scenario(cfg, output_dir="/tmp/verify_reproduction",
                               use_solar=False, make_plots=False)
            got = float(res["final_error_m"])
            ref = float(row["final_error_m"])
            delta = abs(got - ref)
            worst = max(worst, delta)
            flag = "" if delta <= args.tolerance else "   <-- OUT OF TOLERANCE"
            if flag:
                failures.append((sid, seed, ref, got, delta))
            print(f"  {seed:4d} {ref:10.3f}m {got:10.3f}m {delta:8.3f}m{flag}")
        print()

    print("=" * 62)
    if failures:
        print(f"FAIL: {len(failures)} run(s) outside {args.tolerance} m "
              f"(worst delta {worst:.3f} m)")
        print("      A gap this large is a configuration difference, not numerical")
        print("      noise. Check the solver, then requirements-repro.txt.")
        return 1
    print(f"PASS: all {sum(len(published[s][:n_seeds]) for s in scenarios)} run(s) "
          f"within {args.tolerance} m (worst delta {worst:.3f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
