#!/usr/bin/env python3
"""
tools/landmark_density_sweep.py

Landmark density sensitivity sweep (L0–L4) for the Baseline (ATL→MCN) scenario.

Tests how pass rate and final error change as we progressively add evenly-spaced
synthetic landmarks along the corridor, from zero to a dense upper-bound grid.

Conditions
----------
L0 : 0 landmarks  (pure dead reckoning + celestial/solar)
L1 : real OSM landmarks only (baseline — current default)
L2 : 2× density  (real + interpolated midpoints)
L3 : 4× density  (real + 3 interpolated points per gap)
L4 : dense upper bound — one landmark every ~10 km

Each condition is swept over N_SEEDS × nighttime mode.

Results
-------
  results/landmark_density/density_results.csv
  results/landmark_density/density_summary.md
  results/landmark_density/density_vs_pasrate.png

Usage:
    conda run -n base python tools/landmark_density_sweep.py
    conda run -n base python tools/landmark_density_sweep.py --seeds 5 --output results/density_quick
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uav_nav.demo.run_scenario import run_scenario
from uav_nav.demo.scenario_configs import (
    SCENARIO_BASELINE, SCENARIO_MUCH_LONGER, ALL_SCENARIOS,
    LandmarkConfig, ScenarioConfig,
)
from uav_nav.vlm.schemas import LandmarkType

_SCENARIO_MAP = {s.scenario_id: s for s in ALL_SCENARIOS}

OUTPUT_DIR = Path("results/landmark_density")
N_SEEDS_DEFAULT = 10
PASS_M = 50.0


# ---------------------------------------------------------------------------
# Landmark interpolation
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _interpolate_landmarks(lms: list[LandmarkConfig], n_interp: int) -> list[LandmarkConfig]:
    """Insert n_interp evenly-spaced synthetic landmarks between each adjacent pair."""
    result = list(lms)
    extras = []
    for i in range(len(lms) - 1):
        a, b = lms[i], lms[i + 1]
        for k in range(1, n_interp + 1):
            t = k / (n_interp + 1)
            lat = a.lat_deg + t * (b.lat_deg - a.lat_deg)
            lon = a.lon_deg + t * (b.lon_deg - a.lon_deg)
            extras.append(LandmarkConfig(
                name=f"synth_{i}_{k}",
                landmark_type=LandmarkType.transformer_substation,
                lat_deg=lat,
                lon_deg=lon,
                visible_range_km=15.0,
            ))
    return result + extras


def _dense_landmarks(base_cfg: ScenarioConfig, spacing_km: float = 10.0) -> list[LandmarkConfig]:
    """Generate one synthetic landmark every spacing_km along the great-circle route."""
    dist = _haversine_km(base_cfg.origin_lat, base_cfg.origin_lon,
                         base_cfg.dest_lat, base_cfg.dest_lon)
    n = max(1, int(dist / spacing_km))
    lms = []
    for i in range(1, n):
        t = i / n
        lat = base_cfg.origin_lat + t * (base_cfg.dest_lat - base_cfg.origin_lat)
        lon = base_cfg.origin_lon + t * (base_cfg.dest_lon - base_cfg.origin_lon)
        lms.append(LandmarkConfig(
            name=f"dense_{i}",
            landmark_type=LandmarkType.transformer_substation,
            lat_deg=lat,
            lon_deg=lon,
            visible_range_km=15.0,
        ))
    return lms


def _build_density_levels(base_cfg: ScenarioConfig) -> list[tuple[str, str, list[LandmarkConfig]]]:
    """Return list of (level_id, label, landmarks) for L0–L4."""
    real = list(base_cfg.landmarks)
    return [
        ("L0", "0 landmarks (dead reckoning)", []),
        ("L1", f"{len(real)} landmarks (real OSM, baseline)", real),
        ("L2", f"{len(_interpolate_landmarks(real, 1))} landmarks (2× density)", _interpolate_landmarks(real, 1)),
        ("L3", f"{len(_interpolate_landmarks(real, 3))} landmarks (4× density)", _interpolate_landmarks(real, 3)),
        ("L4", f"{len(_dense_landmarks(base_cfg))} landmarks (1 per ~10 km, upper bound)", _dense_landmarks(base_cfg)),
    ]


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "scenario_label", "level", "label", "n_landmarks", "seed",
    "final_error_m", "mean_error_m", "max_error_m",
    "pass_50m", "vlm_ukf_updates", "celestial_updates",
]


def run_density_sweep(n_seeds: int, output_dir: Path, base_cfg: ScenarioConfig = SCENARIO_BASELINE) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "density_results.csv"
    levels = _build_density_levels(base_cfg)

    rows: list[dict] = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        for level_id, label, lms in levels:
            print(f"\n── {level_id}: {label} ──")
            level_rows = []
            for seed in range(1, n_seeds + 1):
                cfg = dataclasses.replace(base_cfg, seed=seed, landmarks=lms)
                result = run_scenario(
                    cfg,
                    output_dir=str(output_dir),
                    use_solar=False,
                    use_vlm=True,
                    use_celestial=True,
                )
                fe = result["final_error_m"]
                row = {
                    "scenario_label": base_cfg.label,
                    "level": level_id,
                    "label": label,
                    "n_landmarks": len(lms),
                    "seed": seed,
                    "final_error_m": round(fe, 1),
                    "mean_error_m": round(result["mean_error_m"], 1),
                    "max_error_m": round(result["max_error_m"], 1),
                    "pass_50m": fe < PASS_M,
                    "vlm_ukf_updates": result.get("vlm_ukf_updates", 0),
                    "celestial_updates": result.get("celestial_updates", 0),
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)
                level_rows.append(row)
                status = "✓" if fe < PASS_M else "✗"
                print(f"  seed {seed}: {fe:.0f} m {status}")

            n_pass = sum(1 for r in level_rows if r["pass_50m"])
            errs = [r["final_error_m"] for r in level_rows]
            print(f"  → {n_pass}/{n_seeds} PASS  mean={sum(errs)/len(errs):.0f} m")

    return rows


def _write_summary(rows: list[dict], output_dir: Path) -> None:
    from collections import defaultdict
    import statistics as _stats

    by_level: dict[str, list] = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)

    scenario_names = {r.get("scenario_label", "") for r in rows}
    scenario_name = next(iter(scenario_names - {""}), "")
    lines = [
        "# Landmark Density Sensitivity\n",
        f"## Scenario: {scenario_name}\n" if scenario_name else "## Scenario\n",
        "| Level | Description | N landmarks | Pass@50m | Mean err | Median | Max |",
        "|---|---|---|---|---|---|---|",
    ]
    for level_id in ["L0", "L1", "L2", "L3", "L4"]:
        grp = by_level.get(level_id, [])
        if not grp:
            continue
        errs = [r["final_error_m"] for r in grp]
        n_pass = sum(1 for r in grp if r["pass_50m"])
        n = len(grp)
        label = grp[0]["label"]
        n_lm = grp[0]["n_landmarks"]
        lines.append(
            f"| {level_id} | {label} | {n_lm} | {n_pass}/{n} |"
            f" {_stats.mean(errs):.1f} m | {_stats.median(errs):.1f} m | {max(errs):.1f} m |"
        )

    out = output_dir / "density_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSummary → {out}")


def _plot(rows: list[dict], output_dir: Path) -> None:
    from collections import defaultdict
    import statistics as _stats

    by_level: dict[str, list] = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)

    levels = [l for l in ["L0", "L1", "L2", "L3", "L4"] if l in by_level]
    x = [by_level[l][0]["n_landmarks"] for l in levels]
    pass_rates = [sum(1 for r in by_level[l] if r["pass_50m"]) / len(by_level[l]) * 100 for l in levels]
    mean_errs = [_stats.mean(r["final_error_m"] for r in by_level[l]) for l in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(x, pass_rates, "o-", color="#3fb950", lw=2)
    ax1.axhline(100, color="#3fb950", ls=":", lw=0.8, alpha=0.5)
    ax1.set_xlabel("Number of landmarks")
    ax1.set_ylabel("Pass rate @ 50 m (%)")
    ax1.set_title("Pass rate vs landmark density")
    ax1.set_ylim(0, 105)
    for xi, y, l in zip(x, pass_rates, levels):
        ax1.annotate(l, (xi, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax2.plot(x, mean_errs, "s-", color="#f0a030", lw=2)
    ax2.axhline(50, color="#e3b341", ls=":", lw=1.2, label="50 m criterion")
    ax2.set_xlabel("Number of landmarks")
    ax2.set_ylabel("Mean final error (m)")
    ax2.set_title("Mean error vs landmark density")
    ax2.legend(fontsize=8)
    for xi, y, l in zip(x, mean_errs, levels):
        ax2.annotate(l, (xi, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    scenario_names = {r.get("scenario_label", "") for r in rows}
    sc_name = next(iter(scenario_names - {""}), "Scenario")
    fig.suptitle(f"{sc_name} — Landmark Density Sensitivity", y=1.01)
    fig.tight_layout()
    out = output_dir / "density_vs_passrate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {out}")


def main():
    parser = argparse.ArgumentParser(description="Landmark density sensitivity sweep")
    parser.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--output", default=None, help="Output dir (default: results/landmark_density/<scenario_id>)")
    parser.add_argument("--scenario", default="baseline",
                        choices=list(_SCENARIO_MAP.keys()),
                        help="Scenario to sweep (default: baseline)")
    args = parser.parse_args()

    base_cfg = _SCENARIO_MAP[args.scenario]
    out = Path(args.output) if args.output else Path(f"results/landmark_density/{args.scenario}")

    print("=" * 60)
    print("  Landmark Density Sweep  (L0–L4)")
    print(f"  Scenario : {base_cfg.label}")
    print(f"  Mode     : nighttime")
    print(f"  Seeds    : 1–{args.seeds}")
    print("=" * 60)

    rows = run_density_sweep(args.seeds, out, base_cfg)
    _write_summary(rows, out)
    _plot(rows, out)
    print("\nDone.")


if __name__ == "__main__":
    main()
