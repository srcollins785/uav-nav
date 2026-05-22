"""
run_adhoc.py — CLI for running UAV navigation scenarios.

Usage examples:

  # Run a pre-defined YAML scenario
  python -m uav_nav.demo.run_adhoc --config uav_nav/demo/scenarios/baseline.yaml --output-dir ./results

  # List all available pre-defined scenarios
  python -m uav_nav.demo.run_adhoc --list

  # Run with inline arguments (saves a YAML config for reproducibility)
  python -m uav_nav.demo.run_adhoc \\
    --origin-lat 33.641 --origin-lon -84.427 \\
    --dest-lat 32.127 --dest-lon -81.202 \\
    --airspeed 50 \\
    --landmark transformer_substation "Mid-GA Sub" 33.0 -83.0 15.0 \\
    --label "ATL to SAV" --seed 42 \\
    --output-dir ./results \\
    --save-config ./results/atl_sav.yaml

  # Rerun a previously saved config (identical results, same seed)
  python -m uav_nav.demo.run_adhoc --config ./results/atl_sav.yaml --output-dir ./results2
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m uav_nav.demo.run_adhoc",
        description="Run a UAV navigation scenario from a YAML config or inline args.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Source: YAML or inline ---
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--config", metavar="PATH",
        help="Path to a YAML scenario file. All other route/env arguments are ignored.",
    )
    source.add_argument(
        "--list", action="store_true",
        help="List all available pre-defined scenario IDs and exit.",
    )

    # --- Inline route ---
    p.add_argument("--origin-lat", type=float, metavar="DEG",
                   help="Origin latitude (decimal degrees, N positive).")
    p.add_argument("--origin-lon", type=float, metavar="DEG",
                   help="Origin longitude (decimal degrees, E positive).")
    p.add_argument("--origin-alt", type=float, default=500.0, metavar="M",
                   help="Origin altitude MSL in metres (default: 500).")
    p.add_argument("--dest-lat", type=float, metavar="DEG",
                   help="Destination latitude.")
    p.add_argument("--dest-lon", type=float, metavar="DEG",
                   help="Destination longitude.")
    p.add_argument("--airspeed", type=float, default=50.0, metavar="M/S",
                   help="True airspeed in m/s (default: 50).")

    # --- Inline environment ---
    p.add_argument("--mag-decl", type=float, default=-4.9, metavar="DEG",
                   help="Magnetic declination in degrees (default: -4.9).")
    p.add_argument("--mag-anom", type=float, default=3.0, metavar="DEG",
                   help="Magnetic anomaly in degrees (default: 3.0).")
    p.add_argument("--t-start", default="2026-01-28T02:00:00Z", metavar="ISO8601",
                   help="UTC start time (default: 2026-01-28T02:00:00Z).")

    # --- Inline landmark (repeatable) ---
    p.add_argument(
        "--landmark", nargs=5,
        metavar=("TYPE", "NAME", "LAT", "LON", "RANGE_KM"),
        action="append", default=[],
        help=(
            "Add a landmark. TYPE must be a valid LandmarkType name/value "
            "(e.g. transformer_substation). Repeat for multiple landmarks."
        ),
    )

    # --- Meta ---
    p.add_argument("--label", default="", metavar="STR",
                   help="Human-readable scenario label.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility (default: 42).")
    p.add_argument("--scenario-id", default="", metavar="ID",
                   help="Scenario ID used in output filenames (default: derived from label).")

    # --- Output ---
    p.add_argument("--output-dir", default=".", metavar="DIR",
                   help="Directory for PNG and MP4 outputs (default: current dir).")
    p.add_argument("--save-config", metavar="PATH",
                   help="Write the effective scenario to a YAML file for future reruns.")

    return p


def _inline_to_scenario_config(args):
    """Build a ScenarioConfig from parsed inline CLI arguments."""
    from uav_nav.demo.scenario_configs import LandmarkConfig, ScenarioConfig
    from uav_nav.demo.scenario_loader import _get_landmark_type_map

    # Validate required fields
    missing = [f for f, v in [
        ("--origin-lat", args.origin_lat),
        ("--origin-lon", args.origin_lon),
        ("--dest-lat", args.dest_lat),
        ("--dest-lon", args.dest_lon),
    ] if v is None]
    if missing:
        print(f"error: the following arguments are required: {', '.join(missing)}",
              file=sys.stderr)
        sys.exit(2)

    lm_type_map = _get_landmark_type_map()
    landmarks = []
    for (ltype, lname, llat, llon, lrange) in args.landmark:
        if ltype not in lm_type_map:
            valid = sorted(lm_type_map.keys())
            print(f"error: unknown landmark type '{ltype}'. Valid: {valid}", file=sys.stderr)
            sys.exit(2)
        landmarks.append(LandmarkConfig(
            name=lname,
            landmark_type=lm_type_map[ltype],
            lat_deg=float(llat),
            lon_deg=float(llon),
            visible_range_km=float(lrange),
        ))

    t_str = args.t_start.replace("Z", "+00:00")
    t_start = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)

    label = args.label or f"Ad hoc ({args.origin_lat:.2f}N → {args.dest_lat:.2f}N)"
    sid = args.scenario_id or label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("→", "to").replace(",", "")[:32]

    return ScenarioConfig(
        scenario_id=sid,
        label=label,
        description="Ad hoc scenario from CLI arguments.",
        origin_lat=args.origin_lat,
        origin_lon=args.origin_lon,
        origin_alt_m=args.origin_alt,
        dest_lat=args.dest_lat,
        dest_lon=args.dest_lon,
        airspeed_ms=args.airspeed,
        t_start_utc=t_start,
        landmarks=landmarks,
        mag_declination_deg=args.mag_decl,
        mag_anomaly_deg=args.mag_anom,
        seed=args.seed,
        plot_filename=f"{sid}.png",
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --list
    if args.list:
        from uav_nav.demo.scenario_loader import list_scenarios
        ids = list_scenarios()
        if ids:
            print("Available scenarios:")
            for sid in ids:
                print(f"  {sid}")
        else:
            print("No scenarios found in uav_nav/demo/scenarios/.")
        return

    # Load config
    if args.config:
        from uav_nav.demo.scenario_loader import load_scenario
        cfg = load_scenario(args.config)
        print(f"Loaded scenario '{cfg.scenario_id}' from {args.config}", file=sys.stderr)
    else:
        cfg = _inline_to_scenario_config(args)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally save effective config as YAML (before run, so config persists even if run fails)
    if args.save_config:
        from uav_nav.demo.scenario_loader import scenario_to_yaml
        yaml_text = scenario_to_yaml(cfg)
        save_path = Path(args.save_config)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(yaml_text)
        print(f"Config saved → {save_path}", file=sys.stderr)

    # Run
    from uav_nav.demo.run_scenario import run_scenario
    result = run_scenario(cfg, output_dir=str(output_dir))

    # Print summary JSON (excluding large arrays)
    summary = {k: v for k, v in result.items()
               if k not in ("log_entries", "error_samples", "vlm_events",
                            "celestial_events", "landmarks")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
