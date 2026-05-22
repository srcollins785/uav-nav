"""
daytime_showcase_scenario.py — Solar navigation isolation demo.

Route: ATL (33.64°N, 84.43°W) → Huntsville AL (34.73°N, 86.59°W), ~220 km NW
Start: 2026-07-15 14:00 UTC (10:00 AM EDT) — solar elevation ~55° throughout

No VLM landmarks are registered, so the three runs show:
  Run A — Dead reckoning : IMU + Baro + Mag only     (error climbs to km-scale)
  Run B — Solar-aided    : IMU + Baro + Mag + Solar  (solar is the ONLY fix source)
  Run C — Full system    : same as B (no landmarks → VLM never fires anyway)

Runs A vs B demonstrate solar contribution unambiguously.

Usage:
    python -m uav_nav.demo.daytime_showcase_scenario
    python -m uav_nav.demo.daytime_showcase_scenario --seed 7 --output-dir results/
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Optional

PASS_THRESHOLD_M = 500.0

_ORIGIN = (33.6413, -84.4277, 463.0)   # ATL: lat, lon, alt_m
_DEST   = (34.7304, -86.5861, 200.0)   # Huntsville Executive Airport, AL


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _make_showcase_config(seed: int = 7):
    from uav_nav.demo.scenario_configs import ScenarioConfig
    return ScenarioConfig(
        scenario_id="showcase_solar",
        label="Solar Showcase (ATL→HSV)",
        description=(
            "Landmark-sparse ATL→Huntsville flight.\n"
            "No VLM corrections — solar is the only external fix source.\n"
            "Demonstrates GPS-denied navigation using only the sun."
        ),
        origin_lat=_ORIGIN[0],
        origin_lon=_ORIGIN[1],
        origin_alt_m=_ORIGIN[2],
        dest_lat=_DEST[0],
        dest_lon=_DEST[1],
        airspeed_ms=50.0,
        t_start_utc=datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc),
        landmarks=[],   # intentionally empty — no VLM corrections possible
        seed=seed,
        plot_filename="scenario_showcase_solar.png",
    )


def _run_one(cfg, use_solar: bool, use_mag: bool, label: str, output_dir: str) -> dict:
    """Run the showcase scenario with or without solar fixes and magnetometer."""
    from uav_nav.demo.generic_scenario import generate_scenario_from_config, _compute_heading
    from uav_nav.fusion.constraint_converter import ConstraintConverter
    from uav_nav.fusion.sensor_scheduler import SensorScheduler
    from uav_nav.navigation.state import UAVState
    from uav_nav.navigation.ukf import UAVKalmanFilter
    from uav_nav.solar.solver import get_solar_fix_warmstart
    from uav_nav.solar.ephemeris import solar_position, get_illumination_mode

    dist_m = _haversine_m(cfg.origin_lat, cfg.origin_lon, cfg.dest_lat, cfg.dest_lon)
    duration_s = dist_m / cfg.airspeed_ms

    heading = _compute_heading(cfg.origin_lat, cfg.origin_lon, cfg.dest_lat, cfg.dest_lon)
    heading_rad = math.radians(heading)

    initial_state = UAVState(
        lat_deg=cfg.origin_lat,
        lon_deg=cfg.origin_lon,
        alt_m=cfg.origin_alt_m,
        v_north_ms=cfg.airspeed_ms * math.cos(heading_rad),
        v_east_ms=cfg.airspeed_ms * math.sin(heading_rad),
        v_down_ms=0.0,
        heading_deg=heading,
        mag_bias_deg=cfg.mag_anomaly_deg,
    )

    kf = UAVKalmanFilter(initial_state)
    converter = ConstraintConverter()
    scheduler = SensorScheduler()
    # No landmarks registered → VLM obs will always have empty landmark list

    frames = generate_scenario_from_config(cfg)

    log_entries = []
    error_samples = []
    solar_events = []
    next_log_t = 0.0
    log_interval_s = max(1.0, duration_s / 1800.0)

    for frame in frames:
        dt = frame.imu.dt_s
        scheduler.tick(dt)
        kf.set_timestamp(frame.timestamp_utc)

        kf.predict(frame.imu)

        if scheduler.airspeed_due:
            kf.update_velocity_from_heading(cfg.airspeed_ms, sigma_ms=2.0)
        if frame.baro and scheduler.baro_due:
            kf.update_barometer(frame.baro)
        if use_mag and frame.mag and scheduler.mag_due:
            kf.update_magnetometer(frame.mag)
        # No VLM — landmarks list is empty so converter.process() always returns 0

        if use_solar and frame.solar_obs is not None:
            _s = kf.state
            fix = get_solar_fix_warmstart(
                frame.solar_obs,
                frame.timestamp_utc,
                init_lat_deg=_s.lat_deg,
                init_lon_deg=_s.lon_deg,
            )
            if fix is not None and fix.optimizer_success and fix.rmse_normalized < 3.0:
                kf.update_solar_fix(fix)
                solar_events.append(frame.t_s)

        state = kf.state
        err_m = _haversine_m(state.lat_deg, state.lon_deg,
                             frame.truth.lat_deg, frame.truth.lon_deg)
        error_samples.append(err_m)

        if frame.t_s >= next_log_t:
            sun = solar_position(state.lat_deg, state.lon_deg, frame.timestamp_utc)
            speed_ms = math.sqrt(state.v_north_ms ** 2 + state.v_east_ms ** 2)
            log_entries.append({
                "t_s": round(frame.t_s, 1),
                "lat": round(state.lat_deg, 6),
                "lon": round(state.lon_deg, 6),
                "error_m": round(err_m, 1),
                "true_lat": round(frame.truth.lat_deg, 6),
                "true_lon": round(frame.truth.lon_deg, 6),
                "heading_deg": round(state.heading_deg, 2),
                "speed_ms": round(speed_ms, 2),
                "alt_m": round(state.alt_m, 1),
                "gyro_bias_dps": round(state.gyro_bias_z_dps, 4),
                "solar_elevation_deg": round(sun.elevation_deg, 2),
                "solar_azimuth_deg": round(sun.azimuth_deg, 2),
            })
            next_log_t += log_interval_s

    final_state = kf.state
    last_truth = frames[-1].truth
    final_err = _haversine_m(final_state.lat_deg, final_state.lon_deg,
                             last_truth.lat_deg, last_truth.lon_deg)
    errors_sorted = sorted(error_samples)
    p95 = errors_sorted[int(len(errors_sorted) * 0.95)]
    mean_err = sum(error_samples) / len(error_samples)
    passed = final_err < PASS_THRESHOLD_M

    _est = timezone(timedelta(hours=-5))
    _ts = datetime.now(_est).strftime("%Y%m%d-%H%M%SEST")
    _tag = "solar" if use_solar else "dr"
    plot_path = os.path.join(output_dir, f"showcase_{_tag}_{_ts}.png")
    video_path = os.path.join(output_dir, f"showcase_{_tag}_{_ts}.mp4")

    return {
        "scenario_id": cfg.scenario_id,
        "label": label,
        "duration_s": duration_s,
        "dist_km": dist_m / 1000.0,
        "final_error_m": final_err,
        "p95_error_m": p95,
        "mean_error_m": mean_err,
        "vlm_updates": 0,
        "celestial_updates": 0,
        "solar_updates": len(solar_events),
        "nees_alarm_count": kf.nees_alarm_count,
        "passed": passed,
        "plot_path": plot_path,
        "video_path": video_path,
        "log_entries": log_entries,
        "error_samples": error_samples,
        "vlm_events": [],
        "celestial_events": solar_events,   # nav_viz compat key
        "solar_events": solar_events,
        "landmarks": [],
        "origin_lat": cfg.origin_lat,
        "origin_lon": cfg.origin_lon,
        "dest_lat": cfg.dest_lat,
        "dest_lon": cfg.dest_lon,
        "airspeed_ms": cfg.airspeed_ms,
        "seed": cfg.seed,
        "t_start_utc": cfg.t_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def run_showcase(seed: int = 7, output_dir: str = ".") -> dict:
    """
    Run the 3-way comparison and generate the demo video.

    Returns a dict with keys 'dead_reckoning', 'solar_aided', and
    'improvement_m' for downstream use.
    """
    cfg = _make_showcase_config(seed=seed)

    dist_m = _haversine_m(cfg.origin_lat, cfg.origin_lon, cfg.dest_lat, cfg.dest_lon)
    dist_km = dist_m / 1000.0
    duration_min = (dist_m / cfg.airspeed_ms) / 60.0

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Solar Showcase — ATL → Huntsville AL", file=sys.stderr)
    print(f"  Distance: {dist_km:.1f} km  |  Duration: {duration_min:.0f} min", file=sys.stderr)
    print(f"  Landmarks: NONE (solar is the only position fix source)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    print("\n[1/2] Dead reckoning (IMU+Baro only — mag disabled to simulate", file=sys.stderr)
    print("       magnetically disturbed environment)...", file=sys.stderr)
    result_dr = _run_one(cfg, use_solar=False, use_mag=False,
                         label=f"Dead Reckoning — ATL→HSV (seed={seed})",
                         output_dir=output_dir)
    _print_result(result_dr)

    print("\n[2/2] Solar-aided (IMU+Baro+Solar — same mag-denied environment)...",
          file=sys.stderr)
    result_sol = _run_one(cfg, use_solar=True, use_mag=False,
                          label=f"Solar-Aided — ATL→HSV (seed={seed})",
                          output_dir=output_dir)
    _print_result(result_sol)

    improvement_m = result_dr["final_error_m"] - result_sol["final_error_m"]
    pct = (improvement_m / result_dr["final_error_m"] * 100.0
           if result_dr["final_error_m"] > 0 else 0.0)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"COMPARISON SUMMARY", file=sys.stderr)
    print(f"  Dead reckoning final error : {result_dr['final_error_m']:.0f} m  "
          f"({'PASS' if result_dr['passed'] else 'FAIL'})", file=sys.stderr)
    print(f"  Solar-aided final error    : {result_sol['final_error_m']:.0f} m  "
          f"({'PASS' if result_sol['passed'] else 'FAIL'})", file=sys.stderr)
    print(f"  Solar improvement          : {improvement_m:+.0f} m  ({pct:.0f}% reduction)",
          file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Generate comparison video
    try:
        from nav_viz.solar_comparison_video import generate_solar_comparison_video
        _est = timezone(timedelta(hours=-5))
        _ts = datetime.now(_est).strftime("%Y%m%d-%H%M%SEST")
        video_path = os.path.join(output_dir, f"showcase_comparison_{_ts}.mp4")
        generate_solar_comparison_video(result_dr, result_sol, video_path)
        print(f"\nDemo video → {video_path}", file=sys.stderr)
    except Exception as exc:
        print(f"\n[warn] Could not generate comparison video: {exc}", file=sys.stderr)

    return {
        "dead_reckoning": result_dr,
        "solar_aided": result_sol,
        "improvement_m": improvement_m,
        "improvement_pct": pct,
    }


def _print_result(r: dict) -> None:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"  Final error : {r['final_error_m']:.0f} m  [{status}]", file=sys.stderr)
    print(f"  P95 error   : {r['p95_error_m']:.0f} m", file=sys.stderr)
    print(f"  Solar fixes : {r['solar_updates']}", file=sys.stderr)
    print(f"  NEES alarms : {r['nees_alarm_count']}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solar navigation isolation demo — ATL→Huntsville, no VLM landmarks."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    result = run_showcase(seed=args.seed, output_dir=args.output_dir)
