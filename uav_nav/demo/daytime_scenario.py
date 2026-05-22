"""
Daytime solar navigation scenario — ATL → MCN at 14:00 UTC (10:00 AM EDT).

Demonstrates 24-hour celestial aiding: the same ATL→MCN route as the night
baseline but flown during the day, using solar position fixes instead of
star-based celestial fixes.  At 14:00 UTC the sun elevation over Atlanta is
~55°, well into the DAYTIME zone with good azimuth separability.

Usage:
    python -m uav_nav.demo.daytime_scenario
    python -m uav_nav.demo.daytime_scenario --seed 77 --output-dir results/

The script runs the navigation filter twice (with and without solar fixes) and
prints a side-by-side comparison so the impact of solar navigation is visible.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from uav_nav.config import get_config as _get_config


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _make_daytime_config(seed: int = 501):
    """
    Build a ScenarioConfig for the daytime ATL→MCN route.

    t_start_utc = 2026-07-15 17:00:00 UTC (1:00 PM EDT).
    Optimal solar geometry: elevation ~74°, GDOP ~0.175° (best of day).
    Solar azimuth sweeps 139°→183° during 43-min flight — strong lon discriminability.
    """
    from uav_nav.demo.scenario_configs import ScenarioConfig, LandmarkConfig
    from uav_nav.vlm.schemas import LandmarkType

    return ScenarioConfig(
        scenario_id="daytime_solar",
        label="Daytime Solar (ATL→MCN)",
        description=(
            "Daytime ATL→MCN flight using solar position fixes.\n"
            "17:00 UTC (13:00 EDT) — sun elevation ~74°, GDOP ~0.175° (optimal).\n"
            "Demonstrates 24-hour celestial navigation aiding."
        ),
        origin_lat=33.6413,
        origin_lon=-84.4277,
        origin_alt_m=463.0,
        dest_lat=32.6930,
        dest_lon=-83.6490,
        airspeed_ms=50.0,
        t_start_utc=datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc),
        landmarks=[
            # Real OSM substations along the ATL→MCN corridor.
            # Same corridor as SCENARIO_BASELINE; real infrastructure visible
            # both day and night.  Fills the first-30-km gap that previously
            # left the filter in dead-reckoning for ~31 min after take-off.
            LandmarkConfig("Battle Creek Substation",
                           LandmarkType.transformer_substation,
                           33.5496, -84.3430, 15.0, 15.0),
            LandmarkConfig("Georgia Power (Griffin)",
                           LandmarkType.transformer_substation,
                           33.3001, -84.1890, 15.0, 15.0),
            LandmarkConfig("Georgia Power (Barnesville)",
                           LandmarkType.transformer_substation,
                           33.2243, -84.0608, 15.0, 15.0),
            LandmarkConfig("Forsyth Substation",
                           LandmarkType.transformer_substation,
                           33.0475, -83.9396, 15.0, 15.0),
            LandmarkConfig("Twin Pines Substation",
                           LandmarkType.transformer_substation,
                           32.8356, -83.7458, 15.0, 15.0),
            # Daytime-specific landmarks near destination
            LandmarkConfig("Macon Water Body",
                           LandmarkType.water_body,
                           32.810, -83.730, 20.0),
            LandmarkConfig("Warner Robins AF Substation",
                           LandmarkType.transformer_substation,
                           32.630, -83.625, 15.0, 15.0),
        ],
        seed=seed,
        navigation_mode="daytime",
        plot_filename="scenario_daytime_solar.png",
    )


def run_daytime_scenario(seed: int = 501, use_solar: bool = True,
                         use_vlm: bool = True,
                         output_dir: str = ".") -> dict:
    """
    Run the daytime solar navigation scenario.

    Args:
        seed       : RNG seed for reproducibility.
        use_solar  : If True, apply solar fixes when available.
        output_dir : Directory to write output files.

    Returns:
        Result dict compatible with run_scenario.run_scenario output format.
    """
    from uav_nav.demo.generic_scenario import generate_scenario_from_config
    from uav_nav.demo.generic_scenario import _compute_heading
    from uav_nav.fusion.constraint_converter import ConstraintConverter
    from uav_nav.fusion.sensor_scheduler import SensorScheduler
    from uav_nav.navigation.state import UAVState
    from uav_nav.navigation.ukf import UAVKalmanFilter
    from uav_nav.solar.solver import get_solar_fix_warmstart

    PASS_THRESHOLD_M = _get_config().pass_threshold_m
    cfg = _make_daytime_config(seed=seed)

    dist_m = _haversine_m(cfg.origin_lat, cfg.origin_lon,
                          cfg.dest_lat, cfg.dest_lon)
    duration_s = dist_m / cfg.airspeed_ms

    parts = []
    if use_vlm:
        parts.append("VLM")
    if use_solar:
        parts.append("solar")
    mode_label = ("+".join(parts) + " aided") if parts else "dead reckoning only"
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Daytime Solar Scenario ({mode_label})", file=sys.stderr)
    _edt_start = cfg.t_start_utc + timedelta(hours=-4)
    print(f"  UTC start: {cfg.t_start_utc.strftime('%Y-%m-%d %H:%M')} UTC "
          f"({_edt_start.strftime('%I:%M %p')} EDT)", file=sys.stderr)
    print(f"  Route:  ATL ({cfg.origin_lat:.3f}°N, {cfg.origin_lon:.3f}°W)"
          f" → MCN ({cfg.dest_lat:.3f}°N, {cfg.dest_lon:.3f}°W)", file=sys.stderr)
    print(f"  Distance: {dist_m/1000:.1f} km  |  Duration: {duration_s/60:.1f} min",
          file=sys.stderr)
    print(f"Generating scenario frames...", file=sys.stderr)

    frames = generate_scenario_from_config(cfg)
    print(f"  {len(frames):,} frames generated.", file=sys.stderr)

    # --- Initialise UKF ---
    heading = _compute_heading(cfg.origin_lat, cfg.origin_lon,
                               cfg.dest_lat, cfg.dest_lon)
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

    for lm in cfg.landmarks:
        converter.register_known_landmark(
            lm.landmark_type,
            lat_deg=lm.lat_deg,
            lon_deg=lm.lon_deg,
            physical_height_m=lm.physical_height_m,
        )

    # --- Main loop ---
    log_entries    = []
    error_samples  = []
    vlm_events     = []
    solar_events   = []
    solar_fix_log  = []          # list of {t_s, gdop, lat, lon}
    solar_gdop_last = float("nan")
    next_log_t = 0.0
    log_interval_s = max(1.0, duration_s / 1800.0)

    print("Running navigation filter...", file=sys.stderr)
    for frame in frames:
        dt = frame.imu.dt_s
        scheduler.tick(dt)
        kf.set_timestamp(frame.timestamp_utc)

        kf.predict(frame.imu)

        if scheduler.airspeed_due:
            kf.update_velocity_from_heading(cfg.airspeed_ms, sigma_ms=2.0)
        if frame.baro and scheduler.baro_due:
            kf.update_barometer(frame.baro)
        if frame.mag and scheduler.mag_due:
            kf.update_magnetometer(frame.mag)
        if use_vlm and frame.vlm_obs is not None:
            n = converter.process(frame.vlm_obs, kf)
            if n > 0:
                vlm_events.append(frame.t_s)

        # Solar fix (daytime celestial aiding)
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
                solar_gdop_last = fix.position_gdop
                solar_fix_log.append({
                    "t_s": round(frame.t_s, 1),
                    "gdop": round(fix.position_gdop, 4),
                    "lat": round(fix.lat_deg, 6),
                    "lon": round(fix.lon_deg, 6),
                })

        state = kf.state
        err_m = _haversine_m(state.lat_deg, state.lon_deg,
                             frame.truth.lat_deg, frame.truth.lon_deg)
        error_samples.append(err_m)

        if frame.t_s >= next_log_t:
            from uav_nav.solar.ephemeris import solar_position
            sun = solar_position(state.lat_deg, state.lon_deg, frame.timestamp_utc)
            speed_ms = math.sqrt(state.v_north_ms ** 2 + state.v_east_ms ** 2)
            # Solar position accuracy: GDOP × σ_obs(0.2°) × 111 km/°
            sol_acc_km = (solar_gdop_last * 0.2 * 111.0
                          if not math.isnan(solar_gdop_last) else float("nan"))
            log_entries.append({
                "t_s": round(frame.t_s, 1),
                "utc_time": frame.timestamp_utc.strftime("%H:%M:%S UTC"),
                "edt_time": (frame.timestamp_utc + timedelta(hours=-4)).strftime("%H:%M EDT"),
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
                "solar_gdop": round(solar_gdop_last, 4) if not math.isnan(solar_gdop_last) else None,
                "solar_accuracy_km": round(sol_acc_km, 2) if not math.isnan(sol_acc_km) else None,
            })
            next_log_t += log_interval_s

    # --- Stats ---
    final_state = kf.state
    last_truth  = frames[-1].truth
    final_err   = _haversine_m(final_state.lat_deg, final_state.lon_deg,
                               last_truth.lat_deg, last_truth.lon_deg)
    errors_sorted = sorted(error_samples)
    p95      = errors_sorted[int(len(errors_sorted) * 0.95)]
    mean_err = sum(error_samples) / len(error_samples)
    passed   = final_err < PASS_THRESHOLD_M

    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"\nResults — Daytime Solar ({mode_label}):", file=sys.stderr)
    print(f"  Final error:   {final_err:.0f} m  [{status}]", file=sys.stderr)
    print(f"  P95 error:     {p95:.0f} m", file=sys.stderr)
    print(f"  Mean error:    {mean_err:.0f} m", file=sys.stderr)
    print(f"  VLM updates:   {len(vlm_events)}", file=sys.stderr)
    print(f"  Solar fixes:   {len(solar_events)}", file=sys.stderr)
    print(f"  NEES alarms:   {kf.nees_alarm_count}", file=sys.stderr)

    import os
    _est = timezone(timedelta(hours=-5))
    _ts  = datetime.now(_est).strftime("%Y%m%d-%H%M%SEST")
    _aids = ("solar_vlm" if (use_solar and use_vlm)
             else "solar_novlm" if use_solar
             else "vlm_nosolar" if use_vlm
             else "dead_reckoning")
    _stem = f"scenario_daytime_solar_{_aids}"
    plot_path  = os.path.join(output_dir, f"{_stem}_{_ts}.png")
    video_path = os.path.join(output_dir, f"{_stem}_{_ts}.mp4")

    result = {
        "scenario_id": cfg.scenario_id,
        "label": cfg.label + (f" (seed={seed}, {mode_label})"),
        "duration_s": duration_s,
        "dist_km": dist_m / 1000.0,
        "final_error_m": final_err,
        "p95_error_m": p95,
        "mean_error_m": mean_err,
        "vlm_updates": len(vlm_events),
        "celestial_updates": 0,
        "solar_updates": len(solar_events),
        "nees_alarm_count": kf.nees_alarm_count,
        "pass_threshold_m": PASS_THRESHOLD_M,
        "navigation_mode": "daytime",
        "passed": passed,
        "plot_path": plot_path,
        "video_path": video_path,
        "log_entries": log_entries,
        "error_samples": error_samples,
        "vlm_events": vlm_events,
        "celestial_events": solar_events,   # nav_viz compat key
        "solar_events": solar_events,
        "solar_fix_log": solar_fix_log,
        "solar_gdop_mean": (sum(f["gdop"] for f in solar_fix_log) / len(solar_fix_log)
                            if solar_fix_log else None),
        "solar_gdop_best": (min(f["gdop"] for f in solar_fix_log)
                            if solar_fix_log else None),
        "t_start_utc": cfg.t_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "landmarks": [
            {"name": lm.name, "lat_deg": lm.lat_deg,
             "lon_deg": lm.lon_deg, "visible_range_km": lm.visible_range_km}
            for lm in cfg.landmarks
        ],
        "origin_lat": cfg.origin_lat,
        "origin_lon": cfg.origin_lon,
        "dest_lat": cfg.dest_lat,
        "dest_lon": cfg.dest_lon,
        "airspeed_ms": cfg.airspeed_ms,
        "seed": seed,
    }

    try:
        from nav_viz import generate_flight_png, generate_flight_video
        generate_flight_png(result, plot_path)
        generate_flight_video(result, video_path)
    except ImportError:
        pass   # nav_viz optional; no plot needed for programmatic use

    return result


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(
        description="Run the daytime solar navigation scenario."
    )
    parser.add_argument("--seed", type=int, default=501,
                        help="RNG seed (default: 501)")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for plots/video (default: .)")
    args = parser.parse_args()

    # Run with solar fixes
    result_solar = run_daytime_scenario(
        seed=args.seed, use_solar=True, output_dir=args.output_dir
    )

    # Run without solar fixes (dead reckoning + VLM only) for comparison
    result_imu = run_daytime_scenario(
        seed=args.seed, use_solar=False, output_dir=args.output_dir
    )

    print("\n" + "=" * 60, file=sys.stderr)
    print("COMPARISON: Solar-Aided vs Dead Reckoning+VLM", file=sys.stderr)
    print(f"  Solar fix final error : {result_solar['final_error_m']:.0f} m  "
          f"({'PASS' if result_solar['passed'] else 'FAIL'})", file=sys.stderr)
    print(f"  IMU+VLM final error   : {result_imu['final_error_m']:.0f} m  "
          f"({'PASS' if result_imu['passed'] else 'FAIL'})", file=sys.stderr)
    improvement = result_imu["final_error_m"] - result_solar["final_error_m"]
    print(f"  Solar improvement     : {improvement:+.0f} m", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    print(json.dumps(
        {k: v for k, v in result_solar.items()
         if k not in ("log_entries", "error_samples")},
        indent=2,
    ))
