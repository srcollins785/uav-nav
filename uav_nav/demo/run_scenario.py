"""
Generic scenario runner.

Runs any ScenarioConfig through the full navigation pipeline and
produces a standalone trajectory + error plot saved as a PNG.

Usage:
    python -m uav_nav.demo.run_scenario --scenario baseline
    python -m uav_nav.demo.run_scenario --scenario very_short
    python -m uav_nav.demo.run_scenario --scenario short
    python -m uav_nav.demo.run_scenario --scenario longer
    python -m uav_nav.demo.run_scenario --scenario much_longer

Returns the final position error (m) as an integer exit code proxy
(printed to stderr summary).
"""
from __future__ import annotations

import json
import logging
import math
import sys
from typing import List

import numpy as np

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def run_scenario(
    cfg,
    output_dir: str = ".",
    use_solar: bool = False,
    use_vlm: bool = True,
    use_celestial: bool = True,
    use_barometer: bool = True,
    use_magnetometer: bool = True,
    use_airspeed: bool = True,
    make_plots: bool = True,
) -> dict:
    """
    Run one ScenarioConfig end-to-end.

    Returns a result dict with keys:
        scenario_id, label, duration_s, dist_km,
        final_error_m, p95_error_m, mean_error_m,
        vlm_updates, passed, plot_path
    """
    from uav_nav.celestial.solver import get_celestial_fix_warmstart
    from uav_nav.solar.solver import get_solar_fix_warmstart
    from uav_nav.demo.generic_scenario import (
        generate_scenario_from_config, _haversine_m as hav,
    )
    from uav_nav.demo.generic_scenario import _compute_heading
    from uav_nav.fusion.constraint_converter import ConstraintConverter
    from uav_nav.fusion.sensor_scheduler import SensorScheduler
    from uav_nav.navigation.state import UAVState
    from uav_nav.navigation.ukf import UAVKalmanFilter

    dist_m = _haversine_m(cfg.origin_lat, cfg.origin_lon,
                          cfg.dest_lat, cfg.dest_lon)
    duration_s = dist_m / cfg.airspeed_ms

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Scenario: {cfg.label}", file=sys.stderr)
    print(f"  Route:    ({cfg.origin_lat:.3f}°N, {cfg.origin_lon:.3f}°W) → "
          f"({cfg.dest_lat:.3f}°N, {cfg.dest_lon:.3f}°W)", file=sys.stderr)
    print(f"  Distance: {dist_m/1000:.1f} km  |  Duration: {duration_s/60:.1f} min",
          file=sys.stderr)
    print(f"  Landmarks: {len(cfg.landmarks)}", file=sys.stderr)
    print(f"Generating scenario frames...", file=sys.stderr)

    frames = generate_scenario_from_config(cfg)
    print(f"  {len(frames):,} IMU frames generated.", file=sys.stderr)

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

    # Register all landmarks
    for lm in cfg.landmarks:
        converter.register_known_landmark(
            lm.landmark_type,
            lat_deg=lm.lat_deg,
            lon_deg=lm.lon_deg,
            physical_height_m=lm.physical_height_m,
        )

    # --- Main loop ---
    log_entries = []
    error_samples = []
    true_nees_2d_samples = []
    vlm_events = []
    celestial_events = []
    solar_events = []
    solar_fix_log = []
    solar_gdop_last = float("nan")
    next_log_t = 0.0
    # R8: celestial/solar convergence counters
    cel_attempts = cel_solver_fail = cel_gate_reject = 0
    sol_attempts = sol_solver_fail = sol_gate_reject = 0
    log_interval_s = max(1.0, duration_s / 1800.0)  # ~1800 log points max

    print("Running navigation filter...", file=sys.stderr)
    for frame in frames:
        dt = frame.imu.dt_s
        scheduler.tick(dt)
        kf.set_timestamp(frame.timestamp_utc)

        kf.predict(frame.imu)

        if use_airspeed and scheduler.airspeed_due:
            kf.update_velocity_from_heading(cfg.airspeed_ms, sigma_ms=2.0)
        if use_barometer and frame.baro and scheduler.baro_due:
            kf.update_barometer(frame.baro)
        if use_magnetometer and frame.mag and scheduler.mag_due:
            kf.update_magnetometer(frame.mag)
        if use_vlm and frame.vlm_obs is not None:
            n = converter.process(frame.vlm_obs, kf)
            if n > 0:
                vlm_events.append(frame.t_s)

        if use_celestial and frame.celestial_obs and not use_solar:
            cel_attempts += 1
            _s = kf.state
            fix = get_celestial_fix_warmstart(
                frame.celestial_obs, frame.timestamp_utc,
                init_lat_deg=_s.lat_deg,
                init_lon_deg=_s.lon_deg,
            )
            if fix is None or not fix.optimizer_success:
                cel_solver_fail += 1
            elif fix.rmse_normalized >= 3.0:
                cel_gate_reject += 1
            else:
                kf.update_celestial_fix(fix)
                celestial_events.append(frame.t_s)

        if use_solar and frame.solar_obs is not None:
            sol_attempts += 1
            _s = kf.state
            fix = get_solar_fix_warmstart(
                frame.solar_obs, frame.timestamp_utc,
                init_lat_deg=_s.lat_deg,
                init_lon_deg=_s.lon_deg,
            )
            if fix is None or not fix.optimizer_success:
                sol_solver_fail += 1
            elif fix.rmse_normalized >= 3.0:
                sol_gate_reject += 1
            else:
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

        # True 2D position NEES — expected value = 2.0 for a well-calibrated filter
        _d = np.array([frame.truth.lat_deg - state.lat_deg,
                       frame.truth.lon_deg - state.lon_deg])
        try:
            true_nees_2d_samples.append(float(_d @ np.linalg.solve(kf._ukf.P[0:2, 0:2], _d)))
        except np.linalg.LinAlgError:
            pass

        if frame.t_s >= next_log_t:
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
            })
            next_log_t += log_interval_s

    # --- Stats ---
    final_state = kf.state
    last_truth = frames[-1].truth
    final_err = _haversine_m(final_state.lat_deg, final_state.lon_deg,
                             last_truth.lat_deg, last_truth.lon_deg)
    from uav_nav.config import get_config
    import statistics as _stats
    _thresh = get_config().pass_threshold_m
    errors_sorted = sorted(error_samples)
    n_err = len(errors_sorted)
    p90 = errors_sorted[int(n_err * 0.90)]
    p95 = errors_sorted[int(n_err * 0.95)]
    mean_err = sum(error_samples) / n_err
    median_err = _stats.median(error_samples)
    max_err = max(error_samples)
    passed = final_err < _thresh
    true_nees_2d_max  = max(true_nees_2d_samples)  if true_nees_2d_samples else float("nan")
    true_nees_2d_mean = (sum(true_nees_2d_samples) / len(true_nees_2d_samples)
                         if true_nees_2d_samples else float("nan"))
    # R7: gyro bias observability — true bias in simulation is 0.005 dps (constant drift)
    _TRUE_GYRO_BIAS_DPS = 0.005
    final_gyro_bias_err_dps = final_state.gyro_bias_z_dps - _TRUE_GYRO_BIAS_DPS

    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"\nResults — {cfg.label}:", file=sys.stderr)
    print(f"  Final error:  {final_err:.0f} m  [{status}]", file=sys.stderr)
    print(f"  P95 error:    {p95:.0f} m", file=sys.stderr)
    print(f"  Mean error:   {mean_err:.0f} m", file=sys.stderr)
    print(f"  VLM updates:  {len(vlm_events)}", file=sys.stderr)
    if use_solar:
        print(f"  Solar fixes:  {len(solar_events)}", file=sys.stderr)
    else:
        print(f"  Cel updates:  {len(celestial_events)}", file=sys.stderr)
    print(f"  NIS alarms:   {kf.nees_alarm_count}", file=sys.stderr)
    print(f"  TrueNEES-2D:  max={true_nees_2d_max:.1f}  mean={true_nees_2d_mean:.1f}  (expected~2.0)", file=sys.stderr)

    # --- Build result dict (consumed by nav_viz and callers) ---
    import os
    from datetime import datetime, timezone, timedelta
    _est = timezone(timedelta(hours=-5))
    _ts = datetime.now(_est).strftime("%Y%m%d-%H%M%SEST")
    _mode = getattr(cfg, "navigation_mode", "nighttime")
    _stem = f"scenario_{cfg.scenario_id}_{_mode}"
    plot_path  = os.path.join(output_dir, f"{_stem}_{_ts}.png")
    video_path = os.path.join(output_dir, f"{_stem}_{_ts}.mp4")

    # Landmarks as plain dicts (nav_viz is decoupled from uav_nav dataclasses)
    landmarks_dicts = [
        {
            "name": lm.name,
            "lat_deg": lm.lat_deg,
            "lon_deg": lm.lon_deg,
            "visible_range_km": lm.visible_range_km,
        }
        for lm in cfg.landmarks
    ]

    result = {
        # --- identification ---
        "scenario_id": cfg.scenario_id,
        "label": cfg.label,
        "navigation_mode": _mode,
        "seed": getattr(cfg, "seed", 42),
        "t_start_utc": cfg.t_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        # --- route ---
        "duration_s": duration_s,
        "dist_km": dist_m / 1000.0,
        "origin_lat": cfg.origin_lat,
        "origin_lon": cfg.origin_lon,
        "dest_lat": cfg.dest_lat,
        "dest_lon": cfg.dest_lon,
        "airspeed_ms": cfg.airspeed_ms,
        "n_landmarks": len(cfg.landmarks),
        # --- error metrics ---
        "final_error_m": final_err,
        "max_error_m": max_err,
        "mean_error_m": mean_err,
        "median_error_m": median_err,
        "p90_error_m": p90,
        "p95_error_m": p95,
        # --- pass/fail at multiple thresholds ---
        "pass_10m": final_err < 10.0,
        "pass_25m": final_err < 25.0,
        "pass_50m": final_err < 50.0,
        "pass_100m": final_err < 100.0,
        "passed": passed,           # alias for pass_50m
        "pass_threshold_m": _thresh,
        # --- sensor flags ---
        "imu_enabled": True,
        "barometer_enabled": use_barometer,
        "magnetometer_enabled": use_magnetometer,
        "airspeed_enabled": use_airspeed,
        "vlm_enabled": use_vlm,
        "celestial_enabled": use_celestial and not use_solar,
        "solar_enabled": use_solar,
        # --- update counts ---
        "vlm_updates": len(vlm_events),
        "celestial_updates": len(celestial_events),
        "solar_updates": len(solar_events),
        "nees_alarm_count": kf.nees_alarm_count,   # NIS-based divergence monitor (key kept for compat)
        # --- R8: solver convergence ---
        "cel_attempts": cel_attempts,
        "cel_solver_fail": cel_solver_fail,
        "cel_gate_reject": cel_gate_reject,
        "sol_attempts": sol_attempts,
        "sol_solver_fail": sol_solver_fail,
        "sol_gate_reject": sol_gate_reject,
        "true_nees_2d_max": true_nees_2d_max,      # true position NEES; expected ~2.0
        "true_nees_2d_mean": true_nees_2d_mean,
        # --- R7: gyro bias observability ---
        "final_gyro_bias_est_dps": final_state.gyro_bias_z_dps,
        "final_gyro_bias_true_dps": _TRUE_GYRO_BIAS_DPS,
        "final_gyro_bias_err_dps": final_gyro_bias_err_dps,
        # --- VLM gating breakdown (from ConstraintConverter) ---
        **converter.vlm_stats,
        # --- solar detail ---
        "solar_fix_log": solar_fix_log,
        "solar_gdop_mean": (sum(f["gdop"] for f in solar_fix_log) / len(solar_fix_log)
                            if solar_fix_log else None),
        "solar_gdop_best": (min(f["gdop"] for f in solar_fix_log)
                            if solar_fix_log else None),
        # --- full trajectory data (used by nav_viz) ---
        "log_entries": log_entries,
        "error_samples": error_samples,
        "vlm_events": vlm_events,
        "celestial_events": celestial_events,
        "landmarks": landmarks_dicts,
        "plot_path": plot_path,
        "video_path": video_path,
    }

    # --- Visualize ---
    # make_plots=False skips PNG/MP4 rendering. Rendering happens after the
    # filter has run and consumes no randomness, so metrics are identical
    # either way; skipping it just makes multi-seed verification fast.
    if make_plots:
        try:
            from nav_viz import generate_flight_png, generate_flight_video
            generate_flight_png(result, plot_path)
            generate_flight_video(result, video_path)
        except ImportError:
            print("nav_viz not available — falling back to internal plot.", file=sys.stderr)
            _plot_scenario(cfg, frames, log_entries, vlm_events, final_err, p95, plot_path)
    else:
        result["plot_path"] = None
        result["video_path"] = None

    return result


def _plot_scenario(cfg, frames, log_entries, vlm_events, final_err, p95, plot_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot.", file=sys.stderr)
        return

    true_lats = [f.truth.lat_deg for f in frames[::100]]
    true_lons = [f.truth.lon_deg for f in frames[::100]]
    est_lats  = [e["lat"] for e in log_entries]
    est_lons  = [e["lon"] for e in log_entries]
    errors    = [e["error_m"] for e in log_entries]
    times     = [e["t_s"] / 60.0 for e in log_entries]   # minutes

    from uav_nav.config import get_config
    _thresh = get_config().pass_threshold_m
    dist_km = _haversine_m(cfg.origin_lat, cfg.origin_lon,
                           cfg.dest_lat, cfg.dest_lon) / 1000.0
    duration_min = dist_km * 1000 / cfg.airspeed_ms / 60.0
    passed = final_err < _thresh
    status_color = "#2ca02c" if passed else "#d62728"

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#f8f9fa")

    # ---------- left: trajectory map ----------
    ax_map = fig.add_axes([0.05, 0.12, 0.52, 0.78])
    ax_map.set_facecolor("#eaf0fb")

    ax_map.plot(true_lons, true_lats, color="#2ca02c", lw=2.5,
                label="Ground truth", alpha=0.8, zorder=3)
    ax_map.plot(est_lons, est_lats, color="#1f77b4", lw=1.8, ls="--",
                label="UKF estimate", zorder=4)

    # VLM update scatter
    vlm_set = {round(t, 1) for t in vlm_events}
    vlm_pts = [e for e in log_entries if e["t_s"] in vlm_set]
    if vlm_pts:
        ax_map.scatter([e["lon"] for e in vlm_pts],
                       [e["lat"] for e in vlm_pts],
                       s=35, c="#ff7f0e", zorder=5, label="VLM update", alpha=0.8)

    # Landmarks
    for lm in cfg.landmarks:
        ax_map.plot(lm.lon_deg, lm.lat_deg, marker="D", ms=9,
                    color="#9467bd", zorder=6)
        ax_map.annotate(lm.name, xy=(lm.lon_deg, lm.lat_deg),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=6.5, color="#5c3d8f")

    # Origin / destination
    ax_map.plot(cfg.origin_lon, cfg.origin_lat, "gs", ms=11,
                label="Origin", zorder=7)
    ax_map.plot(cfg.dest_lon, cfg.dest_lat, "r^", ms=11,
                label="Destination", zorder=7)

    ax_map.set_xlabel("Longitude (°)", fontsize=10)
    ax_map.set_ylabel("Latitude (°)", fontsize=10)
    ax_map.set_title(f"Trajectory: {cfg.label}\n"
                     f"{cfg.origin_lat:.3f}°N → {cfg.dest_lat:.3f}°N  "
                     f"| {dist_km:.0f} km | {duration_min:.0f} min",
                     fontsize=11, fontweight="bold")
    ax_map.legend(fontsize=8, loc="best")
    ax_map.grid(True, alpha=0.3, lw=0.5)

    # ---------- right: error vs time ----------
    ax_err = fig.add_axes([0.62, 0.55, 0.35, 0.35])
    ax_err.set_facecolor("#f0f4ff")
    ax_err.plot(times, errors, color="#1f77b4", lw=1.5)
    ax_err.axhline(_thresh, color="#d62728", ls=":", lw=1.5,
                   label=f"{_thresh:.0f} m target")
    ax_err.fill_between(times, errors, alpha=0.15, color="#1f77b4")
    if vlm_pts:
        ax_err.axvspan(
            min(e["t_s"] for e in vlm_pts) / 60.0,
            max(e["t_s"] for e in vlm_pts) / 60.0,
            alpha=0.12, color="#ff7f0e", label="VLM active"
        )
    ax_err.set_xlabel("Time (min)", fontsize=9)
    ax_err.set_ylabel("Horiz. error (m)", fontsize=9)
    ax_err.set_title("Position Error vs Time", fontsize=9, fontweight="bold")
    ax_err.legend(fontsize=7)
    ax_err.grid(True, alpha=0.3, lw=0.5)
    ax_err.set_xlim(0, duration_min)
    ax_err.set_ylim(0, max(max(errors) * 1.15, _thresh * 1.1))

    # ---------- bottom-right: stats box ----------
    ax_stats = fig.add_axes([0.62, 0.12, 0.35, 0.35])
    ax_stats.set_facecolor("#f0f4ff")
    ax_stats.axis("off")

    stats_text = (
        f"{'PASS' if passed else 'FAIL'} — "
        f"{'< ' + str(int(_thresh)) + ' m criterion met' if passed else '>= ' + str(int(_thresh)) + ' m criterion exceeded'}\n\n"
        f"  Final error :  {final_err:>7.0f} m\n"
        f"  P95 error   :  {p95:>7.0f} m\n"
        f"  VLM updates :  {len(vlm_pts):>7d}\n"
        f"  Landmarks   :  {len(cfg.landmarks):>7d}\n"
        f"  Distance    :  {dist_km:>7.1f} km\n"
        f"  Duration    :  {duration_min:>7.1f} min\n"
        f"  Speed       :  {cfg.airspeed_ms:>7.1f} m/s\n"
    )
    ax_stats.text(0.05, 0.95, stats_text,
                  transform=ax_stats.transAxes,
                  fontsize=9, va="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5",
                            facecolor="white", edgecolor=status_color, lw=2))
    ax_stats.set_title("Navigation Statistics", fontsize=9, fontweight="bold")

    # ---------- super-title ----------
    fig.suptitle(
        f"uav_nav Offline Navigation — {cfg.label}  "
        f"({'PASS' if passed else 'FAIL'}: {final_err:.0f} m)",
        fontsize=13, fontweight="bold", color=status_color, y=0.98
    )

    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {plot_path}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    from uav_nav.demo.scenario_configs import ALL_SCENARIOS

    choices = {s.scenario_id: s for s in ALL_SCENARIOS}

    parser = argparse.ArgumentParser(description="Run a single named scenario.")
    parser.add_argument("--scenario", required=True, choices=list(choices),
                        help="Scenario ID to run.")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write plot PNG (default: .)")
    args = parser.parse_args()

    result = run_scenario(choices[args.scenario], output_dir=args.output_dir)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("log_entries", "error_samples")}, indent=2))
