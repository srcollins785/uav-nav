#!/usr/bin/env python3
"""
tools/sea_pdx_statistical_test.py

R16 — Paired statistical test for SEA→PDX daytime vs nighttime.

The 8/10 (nighttime) vs 10/10 (daytime) headline difference at 10 seeds
is not statistically powered.  Run 30 seeds to provide a proper test and
95% CI on the mean difference.

Statistical tests:
  - Paired t-test (day vs night, same seed)
  - Wilcoxon signed-rank
  - 95% CI on mean difference

Results: results/sea_pdx_statistical_test.json
"""
from __future__ import annotations

import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_nav.demo.scenario_configs import SCENARIO_SEA_PDX

N_SEEDS   = 30
PASS_M    = 50.0
OUTPUT_JSON = PROJECT_ROOT / "results" / "sea_pdx_statistical_test.json"

T_DAYTIME_UTC   = datetime(2026, 7, 15, 20, 0, 0, tzinfo=timezone.utc)  # 13:00 PDT — good solar elevation at 47°N
T_NIGHTTIME_UTC = datetime(2026, 7, 16,  7, 0, 0, tzinfo=timezone.utc)  # 00:00 PDT — clear sky nighttime


def _run_one(cfg, seed: int) -> float:
    import math as _math
    from uav_nav.demo.generic_scenario import (
        _haversine_m as _hav,
        _compute_heading as _hdg,
        generate_scenario_from_config,
    )
    from uav_nav.celestial.solver import get_celestial_fix_warmstart
    from uav_nav.solar.solver import get_solar_fix_warmstart
    from uav_nav.fusion.constraint_converter import ConstraintConverter
    from uav_nav.fusion.sensor_scheduler import SensorScheduler
    from uav_nav.navigation.state import UAVState
    from uav_nav.navigation.ukf import UAVKalmanFilter

    cfg      = copy.deepcopy(cfg)
    cfg.seed = seed

    frames  = generate_scenario_from_config(cfg)
    heading = _hdg(cfg.origin_lat, cfg.origin_lon, cfg.dest_lat, cfg.dest_lon)
    hr      = _math.radians(heading)
    state0  = UAVState(
        lat_deg=cfg.origin_lat, lon_deg=cfg.origin_lon, alt_m=cfg.origin_alt_m,
        v_north_ms=cfg.airspeed_ms * _math.cos(hr),
        v_east_ms=cfg.airspeed_ms  * _math.sin(hr),
        v_down_ms=0.0, heading_deg=heading,
        mag_bias_deg=cfg.mag_anomaly_deg,
    )
    kf        = UAVKalmanFilter(state0)
    converter = ConstraintConverter()
    scheduler = SensorScheduler()
    for lm in cfg.landmarks:
        converter.register_known_landmark(
            lm.landmark_type,
            lat_deg=lm.lat_deg,
            lon_deg=lm.lon_deg,
            physical_height_m=lm.physical_height_m,
        )

    use_solar = (cfg.navigation_mode == "daytime")

    for frame in frames:
        kf.set_timestamp(frame.timestamp_utc)
        kf.predict(frame.imu)
        scheduler.tick(frame.imu.dt_s)
        if scheduler.airspeed_due:
            kf.update_velocity_from_heading(cfg.airspeed_ms, sigma_ms=2.0)
        if frame.baro and scheduler.baro_due:
            kf.update_barometer(frame.baro)
        if frame.mag and scheduler.mag_due:
            kf.update_magnetometer(frame.mag)
        if frame.vlm_obs is not None:
            converter.process(frame.vlm_obs, kf)

        if use_solar and frame.solar_obs is not None:
            fix = get_solar_fix_warmstart(
                frame.solar_obs, frame.timestamp_utc,
                init_lat_deg=kf.state.lat_deg,
                init_lon_deg=kf.state.lon_deg,
            )
            if fix is not None and fix.optimizer_success and fix.rmse_normalized < 3.0:
                kf.update_solar_fix(fix)
        elif not use_solar and frame.celestial_obs:
            fix = get_celestial_fix_warmstart(
                frame.celestial_obs, frame.timestamp_utc,
                init_lat_deg=kf.state.lat_deg,
                init_lon_deg=kf.state.lon_deg,
            )
            if fix is not None and fix.optimizer_success and fix.rmse_normalized < 3.0:
                kf.update_celestial_fix(fix)

    last = frames[-1].truth
    st   = kf.state
    return _hav(st.lat_deg, st.lon_deg, last.lat_deg, last.lon_deg)


def main():
    day_errs   = []
    night_errs = []

    print(f"SEA→PDX paired test — {N_SEEDS} seeds, PASS < {PASS_M} m\n")
    print(f"  {'Seed':<6} {'Day (m)':>10} {'Night (m)':>10} {'Diff (m)':>10} {'Day<50':>7} {'Night<50':>8}")
    print("  " + "-" * 55)

    for seed in range(1, N_SEEDS + 1):
        day_cfg = copy.deepcopy(SCENARIO_SEA_PDX)
        day_cfg.navigation_mode = "daytime"
        day_cfg.t_start_utc     = T_DAYTIME_UTC

        night_cfg = copy.deepcopy(SCENARIO_SEA_PDX)
        night_cfg.navigation_mode = "nighttime"
        night_cfg.t_start_utc     = T_NIGHTTIME_UTC

        try:
            day_err   = _run_one(day_cfg,   seed)
            night_err = _run_one(night_cfg, seed)
            day_errs.append(day_err)
            night_errs.append(night_err)
            print(f"  {seed:<6} {day_err:>10.1f} {night_err:>10.1f} "
                  f"{day_err-night_err:>+10.1f} "
                  f"{'PASS':>7} " if day_err < PASS_M else f"  {seed:<6} {day_err:>10.1f} {night_err:>10.1f} "
                  f"{day_err-night_err:>+10.1f} {'FAIL':>7} ", end="")
            print(f"{'PASS':>8}" if night_err < PASS_M else f"{'FAIL':>8}")
        except Exception as exc:
            print(f"  {seed:<6} FAILED: {exc}")

    if not day_errs:
        print("No results!", file=sys.stderr)
        return

    n = len(day_errs)
    day_arr   = np.array(day_errs)
    night_arr = np.array(night_errs)
    diffs     = day_arr - night_arr

    mean_day   = float(np.mean(day_arr))
    mean_night = float(np.mean(night_arr))
    mean_diff  = float(np.mean(diffs))
    std_diff   = float(np.std(diffs, ddof=1))
    se_diff    = std_diff / math.sqrt(n)
    t_crit     = float(stats.t.ppf(0.975, df=n - 1))
    ci_lo      = mean_diff - t_crit * se_diff
    ci_hi      = mean_diff + t_crit * se_diff

    t_stat, p_ttest   = stats.ttest_rel(day_arr, night_arr)
    w_stat, p_wilcox  = stats.wilcoxon(diffs)

    day_pass   = int(np.sum(day_arr   < PASS_M))
    night_pass = int(np.sum(night_arr < PASS_M))

    print(f"\n{'='*55}")
    print("SEA→PDX Daytime vs Nighttime — Statistical Summary")
    print(f"{'='*55}")
    print(f"  n = {n} seeds")
    print(f"  Daytime   mean={mean_day:.1f} m  pass={day_pass}/{n}")
    print(f"  Nighttime mean={mean_night:.1f} m  pass={night_pass}/{n}")
    print(f"  Mean diff (day−night): {mean_diff:+.1f} m")
    print(f"  95% CI: [{ci_lo:+.1f}, {ci_hi:+.1f}] m")
    print(f"  Paired t-test:  t={t_stat:.3f}  p={p_ttest:.4f}")
    print(f"  Wilcoxon:       W={w_stat:.0f}  p={p_wilcox:.4f}")

    result = {
        "scenario": "SEA→PDX",
        "n_seeds": n,
        "pass_threshold_m": PASS_M,
        "daytime": {
            "mean_m": mean_day,
            "pass_count": day_pass,
            "pass_rate": day_pass / n,
            "errors": [round(e, 1) for e in day_errs],
        },
        "nighttime": {
            "mean_m": mean_night,
            "pass_count": night_pass,
            "pass_rate": night_pass / n,
            "errors": [round(e, 1) for e in night_errs],
        },
        "mean_diff_m": round(mean_diff, 2),
        "ci_95_lo": round(ci_lo, 2),
        "ci_95_hi": round(ci_hi, 2),
        "t_stat": round(float(t_stat), 4),
        "p_ttest": round(float(p_ttest), 5),
        "wilcoxon_w": round(float(w_stat), 1),
        "p_wilcox": round(float(p_wilcox), 5),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
