"""
flight_plot.py — Static 3-panel PNG generator for UAV navigation results.

Inputs only the result dict from uav_nav.demo.run_scenario.run_scenario().
No imports from uav_nav.
"""
from __future__ import annotations

import sys

PASS_THRESHOLD_M = 50.0


def generate_flight_png(result: dict, output_path: str) -> str:
    """
    Render a 3-panel PNG summarising a completed scenario run.

    Panels:
      Left  — trajectory map (ground truth vs UKF estimate)
      Top-right  — position error vs time
      Bottom-right — statistics box

    Parameters
    ----------
    result : dict
        As returned by run_scenario(). Required keys:
          label, dist_km, duration_s, airspeed_ms (from cfg),
          log_entries, vlm_events, celestial_events,
          final_error_m, p95_error_m, passed,
          landmarks (list of dicts with lat_deg, lon_deg, name),
          origin_lat, origin_lon, dest_lat, dest_lon
    output_path : str
        File path for the PNG output.

    Returns
    -------
    str — output_path on success, empty string on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping PNG.", file=sys.stderr)
        return ""

    log_entries = result["log_entries"]
    vlm_events = result.get("vlm_events", [])
    celestial_events = result.get("celestial_events", [])
    final_err = result["final_error_m"]
    p95 = result["p95_error_m"]
    passed = result["passed"]
    label = result["label"]
    dist_km = result["dist_km"]
    duration_min = result["duration_s"] / 60.0
    landmarks = result.get("landmarks", [])
    origin_lat = result["origin_lat"]
    origin_lon = result["origin_lon"]
    dest_lat = result["dest_lat"]
    dest_lon = result["dest_lon"]
    vlm_count  = result.get("vlm_updates", 0)
    cel_count  = result.get("celestial_updates", 0)
    solar_count = result.get("solar_updates", cel_count)
    t_start    = result.get("t_start_utc", "")
    gdop_mean  = result.get("solar_gdop_mean")
    gdop_best  = result.get("solar_gdop_best")

    est_lats = [e["lat"] for e in log_entries]
    est_lons = [e["lon"] for e in log_entries]
    errors   = [e["error_m"] for e in log_entries]
    times    = [e["t_s"] / 60.0 for e in log_entries]

    # Ground truth — interpolate from log entries' true path bounds
    true_lats = [origin_lat] + [e.get("true_lat", e["lat"]) for e in log_entries] + [dest_lat]
    true_lons = [origin_lon] + [e.get("true_lon", e["lon"]) for e in log_entries] + [dest_lon]
    # Prefer dedicated ground_truth list if present
    if "ground_truth" in result:
        true_lats = [p["lat"] for p in result["ground_truth"]]
        true_lons = [p["lon"] for p in result["ground_truth"]]

    status_color = "#2ca02c" if passed else "#d62728"

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#f8f9fa")

    # ── left: trajectory map ──────────────────────────────────
    ax_map = fig.add_axes([0.05, 0.12, 0.52, 0.78])
    ax_map.set_facecolor("#eaf0fb")

    ax_map.plot(true_lons, true_lats, color="#2ca02c", lw=2.5,
                label="Ground truth", alpha=0.8, zorder=3)
    ax_map.plot(est_lons, est_lats, color="#1f77b4", lw=1.8, ls="--",
                label="UKF estimate", zorder=4)

    # VLM events
    vlm_set = {round(t, 1) for t in vlm_events}
    vlm_pts = [e for e in log_entries if e["t_s"] in vlm_set]
    if vlm_pts:
        ax_map.scatter([e["lon"] for e in vlm_pts],
                       [e["lat"] for e in vlm_pts],
                       s=35, c="#ff7f0e", zorder=5, label="VLM update", alpha=0.8)

    # Celestial events
    cel_set = {round(t, 1) for t in celestial_events}
    cel_pts = [e for e in log_entries if e["t_s"] in cel_set]
    if cel_pts:
        ax_map.scatter([e["lon"] for e in cel_pts],
                       [e["lat"] for e in cel_pts],
                       s=120, c="#f5c518", marker="*", zorder=6, label="Solar fix ☀", alpha=1.0)
        for k, e in enumerate(cel_pts):
            ax_map.annotate(f"☀{k+1}", xy=(e["lon"], e["lat"]),
                            xytext=(5, 3), textcoords="offset points",
                            fontsize=6, color="#c8960c", fontweight="bold")

    for lm in landmarks:
        ax_map.plot(lm["lon_deg"], lm["lat_deg"], marker="D", ms=9,
                    color="#9467bd", zorder=6)
        ax_map.annotate(lm["name"], xy=(lm["lon_deg"], lm["lat_deg"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=6.5, color="#5c3d8f")

    ax_map.plot(origin_lon, origin_lat, "gs", ms=11, label="Origin", zorder=7)
    ax_map.plot(dest_lon, dest_lat, "r^", ms=11, label="Destination", zorder=7)

    ax_map.set_xlabel("Longitude (°)", fontsize=10)
    ax_map.set_ylabel("Latitude (°)", fontsize=10)
    ax_map.set_title(
        f"Trajectory: {label}\n"
        f"{origin_lat:.3f}°N → {dest_lat:.3f}°N  "
        f"| {dist_km:.0f} km | {duration_min:.0f} min",
        fontsize=11, fontweight="bold"
    )
    ax_map.legend(fontsize=8, loc="best")
    ax_map.grid(True, alpha=0.3, lw=0.5)

    # ── top-right: error vs time ──────────────────────────────
    ax_err = fig.add_axes([0.62, 0.55, 0.35, 0.35])
    ax_err.set_facecolor("#f0f4ff")
    ax_err.plot(times, errors, color="#1f77b4", lw=1.5)
    _thresh = result.get("pass_threshold_m", PASS_THRESHOLD_M)
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

    # ── bottom-right: statistics ──────────────────────────────
    ax_stats = fig.add_axes([0.62, 0.12, 0.35, 0.35])
    ax_stats.set_facecolor("#f0f4ff")
    ax_stats.axis("off")

    gdop_str = ""
    if gdop_mean is not None:
        gdop_str = f"  Solar GDOP   :  {gdop_mean:.3f}° mean / {gdop_best:.3f}° best\n"
    t_start_str = f"  Start (UTC)  :  {t_start[:19].replace('T',' ')}\n" if t_start else ""

    stats_text = (
        f"{'PASS' if passed else 'FAIL'} — "
        f"{'Final error < ' + str(int(_thresh)) + ' m (PASS)' if passed else 'Final error ≥ ' + str(int(_thresh)) + ' m (FAIL)'}\n\n"
        f"{t_start_str}"
        f"  Final error  :  {final_err:>7.0f} m\n"
        f"  P95 error    :  {p95:>7.0f} m\n"
        f"  VLM updates  :  {vlm_count:>7d}\n"
        f"  Solar fixes  :  {solar_count:>7d}\n"
        f"{gdop_str}"
        f"  Landmarks    :  {len(landmarks):>7d}\n"
        f"  Distance     :  {dist_km:>7.1f} km\n"
        f"  Duration     :  {duration_min:>7.1f} min\n"
        f"  Speed        :  {result.get('airspeed_ms', 50.0):>7.1f} m/s\n"
        f"  Criterion    :  final error < {_thresh:.0f} m\n"
    )
    ax_stats.text(0.05, 0.95, stats_text,
                  transform=ax_stats.transAxes,
                  fontsize=9, va="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5",
                            facecolor="white", edgecolor=status_color, lw=2))
    ax_stats.set_title("Navigation Statistics", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"uav_nav Offline Navigation — {label}  "
        f"({'PASS' if passed else 'FAIL'}: {final_err:.0f} m)",
        fontsize=13, fontweight="bold", color=status_color, y=0.98
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  PNG saved → {output_path}", file=sys.stderr)
    return output_path
