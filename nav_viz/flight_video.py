"""
flight_video.py — Animated MP4 flight replay generator.

Produces a 1280x720 MP4 with:
  1. Opening title card  (3 s) — all scenario input parameters
  2. Flight replay       (~13 s, 400 frames) — live trajectory + UKF state
  3. Closing summary     (3 s) — final stats

Uses only matplotlib + cv2 (opencv-python-headless); no extra dependencies.
Imports nothing from uav_nav.
"""
from __future__ import annotations

import math
import sys
from typing import List

# Video parameters
_W, _H = 1280, 720
_FPS = 30
_TITLE_FRAMES = 90     # 3 s title card
_SUMMARY_FRAMES = 90   # 3 s closing card
_REPLAY_FRAMES = 400   # flight replay target frame count

PASS_THRESHOLD_M = 50.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _fig_to_bgr(fig) -> "np.ndarray":
    """Convert a matplotlib figure to a BGR numpy array for cv2."""
    import numpy as np
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(_H, _W, 4)
    # RGBA → BGR
    return arr[:, :, [2, 1, 0]].copy()


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def _new_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(_W / 100, _H / 100), dpi=100)
    fig.patch.set_facecolor("#0d1117")
    return fig, plt


# ── Title card ────────────────────────────────────────────────────────────────

def _render_title_card(result: dict) -> "np.ndarray":
    """Render the opening title card as a BGR frame."""
    fig, plt = _new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    label = result["label"]
    origin_lat = result["origin_lat"]
    origin_lon = result["origin_lon"]
    dest_lat   = result["dest_lat"]
    dest_lon   = result["dest_lon"]
    dist_km    = result["dist_km"]
    duration_min = result["duration_s"] / 60.0
    airspeed   = result.get("airspeed_ms", 50.0)
    seed       = result.get("seed", "—")
    landmarks  = result.get("landmarks", [])
    t_start    = result.get("t_start_utc", "2026-01-28 02:00 UTC")

    # Title bar
    ax.text(0.5, 0.90, "GPS-Denied Navigation Simulation",
            transform=ax.transAxes, fontsize=20, fontweight="bold",
            color="#58a6ff", ha="center", va="center")
    ax.text(0.5, 0.82, label,
            transform=ax.transAxes, fontsize=15, fontweight="bold",
            color="#ffffff", ha="center", va="center")

    # Divider line
    ax.plot([0.05, 0.95], [0.76, 0.76], transform=ax.transAxes,
            color="#30363d", lw=1.5, solid_capstyle="butt", zorder=2)

    # Two-column layout: left = route + timing, right = sensors + landmarks
    left_lines = [
        ("Origin",   f"{origin_lat:.4f}°N,  {abs(origin_lon):.4f}°W"),
        ("Dest",     f"{dest_lat:.4f}°N,  {abs(dest_lon):.4f}°W"),
        ("Distance", f"{dist_km:.1f} km"),
        ("Duration", f"~{duration_min:.0f} min"),
        ("Airspeed", f"{airspeed:.0f} m/s  ({airspeed*3.6:.0f} km/h)"),
        ("Start UTC", str(t_start)[:19]),
        ("Seed",     str(seed)),
    ]
    right_lines = [("Sensors", "IMU · Baro · Mag · VLM · Celestial")]
    right_lines += [(f"  Landmark {i+1}", lm["name"])
                    for i, lm in enumerate(landmarks)]

    y0, dy = 0.68, 0.072

    for i, (key, val) in enumerate(left_lines):
        y = y0 - i * dy
        ax.text(0.07, y, f"{key}:", transform=ax.transAxes,
                fontsize=11, color="#8b949e", va="center")
        ax.text(0.26, y, val, transform=ax.transAxes,
                fontsize=11, color="#e6edf3", va="center", fontweight="bold")

    for i, (key, val) in enumerate(right_lines):
        y = y0 - i * dy
        ax.text(0.52, y, f"{key}:", transform=ax.transAxes,
                fontsize=11, color="#8b949e", va="center")
        ax.text(0.66, y, val, transform=ax.transAxes,
                fontsize=11, color="#e6edf3", va="center", fontweight="bold")

    # Footer
    ax.text(0.5, 0.04, "Clark Atlanta University — NSF CPS Medium Proposal",
            transform=ax.transAxes, fontsize=9, color="#484f58",
            ha="center", va="center")

    frame = _fig_to_bgr(fig)
    _close(fig)
    return frame


# ── Replay frame ──────────────────────────────────────────────────────────────

def _render_replay_frame(
    result: dict,
    frame_idx: int,
    total_frames: int,
    log_entries: list,
    vlm_set: set,
    cel_set: set,
) -> "np.ndarray":
    """Render one replay frame showing flight state up to log_entries[frame_idx]."""
    import numpy as np

    fig, plt = _new_fig()

    entry = log_entries[frame_idx]
    t_s = entry["t_s"]
    duration_s = result["duration_s"]
    duration_min = duration_s / 60.0
    dist_km = result["dist_km"]
    passed = result["passed"]
    label = result["label"]
    origin_lat = result["origin_lat"]
    origin_lon = result["origin_lon"]
    dest_lat = result["dest_lat"]
    dest_lon = result["dest_lon"]
    landmarks = result.get("landmarks", [])
    airspeed_ms = result.get("airspeed_ms", 50.0)

    # Slice data up to current frame
    past = log_entries[: frame_idx + 1]
    lats = [e["lat"] for e in past]
    lons = [e["lon"] for e in past]
    errs = [e["error_m"] for e in past]
    times_min = [e["t_s"] / 60.0 for e in past]

    # Ground truth track (full route from origin to dest, straight line approx)
    n_gt = max(20, frame_idx + 2)
    gt_lats = [origin_lat + (dest_lat - origin_lat) * i / (n_gt - 1) for i in range(n_gt)]
    gt_lons = [origin_lon + (dest_lon - origin_lon) * i / (n_gt - 1) for i in range(n_gt)]
    if "ground_truth" in result:
        frac = (frame_idx + 1) / total_frames
        cutoff = int(frac * len(result["ground_truth"]))
        gt_pts = result["ground_truth"][:max(2, cutoff)]
        gt_lats = [p["lat"] for p in gt_pts]
        gt_lons = [p["lon"] for p in gt_pts]

    status_color = "#3fb950" if passed else "#f85149"

    # ── layout: left map (60 %), right panel (38 %) ──
    ax_map = fig.add_axes([0.03, 0.10, 0.57, 0.82])
    ax_map.set_facecolor("#161b22")

    # Ground truth
    ax_map.plot(gt_lons, gt_lats, color="#3fb950", lw=2.0,
                alpha=0.6, label="Ground truth", zorder=2)
    # Estimate trail
    ax_map.plot(lons, lats, color="#58a6ff", lw=2.0, label="UKF estimate", zorder=3)
    # UAV icon at current position
    ax_map.plot(lons[-1], lats[-1], marker="^", ms=12,
                color="#f0e040", zorder=6, label="UAV")

    # VLM events that have occurred
    vlm_past = [e for e in past if e["t_s"] in vlm_set]
    if vlm_past:
        ax_map.scatter([e["lon"] for e in vlm_past],
                       [e["lat"] for e in vlm_past],
                       s=40, c="#ff7f0e", zorder=5, label="VLM update", alpha=0.9)

    # Solar/celestial fix events that have occurred
    cel_past = [e for e in past if e["t_s"] in cel_set]
    if cel_past:
        ax_map.scatter([e["lon"] for e in cel_past],
                       [e["lat"] for e in cel_past],
                       s=220, c="#f5c518", marker="*", zorder=5, label="Solar fix", alpha=1.0)
        # Label each solar fix with its sequence number
        for k, e in enumerate(cel_past):
            ax_map.annotate(f"☀{k+1}", xy=(e["lon"], e["lat"]),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=7, color="#f5c518", fontweight="bold", zorder=7)

    # Landmarks
    for lm in landmarks:
        ax_map.plot(lm["lon_deg"], lm["lat_deg"], marker="D", ms=8,
                    color="#bc8cff", zorder=6)
        ax_map.annotate(lm["name"][:22], xy=(lm["lon_deg"], lm["lat_deg"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=6, color="#bc8cff")

    ax_map.plot(origin_lon, origin_lat, "s", ms=10, color="#3fb950",
                label="Origin", zorder=7)
    ax_map.plot(dest_lon, dest_lat, "^", ms=10, color="#f85149",
                label="Dest", zorder=7)

    ax_map.set_xlabel("Longitude (°)", fontsize=8, color="#8b949e")
    ax_map.set_ylabel("Latitude (°)", fontsize=8, color="#8b949e")
    ax_map.tick_params(colors="#8b949e", labelsize=7)
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#30363d")
    ax_map.set_title(label, fontsize=10, fontweight="bold", color="#e6edf3", pad=4)
    ax_map.legend(fontsize=7, loc="lower right",
                  facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    ax_map.grid(True, alpha=0.15, lw=0.4, color="#8b949e")

    # ── right panel: top = error, bottom = UKF state ──
    _thresh = result.get("pass_threshold_m", 500.0)
    ax_err = fig.add_axes([0.63, 0.52, 0.35, 0.40])
    ax_err.set_facecolor("#161b22")
    ax_err.plot(times_min, errs, color="#58a6ff", lw=1.5)
    ax_err.axhline(_thresh, color="#f85149", ls=":", lw=1.2,
                   label=f"{_thresh:.0f} m target")
    ax_err.fill_between(times_min, errs, alpha=0.2, color="#58a6ff")
    ax_err.set_xlim(0, duration_min)
    ax_err.set_ylim(0, max(max(errs) * 1.2, _thresh * 1.1) if errs else _thresh * 1.2)
    ax_err.set_xlabel("Time (min)", fontsize=8, color="#8b949e")
    ax_err.set_ylabel("Error (m)", fontsize=8, color="#8b949e")
    ax_err.set_title("Position Error", fontsize=9, fontweight="bold",
                     color="#e6edf3", pad=3)
    ax_err.tick_params(colors="#8b949e", labelsize=7)
    for spine in ax_err.spines.values():
        spine.set_edgecolor("#30363d")
    ax_err.grid(True, alpha=0.15, lw=0.4, color="#8b949e")

    # ── UKF state readout ──
    ax_state = fig.add_axes([0.63, 0.10, 0.35, 0.38])
    ax_state.set_facecolor("#161b22")
    ax_state.axis("off")
    for spine in ax_state.spines.values():
        spine.set_edgecolor("#30363d")

    current_err = entry["error_m"]
    heading = entry.get("heading_deg", float("nan"))
    speed = entry.get("speed_ms", float("nan"))
    alt = entry.get("alt_m", float("nan"))
    gyro_bias = entry.get("gyro_bias_dps", float("nan"))
    solar_elev   = entry.get("solar_elevation_deg", float("nan"))
    solar_az     = entry.get("solar_azimuth_deg", float("nan"))
    solar_gdop   = entry.get("solar_gdop", float("nan")) or float("nan")
    solar_acc_km = entry.get("solar_accuracy_km", float("nan")) or float("nan")
    utc_time     = entry.get("utc_time", "")
    edt_time     = entry.get("edt_time", "")

    _thresh = result.get("pass_threshold_m", 500.0)
    err_color = "#3fb950" if current_err < _thresh else "#f85149"
    pass_sym = "✓" if current_err < _thresh else "✗"

    if not math.isnan(solar_elev):
        if solar_elev > 6.0:
            illum_str = "☀  DAYTIME"
            illum_color = "#f5c518"
        elif solar_elev < -12.0:
            illum_str = "★  NIGHTTIME"
            illum_color = "#79c0ff"
        else:
            illum_str = "~  TWILIGHT"
            illum_color = "#e3b341"
    else:
        illum_str = "—"
        illum_color = "#8b949e"

    def _row(label, val, color="#e6edf3"):
        return (label, val, color)

    solar_elev_str = f"{solar_elev:>8.1f} °" if not math.isnan(solar_elev) else "      — °"
    solar_az_str   = f"{solar_az:>8.1f} °"   if not math.isnan(solar_az)   else "      — °"
    gdop_str       = f"{solar_gdop:>8.3f}"   if not math.isnan(solar_gdop) else "       —"
    acc_str        = f"{solar_acc_km:>7.1f} km" if not math.isnan(solar_acc_km) else "      —"
    time_str       = f"{utc_time}  ({edt_time})" if utc_time else f"{t_s:.0f} s"

    state_rows = [
        _row("Clock",      time_str),
        _row("Flt time",   f"{t_s:>7.0f} s  ({t_s/60:.1f} min)"),
        _row("Heading",    f"{heading:>9.1f} °"),
        _row("Altitude",   f"{alt:>9.1f} m MSL"),
        _row("Mode",       illum_str,       illum_color),
        _row("Solar elev", solar_elev_str,  illum_color),
        _row("Solar az",   solar_az_str,    illum_color),
        _row("GDOP",       gdop_str,        illum_color),
        _row("Sol. acc.",  acc_str,         illum_color),
        _row("Error",      f"{current_err:>7.0f} m  {pass_sym}", err_color),
    ]

    ax_state.set_title("UKF State", fontsize=9, fontweight="bold",
                       color="#e6edf3", pad=3)
    ax_state.set_xlim(0, 1)
    ax_state.set_ylim(0, 1)

    y_start, y_step = 0.92, 0.105
    for i, (lbl, val, *col) in enumerate(state_rows):
        color = col[0] if col else "#e6edf3"
        y = y_start - i * y_step
        ax_state.text(0.05, y, lbl + ":", fontsize=8.5, color="#8b949e",
                      transform=ax_state.transAxes, va="center", fontfamily="monospace")
        ax_state.text(0.48, y, val, fontsize=8.5, color=color,
                      transform=ax_state.transAxes, va="center", fontfamily="monospace")

    # ── progress bar at bottom ──
    progress = (frame_idx + 1) / total_frames
    bar_ax = fig.add_axes([0.03, 0.03, 0.94, 0.04])
    bar_ax.set_facecolor("#21262d")
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")
    bar_ax.add_patch(plt.Rectangle((0, 0.1), progress, 0.8,
                                   facecolor="#58a6ff", edgecolor="none"))
    bar_ax.text(0.5, 0.5, f"t = {t_s:.0f} s  ({progress*100:.0f}%)",
                ha="center", va="center", fontsize=8,
                color="#e6edf3", transform=bar_ax.transAxes)

    frame_img = _fig_to_bgr(fig)
    _close(fig)
    return frame_img


# ── Summary card ─────────────────────────────────────────────────────────────

def _render_summary_card(result: dict) -> "np.ndarray":
    """Render the closing summary card."""
    fig, plt = _new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    passed = result["passed"]
    final_err = result["final_error_m"]
    p95 = result["p95_error_m"]
    label = result["label"]
    dist_km = result["dist_km"]
    duration_min = result["duration_s"] / 60.0
    vlm = result.get("vlm_updates", 0)
    cel = result.get("celestial_updates", 0)
    status_color = "#3fb950" if passed else "#f85149"
    status_text = "PASS ✓" if passed else "FAIL ✗"

    ax.text(0.5, 0.88, "Flight Complete — Summary",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            color="#58a6ff", ha="center", va="center")
    ax.text(0.5, 0.79, label,
            transform=ax.transAxes, fontsize=14,
            color="#e6edf3", ha="center", va="center")
    ax.text(0.5, 0.70, status_text,
            transform=ax.transAxes, fontsize=28, fontweight="bold",
            color=status_color, ha="center", va="center")

    solar      = result.get("solar_updates", cel)
    t_start    = result.get("t_start_utc", "")[:19].replace("T", " ")
    gdop_mean  = result.get("solar_gdop_mean")
    gdop_best  = result.get("solar_gdop_best")
    gdop_disp  = (f"{gdop_mean:.3f}° mean / {gdop_best:.3f}° best"
                  if gdop_mean is not None else "—")
    rows = [
        ("Start time (UTC)",      t_start),
        ("Final position error",  f"{final_err:.0f} m"),
        ("P95 position error",    f"{p95:.0f} m"),
        ("Distance flown",        f"{dist_km:.1f} km"),
        ("Flight duration",       f"{duration_min:.0f} min"),
        ("VLM bearing updates",   str(vlm)),
        ("Solar fixes fused",     str(solar)),
        ("Solar GDOP",            gdop_disp),
        ("Success criterion",     f"Final error < {result.get('pass_threshold_m', 500.0):.0f} m  →  " + status_text),
    ]
    y0, dy = 0.56, 0.072
    for i, (k, v) in enumerate(rows):
        y = y0 - i * dy
        ax.text(0.22, y, k + ":", fontsize=12, color="#8b949e",
                transform=ax.transAxes, va="center")
        ax.text(0.62, y, v, fontsize=12, fontweight="bold", color="#e6edf3",
                transform=ax.transAxes, va="center")

    ax.text(0.5, 0.03, "Clark Atlanta University — NSF CPS Medium Proposal",
            transform=ax.transAxes, fontsize=9, color="#484f58",
            ha="center", va="center")

    frame = _fig_to_bgr(fig)
    _close(fig)
    return frame


# ── Public API ────────────────────────────────────────────────────────────────

def generate_flight_video(result: dict, output_path: str, fps: int = _FPS) -> str:
    """
    Generate an animated MP4 flight replay.

    Parameters
    ----------
    result : dict
        As returned by run_scenario(). Required keys:
          label, dist_km, duration_s, origin_lat/lon, dest_lat/lon,
          log_entries (with t_s, lat, lon, error_m, heading_deg,
          speed_ms, alt_m, gyro_bias_dps),
          vlm_events, celestial_events, final_error_m, p95_error_m,
          passed, landmarks, airspeed_ms, seed
    output_path : str
        File path for the MP4 output (must end in .mp4).
    fps : int
        Frames per second (default 30).

    Returns
    -------
    str — output_path on success, empty string on failure.
    """
    try:
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
    except ImportError as e:
        print(f"Video generation requires opencv-python-headless and matplotlib: {e}",
              file=sys.stderr)
        return ""

    log_entries = result.get("log_entries", [])
    if not log_entries:
        print("No log entries — skipping video.", file=sys.stderr)
        return ""

    vlm_set = {round(t, 1) for t in result.get("vlm_events", [])}
    cel_set = {round(t, 1) for t in result.get("celestial_events", [])}

    # Subsample log_entries to _REPLAY_FRAMES
    n = len(log_entries)
    if n <= _REPLAY_FRAMES:
        indices = list(range(n))
    else:
        step = n / _REPLAY_FRAMES
        indices = [int(i * step) for i in range(_REPLAY_FRAMES)]
        indices[-1] = n - 1  # always include last entry

    sampled = [log_entries[i] for i in indices]
    total_replay = len(sampled)

    # OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (_W, _H))
    if not writer.isOpened():
        print(f"cv2.VideoWriter failed to open: {output_path}", file=sys.stderr)
        return ""

    total_frames = _TITLE_FRAMES + total_replay + _SUMMARY_FRAMES
    print(f"  Generating video: {total_frames} frames @ {fps} fps → {output_path}",
          file=sys.stderr)

    # 1 ── title card ────────────────────────────────────────
    title_frame = _render_title_card(result)
    for _ in range(_TITLE_FRAMES):
        writer.write(title_frame)

    # 2 ── flight replay ─────────────────────────────────────
    for fi, entry in enumerate(sampled):
        frame = _render_replay_frame(
            result, fi, total_replay, sampled, vlm_set, cel_set
        )
        writer.write(frame)
        if fi % 50 == 0:
            pct = (fi + 1) / total_replay * 100
            print(f"    replay {pct:.0f}%", file=sys.stderr)

    # 3 ── summary card ──────────────────────────────────────
    summary_frame = _render_summary_card(result)
    for _ in range(_SUMMARY_FRAMES):
        writer.write(summary_frame)

    writer.release()
    size_kb = 0
    try:
        import os
        size_kb = os.path.getsize(output_path) // 1024
    except OSError:
        pass
    print(f"  Video saved → {output_path}  ({size_kb} KB)", file=sys.stderr)
    return output_path
