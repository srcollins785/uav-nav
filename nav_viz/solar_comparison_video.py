"""
solar_comparison_video.py — 3-trajectory comparison MP4 for the solar demo.

Overlays dead-reckoning vs solar-aided trajectories to make the solar
contribution unmistakeable.

Layout (1280×720):
  Left  (62%)  — map: two live trajectories drawn simultaneously
  Right top    — error-over-time for both runs (overlaid)
  Right bottom — solar state readout (elevation, azimuth, fix count, mode)

Cards:
  Title (3 s)   — route, seed, what each colour means
  Replay (~16 s, 480 frames) — simultaneous dual-trajectory animation
  Summary (4 s) — final errors, improvement, solar fix count
"""
from __future__ import annotations

import math
import sys
from typing import List

_W, _H = 1280, 720
_FPS = 30
_TITLE_FRAMES   = 90
_SUMMARY_FRAMES = 120
_REPLAY_FRAMES  = 480

PASS_THRESHOLD_M = 50.0

# Palette
_C_DR  = "#f85149"   # red   — dead reckoning
_C_SOL = "#f0a030"   # amber — solar-aided
_C_GT  = "#3fb950"   # green — ground truth
_C_BG  = "#0d1117"
_C_PANEL = "#161b22"
_C_BORDER = "#30363d"
_C_TEXT  = "#e6edf3"
_C_DIM   = "#8b949e"
_C_SUN   = "#f5c518"


def _fig_to_bgr(fig) -> "np.ndarray":
    import numpy as np
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(_H, _W, 4)
    return arr[:, :, [2, 1, 0]].copy()


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def _new_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(_W / 100, _H / 100), dpi=100)
    fig.patch.set_facecolor(_C_BG)
    return fig, plt


# ── Title card ────────────────────────────────────────────────────────────────

def _render_title(r_dr: dict, r_sol: dict) -> "np.ndarray":
    fig, plt = _new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(_C_BG)
    ax.axis("off")

    ax.text(0.5, 0.91, "GPS-Denied Navigation — Solar Isolation Demo",
            transform=ax.transAxes, fontsize=20, fontweight="bold",
            color="#58a6ff", ha="center", va="center")
    ax.text(0.5, 0.82, "Dead Reckoning  vs  Solar-Aided Navigation",
            transform=ax.transAxes, fontsize=15, color=_C_TEXT,
            ha="center", va="center")

    ax.plot([0.05, 0.95], [0.76, 0.76], transform=ax.transAxes,
            color=_C_BORDER, lw=1.5)

    dist_km = r_dr["dist_km"]
    dur_min = r_dr["duration_s"] / 60.0
    seed    = r_dr.get("seed", "—")
    t_start = r_dr.get("t_start_utc", "—")[:19]

    left = [
        ("Route",      f"ATL → Huntsville AL"),
        ("Distance",   f"{dist_km:.1f} km"),
        ("Duration",   f"~{dur_min:.0f} min  @  50 m/s"),
        ("Start UTC",  t_start),
        ("Seed",       str(seed)),
        ("VLM",        "DISABLED — no landmarks on route"),
        ("Mag",        "DISABLED — magnetically disturbed corridor"),
        ("Solar",      "Every 5 min when elevation > 6°"),
    ]

    right = [
        ("", ""),
        ("Colour key", ""),
        (f"  ─── Dead reckoning", "(IMU + Baro + Mag only)"),
        (f"  ─── Solar-aided",    "(+ solar position fixes)"),
        (f"  ─── Ground truth",   "(simulated true position)"),
        ("", ""),
        ("Hypothesis", "Solar keeps error bounded;"),
        ("",           "dead reckoning diverges."),
    ]

    y0, dy = 0.68, 0.072
    for i, (k, v) in enumerate(left):
        y = y0 - i * dy
        ax.text(0.05, y, f"{k}:", transform=ax.transAxes,
                fontsize=11, color=_C_DIM, va="center")
        ax.text(0.22, y, v, transform=ax.transAxes,
                fontsize=11, color=_C_TEXT, va="center", fontweight="bold")

    for i, (k, v) in enumerate(right):
        y = y0 - i * dy
        col = (_C_DR  if "Dead" in k else
               _C_SOL if "Solar" in k else
               _C_GT  if "Ground" in k else _C_TEXT)
        ax.text(0.52, y, k, transform=ax.transAxes,
                fontsize=11, color=col, va="center", fontweight="bold")
        ax.text(0.75, y, v, transform=ax.transAxes,
                fontsize=10, color=_C_DIM, va="center")

    ax.text(0.5, 0.04, "Clark Atlanta University — NSF CPS Medium Proposal",
            transform=ax.transAxes, fontsize=9, color="#484f58",
            ha="center", va="center")

    frame = _fig_to_bgr(fig)
    _close(fig)
    return frame


# ── Replay frame ──────────────────────────────────────────────────────────────

def _render_replay(
    r_dr: dict, r_sol: dict,
    fi: int, total: int,
    log_dr: list, log_sol: list,
    sol_set: set,
) -> "np.ndarray":
    import numpy as np

    fig, plt = _new_fig()

    entry_dr  = log_dr[fi]
    entry_sol = log_sol[fi]
    t_s = entry_dr["t_s"]
    duration_s = r_dr["duration_s"]

    past_dr  = log_dr[:fi + 1]
    past_sol = log_sol[:fi + 1]

    origin_lat = r_dr["origin_lat"]
    origin_lon = r_dr["origin_lon"]
    dest_lat   = r_dr["dest_lat"]
    dest_lon   = r_dr["dest_lon"]

    # Ground truth: straight-line approx (same for both runs — same route/seed)
    n_gt = max(20, fi + 2)
    gt_lats = [origin_lat + (dest_lat - origin_lat) * i / (n_gt - 1) for i in range(n_gt)]
    gt_lons = [origin_lon + (dest_lon - origin_lon) * i / (n_gt - 1) for i in range(n_gt)]

    # ── Map panel ─────────────────────────────────────────────────────────────
    ax_map = fig.add_axes([0.03, 0.10, 0.59, 0.82])
    ax_map.set_facecolor(_C_PANEL)

    ax_map.plot(gt_lons, gt_lats, color=_C_GT, lw=2.0, alpha=0.5,
                label="Ground truth", zorder=2)

    ax_map.plot([e["lon"] for e in past_dr],
                [e["lat"] for e in past_dr],
                color=_C_DR, lw=2.2, label="Dead reckoning", zorder=3)
    ax_map.plot(entry_dr["lon"], entry_dr["lat"],
                marker="^", ms=11, color=_C_DR, zorder=6)

    ax_map.plot([e["lon"] for e in past_sol],
                [e["lat"] for e in past_sol],
                color=_C_SOL, lw=2.2, label="Solar-aided", zorder=4)
    ax_map.plot(entry_sol["lon"], entry_sol["lat"],
                marker="^", ms=11, color=_C_SOL, zorder=6)

    # Solar fix markers
    sol_past = [e for e in past_sol if e["t_s"] in sol_set]
    if sol_past:
        ax_map.scatter([e["lon"] for e in sol_past],
                       [e["lat"] for e in sol_past],
                       s=240, c=_C_SUN, marker="*", zorder=5, alpha=1.0,
                       label="Solar fix")
        for k, e in enumerate(sol_past):
            ax_map.annotate(f"☀{k+1}", xy=(e["lon"], e["lat"]),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=7, color=_C_SUN, fontweight="bold", zorder=7)

    ax_map.plot(origin_lon, origin_lat, "s", ms=10, color=_C_GT,
                label="Origin", zorder=7)
    ax_map.plot(dest_lon,   dest_lat,   "^", ms=10, color="#f85149",
                label="Dest", zorder=7)

    ax_map.set_xlabel("Longitude (°)", fontsize=8, color=_C_DIM)
    ax_map.set_ylabel("Latitude (°)",  fontsize=8, color=_C_DIM)
    ax_map.tick_params(colors=_C_DIM, labelsize=7)
    for sp in ax_map.spines.values():
        sp.set_edgecolor(_C_BORDER)
    ax_map.set_title("ATL → Huntsville AL  |  Solar Isolation Demo",
                     fontsize=10, fontweight="bold", color=_C_TEXT, pad=4)
    ax_map.legend(fontsize=7, loc="lower right",
                  facecolor=_C_PANEL, edgecolor=_C_BORDER, labelcolor=_C_TEXT)
    ax_map.grid(True, alpha=0.15, lw=0.4, color=_C_DIM)

    # ── Error panel ───────────────────────────────────────────────────────────
    ax_err = fig.add_axes([0.65, 0.52, 0.33, 0.40])
    ax_err.set_facecolor(_C_PANEL)

    times_min = [e["t_s"] / 60.0 for e in past_dr]
    errs_dr   = [e["error_m"] for e in past_dr]
    errs_sol  = [e["error_m"] for e in past_sol]

    ax_err.plot(times_min, errs_dr,  color=_C_DR,  lw=1.8, label="Dead reckoning")
    ax_err.plot(times_min, errs_sol, color=_C_SOL, lw=1.8, label="Solar-aided")
    ax_err.axhline(PASS_THRESHOLD_M, color="#f85149", ls=":", lw=1.2, label="50 m limit")
    ax_err.fill_between(times_min, errs_dr,  alpha=0.12, color=_C_DR)
    ax_err.fill_between(times_min, errs_sol, alpha=0.15, color=_C_SOL)

    ax_err.set_xlim(0, duration_s / 60.0)
    max_err = max(max(errs_dr + [1]), max(errs_sol + [1]), PASS_THRESHOLD_M * 1.1)
    ax_err.set_ylim(0, max_err * 1.15)
    ax_err.set_xlabel("Time (min)", fontsize=8, color=_C_DIM)
    ax_err.set_ylabel("Error (m)",  fontsize=8, color=_C_DIM)
    ax_err.set_title("Position Error", fontsize=9, fontweight="bold",
                     color=_C_TEXT, pad=3)
    ax_err.tick_params(colors=_C_DIM, labelsize=7)
    for sp in ax_err.spines.values():
        sp.set_edgecolor(_C_BORDER)
    ax_err.legend(fontsize=7, facecolor=_C_PANEL, edgecolor=_C_BORDER,
                  labelcolor=_C_TEXT)
    ax_err.grid(True, alpha=0.15, lw=0.4, color=_C_DIM)

    # ── Solar state panel ─────────────────────────────────────────────────────
    ax_state = fig.add_axes([0.65, 0.10, 0.33, 0.38])
    ax_state.set_facecolor(_C_PANEL)
    ax_state.axis("off")
    for sp in ax_state.spines.values():
        sp.set_edgecolor(_C_BORDER)

    solar_elev = entry_sol.get("solar_elevation_deg", float("nan"))
    solar_az   = entry_sol.get("solar_azimuth_deg", float("nan"))
    solar_fixes = len(sol_past)

    if not math.isnan(solar_elev) and solar_elev > 6.0:
        mode_str   = "☀  DAYTIME"
        mode_color = _C_SUN
    elif not math.isnan(solar_elev) and solar_elev < -12.0:
        mode_str   = "★  NIGHTTIME"
        mode_color = "#79c0ff"
    else:
        mode_str   = "~  TWILIGHT"
        mode_color = "#e3b341"

    err_dr_now  = entry_dr["error_m"]
    err_sol_now = entry_sol["error_m"]
    delta_now   = err_dr_now - err_sol_now

    def _row(lbl, val, col=_C_TEXT):
        return lbl, val, col

    rows = [
        _row("Time",        f"{t_s:>7.0f} s  ({t_s/60:.1f} min)"),
        _row("Mode",        mode_str, mode_color),
        _row("Solar elev",  f"{solar_elev:>8.1f} °" if not math.isnan(solar_elev) else "—", _C_SUN),
        _row("Solar az",    f"{solar_az:>8.1f} °"   if not math.isnan(solar_az)   else "—", _C_SUN),
        _row("Fixes so far", f"{solar_fixes}",        _C_SUN),
        _row("DR error",    f"{err_dr_now:>7.0f} m", _C_DR),
        _row("Solar error", f"{err_sol_now:>7.0f} m", _C_SOL),
        _row("Improvement", f"{delta_now:>+7.0f} m",
             "#3fb950" if delta_now > 0 else "#f85149"),
    ]

    ax_state.set_title("Solar State", fontsize=9, fontweight="bold",
                       color=_C_TEXT, pad=3)
    ax_state.set_xlim(0, 1)
    ax_state.set_ylim(0, 1)

    y_start, y_step = 0.89, 0.115
    for i, (lbl, val, col) in enumerate(rows):
        y = y_start - i * y_step
        ax_state.text(0.04, y, lbl + ":", fontsize=8.5, color=_C_DIM,
                      transform=ax_state.transAxes, va="center",
                      fontfamily="monospace")
        ax_state.text(0.48, y, val, fontsize=8.5, color=col,
                      transform=ax_state.transAxes, va="center",
                      fontfamily="monospace")

    # ── Progress bar ──────────────────────────────────────────────────────────
    progress = (fi + 1) / total
    bar_ax = fig.add_axes([0.03, 0.03, 0.94, 0.04])
    bar_ax.set_facecolor("#21262d")
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")
    bar_ax.add_patch(plt.Rectangle((0, 0.1), progress, 0.8,
                                   facecolor="#58a6ff", edgecolor="none"))
    bar_ax.text(0.5, 0.5, f"t = {t_s:.0f} s  ({progress*100:.0f}%)",
                ha="center", va="center", fontsize=8,
                color=_C_TEXT, transform=bar_ax.transAxes)

    img = _fig_to_bgr(fig)
    _close(fig)
    return img


# ── Summary card ─────────────────────────────────────────────────────────────

def _render_summary(r_dr: dict, r_sol: dict) -> "np.ndarray":
    fig, plt = _new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(_C_BG)
    ax.axis("off")

    fe_dr  = r_dr["final_error_m"]
    fe_sol = r_sol["final_error_m"]
    improvement = fe_dr - fe_sol
    pct = improvement / fe_dr * 100.0 if fe_dr > 0 else 0.0

    sol_pass = r_sol["passed"]
    dr_pass  = r_dr["passed"]

    ax.text(0.5, 0.90, "Solar Navigation — Results Summary",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            color="#58a6ff", ha="center", va="center")
    ax.text(0.5, 0.81, "ATL → Huntsville AL  |  No VLM Landmarks",
            transform=ax.transAxes, fontsize=13, color=_C_TEXT,
            ha="center", va="center")

    ax.plot([0.05, 0.95], [0.74, 0.74], transform=ax.transAxes,
            color=_C_BORDER, lw=1.5)

    rows = [
        ("Dead reckoning final error",
         f"{fe_dr:.0f} m  ({'PASS ✓' if dr_pass else 'FAIL ✗'})",
         _C_DR),
        ("Solar-aided final error",
         f"{fe_sol:.0f} m  ({'PASS ✓' if sol_pass else 'FAIL ✗'})",
         _C_SOL),
        ("Solar improvement",
         f"{improvement:+.0f} m  ({pct:.0f}% reduction)",
         "#3fb950" if improvement > 0 else "#f85149"),
        ("Solar fixes fused",
         str(r_sol.get("solar_updates", 0)),
         _C_SUN),
        ("Distance",
         f"{r_sol['dist_km']:.1f} km"),
        ("Duration",
         f"{r_sol['duration_s']/60:.0f} min"),
        ("NEES alarms (solar run)",
         str(r_sol.get("nees_alarm_count", 0))),
        ("Acceptance criterion",
         "< 500 m"),
    ]

    y0, dy = 0.67, 0.076
    for i, row in enumerate(rows):
        lbl, val = row[0], row[1]
        col = row[2] if len(row) > 2 else _C_TEXT
        y = y0 - i * dy
        ax.text(0.10, y, lbl + ":", fontsize=12, color=_C_DIM,
                transform=ax.transAxes, va="center")
        ax.text(0.62, y, val, fontsize=12, fontweight="bold", color=col,
                transform=ax.transAxes, va="center")

    ax.text(0.5, 0.04, "Clark Atlanta University — NSF CPS Medium Proposal",
            transform=ax.transAxes, fontsize=9, color="#484f58",
            ha="center", va="center")

    img = _fig_to_bgr(fig)
    _close(fig)
    return img


# ── Public API ────────────────────────────────────────────────────────────────

def generate_solar_comparison_video(
    result_dr: dict,
    result_sol: dict,
    output_path: str,
    fps: int = _FPS,
) -> str:
    """
    Generate the solar isolation comparison MP4.

    Parameters
    ----------
    result_dr  : result dict from the dead-reckoning run (use_solar=False)
    result_sol : result dict from the solar-aided run   (use_solar=True)
    output_path: path for the MP4 file
    fps        : frames per second (default 30)

    Returns
    -------
    output_path on success, empty string on failure.
    """
    try:
        import cv2
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as e:
        print(f"Video generation requires opencv-python-headless and matplotlib: {e}",
              file=sys.stderr)
        return ""

    log_dr  = result_dr.get("log_entries", [])
    log_sol = result_sol.get("log_entries", [])

    if not log_dr or not log_sol:
        print("Empty log entries — skipping video.", file=sys.stderr)
        return ""

    # Align both logs to the shorter length
    n = min(len(log_dr), len(log_sol))
    log_dr  = log_dr[:n]
    log_sol = log_sol[:n]

    # Subsample to _REPLAY_FRAMES
    if n <= _REPLAY_FRAMES:
        indices = list(range(n))
    else:
        step = n / _REPLAY_FRAMES
        indices = [int(i * step) for i in range(_REPLAY_FRAMES)]
        indices[-1] = n - 1

    s_dr  = [log_dr[i]  for i in indices]
    s_sol = [log_sol[i] for i in indices]
    total_replay = len(s_dr)

    sol_set = {round(t, 1) for t in result_sol.get("solar_events", [])}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (_W, _H))
    if not writer.isOpened():
        print(f"cv2.VideoWriter failed: {output_path}", file=sys.stderr)
        return ""

    total_frames = _TITLE_FRAMES + total_replay + _SUMMARY_FRAMES
    print(f"  Generating comparison video: {total_frames} frames → {output_path}",
          file=sys.stderr)

    title = _render_title(result_dr, result_sol)
    for _ in range(_TITLE_FRAMES):
        writer.write(title)

    for fi in range(total_replay):
        frame = _render_replay(result_dr, result_sol, fi, total_replay,
                               s_dr, s_sol, sol_set)
        writer.write(frame)
        if fi % 80 == 0:
            print(f"    replay {fi / total_replay * 100:.0f}%", file=sys.stderr)

    summary = _render_summary(result_dr, result_sol)
    for _ in range(_SUMMARY_FRAMES):
        writer.write(summary)

    writer.release()
    try:
        import os
        size_kb = os.path.getsize(output_path) // 1024
    except OSError:
        size_kb = 0
    print(f"  Comparison video saved → {output_path}  ({size_kb} KB)", file=sys.stderr)
    return output_path
