#!/usr/bin/env python3
"""
tools/build_paper_figures.py

One-off: generate vector pgfplots figures for the MILCOM paper.

Fig 4 — error-over-time for Much Longer (ATL->JAX, 435 km, 145 min)
  4 lines: daytime passing, daytime failing, nighttime passing, nighttime failing
  Plus the 50 m criterion line.

Fig 5 — pass-rate bar chart across 5 scenarios x 2 modes (from existing sweep JSONs).

Writes:
  research paper/2026-04-24/figures/fig_error_vs_time.tex
  research paper/2026-04-24/figures/fig_pass_rate.tex
  research paper/milcom_2026/figures/fig_error_vs_time.tex   (copy)
  research paper/milcom_2026/figures/fig_pass_rate.tex       (copy)

For fig_error_vs_time we rerun Much Longer at four seeds (two per mode) and
downsample log_entries to ~80 points per trajectory.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_nav.demo.run_scenario import run_scenario
from uav_nav.demo.scenario_configs import ALL_SCENARIOS

_DAYTIME_START_UTC = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)

PAPER_DIRS = [
    PROJECT_ROOT / "research paper" / "2026-04-24" / "figures",
    PROJECT_ROOT / "research paper" / "milcom_2026" / "figures",
]


# ---- Trajectory generation ---------------------------------------

def _downsample(log_entries, n=80):
    if len(log_entries) <= n:
        return log_entries
    step = len(log_entries) // n
    return log_entries[::step]


def _run(cfg, seed, use_solar):
    c = dataclasses.replace(cfg, seed=seed)
    if use_solar:
        c = dataclasses.replace(c, t_start_utc=_DAYTIME_START_UTC,
                                navigation_mode="daytime")
    r = run_scenario(c, output_dir="/tmp/paper_figs", use_solar=use_solar)
    pts = [(e["t_s"] / 60.0, e["error_m"]) for e in _downsample(r["log_entries"])]
    return {
        "seed": seed,
        "final_error_m": r["final_error_m"],
        "passed": r["passed"],
        "trajectory": pts,
    }


def gather_trajectories():
    ml = next(c for c in ALL_SCENARIOS if c.scenario_id == "much_longer")
    # Seeds chosen from post-R6 sweep (results/nighttime_10seed_sweep.json,
    # results/daytime_all5_10seed_sweep.json):
    #   daytime: seed 1 passing (37.7m), seed 3 failing (67.6m)
    #   nighttime: seed 1 passing (15.7m), seed 3 failing (70.5m)
    # Seed 2 now passes in both modes after state-partition removal (R6).
    trajs = {}
    # DR seed=7 gives 84.1 m on Much Longer — representative failing DR trajectory.
    # Seed=1 gives 9.2 m (lucky seed post-Q-fix), seeds 2-6 are 120-283 m (catastrophic).
    print("Running dead-reckoning baseline seed=7 (IMU+baro+mag only)...", file=sys.stderr)
    c_dr = dataclasses.replace(ml, seed=7)
    r_dr = run_scenario(c_dr, output_dir="/tmp/paper_figs",
                        use_solar=False, use_vlm=False, use_celestial=False)
    pts_dr = [(e["t_s"] / 60.0, e["error_m"]) for e in _downsample(r_dr["log_entries"])]
    trajs["baseline"] = {"seed": 7, "final_error_m": r_dr["final_error_m"],
                         "passed": r_dr["passed"], "trajectory": pts_dr}
    print(f"  DR baseline seed=7: final={r_dr['final_error_m']:.1f} m", file=sys.stderr)
    print("Running daytime seed=1 (expect PASS ~37.7m)...", file=sys.stderr)
    trajs["day_pass"] = _run(ml, 1, use_solar=True)
    print("Running daytime seed=3 (expect FAIL ~67.6m)...", file=sys.stderr)
    trajs["day_fail"] = _run(ml, 3, use_solar=True)
    print("Running nighttime seed=1 (expect PASS ~15.7m)...", file=sys.stderr)
    trajs["night_pass"] = _run(ml, 1, use_solar=False)
    print("Running nighttime seed=3 (expect FAIL ~70.5m)...", file=sys.stderr)
    trajs["night_fail"] = _run(ml, 3, use_solar=False)
    return trajs


# ---- Figure 4: error-vs-time ------------------------------------

def _fmt_coords(pts):
    return " ".join(f"({t:.2f},{e:.1f})" for t, e in pts)


def write_fig4(trajs, out_path: Path):
    day_pass = trajs["day_pass"]
    day_fail = trajs["day_fail"]
    night_pass = trajs["night_pass"]
    night_fail = trajs["night_fail"]
    baseline = trajs.get("baseline")  # IMU+baro+mag only, optional

    tmax = 150
    have_baseline = baseline is not None
    if have_baseline:
        peak = max(e for _, e in baseline["trajectory"])
    # Decide y-axis: keep linear when baseline is < 300 m (common case);
    # switch to log when baseline blows up to >300 m so all lines remain
    # readable on the same plot.
    use_log = have_baseline and peak > 300
    if use_log:
        import math
        log_top = max(2.5, math.ceil(math.log10(peak)))
        ymax_log = 10 ** log_top
    else:
        ymax_lin = 250 if have_baseline else 200

    baseline_block = ""
    if have_baseline:
        baseline_block = (
            f"\n%% --- IMU+baro+mag baseline (no VLM, no celestial, no solar) ---\n"
            f"\\addplot[gray!70!black, line width=1.2pt]\n"
            f"  coordinates {{{_fmt_coords(baseline['trajectory'])}}};\n"
            f"\\addlegendentry{{IMU+baro+mag only (no VLM/cel/solar)}}\n"
        )

    if use_log:
        axis_open = (
            f"\\begin{{semilogyaxis}}[\n"
            f"    width=\\columnwidth, height=6.5cm,\n"
            f"    xlabel={{Time (min)}}, ylabel={{Horizontal position error (m, log scale)}},\n"
            f"    xmin=0, xmax={tmax}, ymin=1, ymax={ymax_log:.0f},\n"
            f"    grid=both,\n"
            f"    legend style={{at={{(0.02,0.98)}}, anchor=north west,\n"
            f"        font=\\scriptsize, fill=white, fill opacity=0.85, draw=black!40}},\n"
            f"    tick label style={{font=\\scriptsize}}, label style={{font=\\small}}, clip=true\n"
            f"]"
        )
        axis_close = "\\end{semilogyaxis}"
    else:
        ylim = ymax_lin
        axis_open = (
            f"\\begin{{axis}}[\n"
            f"    width=\\columnwidth, height=6.0cm,\n"
            f"    xlabel={{Time (min)}}, ylabel={{Horizontal position error (m)}},\n"
            f"    xmin=0, xmax={tmax}, ymin=0, ymax={ylim},\n"
            f"    grid=both,\n"
            f"    legend style={{at={{(0.02,0.98)}}, anchor=north west,\n"
            f"        font=\\scriptsize, fill=white, fill opacity=0.85, draw=black!40}},\n"
            f"    tick label style={{font=\\scriptsize}}, label style={{font=\\small}}, clip=true\n"
            f"]"
        )
        axis_close = "\\end{axis}"
    crit_line = (
        f"\\addplot[red, dotted, line width=1.2pt, domain=0:{tmax}] {{50}};\n"
        f"\\addlegendentry{{50\\,m criterion}}"
    )

    with open(out_path, "w") as f:
        f.write(f"""% ============================================================
%  Figure: Horizontal Position Error vs. Time on Much Longer (ATL->JAX)
%  Auto-generated by tools/build_paper_figures.py
% ============================================================
\\begin{{tikzpicture}}
{axis_open}

%% --- 50 m target criterion ---
{crit_line}
{baseline_block}
%% --- Daytime passing (seed={day_pass['seed']}, final {day_pass['final_error_m']:.1f} m) ---
\\addplot[orange!80!black, line width=1.0pt]
  coordinates {{{_fmt_coords(day_pass['trajectory'])}}};
\\addlegendentry{{Daytime, passing seed}}

%% --- Daytime failing (seed={day_fail['seed']}, final {day_fail['final_error_m']:.1f} m) ---
\\addplot[orange!80!black, dashed, line width=1.0pt]
  coordinates {{{_fmt_coords(day_fail['trajectory'])}}};
\\addlegendentry{{Daytime, failing seed}}

%% --- Nighttime passing (seed={night_pass['seed']}, final {night_pass['final_error_m']:.1f} m) ---
\\addplot[blue!70!black, line width=1.0pt]
  coordinates {{{_fmt_coords(night_pass['trajectory'])}}};
\\addlegendentry{{Nighttime, passing seed}}

%% --- Nighttime failing (seed={night_fail['seed']}, final {night_fail['final_error_m']:.1f} m) ---
\\addplot[blue!70!black, dashed, line width=1.0pt]
  coordinates {{{_fmt_coords(night_fail['trajectory'])}}};
\\addlegendentry{{Nighttime, failing seed}}

{axis_close}
\\end{{tikzpicture}}
""")
    print(f"wrote {out_path}", file=sys.stderr)


# ---- Figure 5: pass-rate bar chart -------------------------------

# Pass counts from the two authoritative sweep JSONs (loaded, not hardcoded)
def _load_pass_counts():
    nd = json.loads((PROJECT_ROOT / "results" /
                     "nighttime_10seed_sweep.json").read_text())
    dd = json.loads((PROJECT_ROOT / "results" /
                     "daytime_all5_10seed_sweep.json").read_text())
    # The JSONs key summaries by 'scenario' which is the human label and may
    # include parentheticals (e.g. "Baseline (ATL→MCN)").  We map by
    # scenario_id via per_scenario instead — that is the stable identifier
    # — and read pass_count out of summaries[i] in the same order.
    order = ["very_short", "short", "baseline", "longer", "much_longer"]
    labels = ["Very Short", "Short", "Baseline", "Longer", "Much Longer"]

    def _passes_by_id(data, scenario_id):
        # Find the summary whose per_scenario rows match this id.
        rows = data.get("per_scenario", {}).get(scenario_id, [])
        if rows:
            return sum(1 for r in rows if r.get("passed"))
        # Fallback: try matching summary by index when per_scenario isn't keyed
        for s in data["summaries"]:
            if s.get("scenario_id") == scenario_id:
                return s["pass_count"]
        return 0

    day = [_passes_by_id(dd, sid) for sid in order]
    night = [_passes_by_id(nd, sid) for sid in order]
    return labels, day, night


def write_fig5(out_path: Path):
    labels, day_pass, night_pass = _load_pass_counts()
    # Build coordinates for the two bar series
    # Use numerical x so the bar chart spacing is clean
    day_coords = " ".join(f"({i},{v})" for i, v in enumerate(day_pass))
    night_coords = " ".join(f"({i},{v})" for i, v in enumerate(night_pass))
    xticklabels = ",".join(labels)
    with open(out_path, "w") as f:
        f.write(f"""% ============================================================
%  Figure: 10-seed PASS rate by scenario x mode
%  Auto-generated by tools/build_paper_figures.py
% ============================================================
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=\\columnwidth,
    height=5.5cm,
    ybar,
    bar width=7pt,
    ymin=0, ymax=11,
    ytick={{0,2,4,6,8,10}},
    ylabel={{Seeds passing (out of 10)}},
    symbolic x coords={{Very Short, Short, Baseline, Longer, Much Longer}},
    xtick=data,
    xticklabel style={{font=\\scriptsize, rotate=20, anchor=north east}},
    tick label style={{font=\\scriptsize}},
    label style={{font=\\small}},
    legend style={{
        at={{(0.5,-0.30)}}, anchor=north,
        legend columns=2,
        font=\\scriptsize, fill=white, fill opacity=0.85,
        draw=black!40,
        /tikz/every even column/.append style={{column sep=10pt}}
    }},
    nodes near coords,
    nodes near coords style={{font=\\tiny, /pgf/number format/fixed}},
    enlarge x limits=0.12,
    grid=both,
    grid style={{line width=0.3pt, draw=gray!30}},
    major grid style={{line width=0.5pt, draw=gray!50}}
]
\\addplot[fill=orange!70, draw=orange!90!black]
  coordinates {{({labels[0]},{day_pass[0]}) ({labels[1]},{day_pass[1]}) ({labels[2]},{day_pass[2]}) ({labels[3]},{day_pass[3]}) ({labels[4]},{day_pass[4]})}};
\\addlegendentry{{Daytime (solar + VLM)}}

\\addplot[fill=blue!55, draw=blue!70!black]
  coordinates {{({labels[0]},{night_pass[0]}) ({labels[1]},{night_pass[1]}) ({labels[2]},{night_pass[2]}) ({labels[3]},{night_pass[3]}) ({labels[4]},{night_pass[4]})}};
\\addlegendentry{{Nighttime (stars + VLM)}}

\\end{{axis}}
\\end{{tikzpicture}}
""")
    print(f"wrote {out_path}", file=sys.stderr)


# ---- Main --------------------------------------------------------

def main():
    Path("/tmp/paper_figs").mkdir(exist_ok=True)
    for d in PAPER_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    trajs = gather_trajectories()

    # Save raw trajectory data for reproducibility
    raw_json = PROJECT_ROOT / "results" / "paper_fig_trajectories.json"
    raw_json.parent.mkdir(exist_ok=True)
    with open(raw_json, "w") as f:
        json.dump(trajs, f, indent=2)
    print(f"saved raw trajectory data: {raw_json}", file=sys.stderr)

    for d in PAPER_DIRS:
        write_fig4(trajs, d / "fig_error_vs_time.tex")
        write_fig5(d / "fig_pass_rate.tex")

    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()
