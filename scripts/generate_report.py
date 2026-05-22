#!/usr/bin/env python3
"""
scripts/generate_report.py

Generate a Markdown summary report from any run_experiments.py or
run_ablations.py CSV output file.

Produces:
  <output_dir>/report.md   — per-scenario / per-condition summary table,
                              aggregate pass rates, sensor ablation table

Usage:
    # From an ablation sweep:
    python scripts/generate_report.py results/ablations/ablation_results.csv

    # From a standard experiment sweep:
    python scripts/generate_report.py results/latest/run_results.csv

    # Specify output directory explicitly:
    python scripts/generate_report.py results/latest/run_results.csv --output results/latest
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("final_error_m") not in ("", None, "nan")]


def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _pass_rate(rows: list[dict]) -> tuple[int, int]:
    passed = sum(1 for r in rows if _bool(r.get("pass_50m", r.get("passed", False))))
    return passed, len(rows)


def _stats(rows: list[dict]) -> dict:
    errs = [_safe_float(r["final_error_m"]) for r in rows]
    errs = [e for e in errs if e is not None]
    if not errs:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "mean": statistics.mean(errs),
        "median": statistics.median(errs),
        "max": max(errs),
    }


def _is_ablation(rows: list[dict]) -> bool:
    return any("ablation_name" in r and r["ablation_name"] for r in rows)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header(csv_path: str, n_rows: int) -> str:
    ts = datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M EST")
    return (
        f"# UAV Navigation Experiment Report\n\n"
        f"**Generated:** {ts}  \n"
        f"**Source:** `{csv_path}`  \n"
        f"**Total valid runs:** {n_rows}  \n"
        f"**Pass criterion:** final position error < 50 m  \n\n"
        f"> NOTE: VLM observations are geometry-derived (simulated bearings), not real\n"
        f"> VLM inference. These results characterize estimator and data-association\n"
        f"> behaviour, not end-to-end perception performance.\n\n"
    )


def _scenario_table(rows: list[dict]) -> str:
    by_key: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r.get("scenario_label", r.get("scenario_id", "")), r.get("mode", ""))
        by_key[key].append(r)

    lines = [
        "## Results by Scenario\n",
        "| Scenario | Mode | Seeds | Pass@50m | Mean err | Median | Max |",
        "|---|---|---|---|---|---|---|",
    ]
    for (label, mode), grp in sorted(by_key.items()):
        n_pass, n_total = _pass_rate(grp)
        s = _stats(grp)
        lines.append(
            f"| {label} | {mode} | {n_total} | {n_pass}/{n_total} |"
            f" {s['mean']:.1f} m | {s['median']:.1f} m | {s['max']:.1f} m |"
        )
    return "\n".join(lines) + "\n\n"


def _ablation_table(rows: list[dict]) -> str:
    by_key: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r.get("ablation_name", ""), r.get("ablation_label", ""), r.get("mode", ""))
        by_key[key].append(r)

    lines = [
        "## Sensor Ablation Results\n",
        "| Condition | Label | Mode | Seeds | Pass@50m | Mean err | Max err |",
        "|---|---|---|---|---|---|---|",
    ]
    for (aname, alabel, mode), grp in sorted(by_key.items()):
        n_pass, n_total = _pass_rate(grp)
        s = _stats(grp)
        lines.append(
            f"| {aname} | {alabel} | {mode} | {n_total} |"
            f" {n_pass}/{n_total} | {s['mean']:.1f} m | {s['max']:.1f} m |"
        )

    lines += [
        "",
        "### Sensor Configuration Matrix\n",
        "| Condition | IMU | Baro | Mag | Airspeed | VLM | Cel/Solar |",
        "|---|---|---|---|---|---|---|",
    ]
    seen = set()
    for r in rows:
        name = r.get("ablation_name", "")
        if name in seen:
            continue
        seen.add(name)
        t = lambda k: "✓" if _bool(r.get(k, False)) else "✗"
        lines.append(
            f"| {name} | ✓ | {t('barometer_enabled')} | {t('magnetometer_enabled')}"
            f" | {t('airspeed_enabled')} | {t('vlm_enabled')}"
            f" | {'✓' if _bool(r.get('celestial_enabled')) or _bool(r.get('solar_enabled')) else '✗'} |"
        )
    return "\n".join(lines) + "\n\n"


def _vlm_stats_table(rows: list[dict]) -> str:
    if not any("vlm_obs_total" in r for r in rows):
        return ""
    total_obs = sum(int(r.get("vlm_obs_total") or 0) for r in rows)
    total_passed = sum(int(r.get("vlm_passed_confidence", r.get("vlm_passed", 0)) or 0) for r in rows)
    total_bear = sum(int(r.get("vlm_rejected_bearing") or 0) for r in rows)
    total_amb = sum(int(r.get("vlm_rejected_ambiguity") or 0) for r in rows)
    total_ukf = sum(int(r.get("vlm_ukf_updates") or 0) for r in rows)
    if total_obs == 0:
        return ""
    lines = [
        "## VLM Observation Funnel (aggregate)\n",
        "| Stage | Count | % of total |",
        "|---|---|---|",
        f"| Observations seen | {total_obs} | 100% |",
        f"| Passed confidence gate | {total_passed} | {100*total_passed/total_obs:.1f}% |",
        f"| Rejected: bearing gate | {total_bear} | {100*total_bear/total_obs:.1f}% |",
        f"| Rejected: ambiguity gate | {total_amb} | {100*total_amb/total_obs:.1f}% |",
        f"| UKF updates applied | {total_ukf} | {100*total_ukf/total_obs:.1f}% |",
    ]
    return "\n".join(lines) + "\n\n"


def _aggregate_summary(rows: list[dict]) -> str:
    n_pass, n_total = _pass_rate(rows)
    s = _stats(rows)
    nis_alarms = sum(int(r.get("nis_alarm_count", r.get("nees_alarm_count", 0)) or 0) for r in rows)
    lines = [
        "## Aggregate Summary\n",
        f"- **Total runs:** {n_total}",
        f"- **Pass@50m:** {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)" if n_total else "",
        f"- **Mean final error:** {s['mean']:.1f} m",
        f"- **Median final error:** {s['median']:.1f} m",
        f"- **Max final error:** {s['max']:.1f} m",
        f"- **NIS integrity alarms (total):** {nis_alarms}",
    ]
    return "\n".join(l for l in lines if l) + "\n\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(csv_path: str, output_dir: str | None = None) -> str:
    rows = _load_csv(csv_path)
    if not rows:
        print(f"ERROR: no valid rows in {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else Path(csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"

    sections = [_header(csv_path, len(rows))]
    sections.append(_aggregate_summary(rows))
    sections.append(_scenario_table(rows))
    if _is_ablation(rows):
        sections.append(_ablation_table(rows))
    sections.append(_vlm_stats_table(rows))

    content = "".join(sections)
    report_path.write_text(content)
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown report from experiment CSV")
    parser.add_argument("csv", help="Path to run_results.csv or ablation_results.csv")
    parser.add_argument("--output", default=None, help="Output directory (default: same as CSV)")
    args = parser.parse_args()

    path = generate_report(args.csv, args.output)
    print(f"Report → {path}")


if __name__ == "__main__":
    main()
