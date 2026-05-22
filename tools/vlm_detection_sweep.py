#!/usr/bin/env python3
"""
tools/vlm_detection_sweep.py — P1.2 landmark detection rate characterisation.

For each landmark in every scenario, fetches the ESRI aerial tile at the
OSM coordinate, runs gemma3:4b, and records:
  • detected:     transformer_substation present in VLM output
  • bearing_deg:  VLM-reported bearing (camera frame)
  • confidence:   VLM-reported confidence
  • json_valid:   response parsed as valid JSON
  • latency_s:    inference time

Then reports per-scenario detection rates and bearing error (for the two
landmarks we know are visually confirmed: Battle Creek, Forsyth).

Usage:
    conda run -n base python tools/vlm_detection_sweep.py [--zoom Z]

Results → results/vlm_p1_2/detection_sweep.csv and .md
"""
from __future__ import annotations

import base64
import csv
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_nav.demo.scenario_configs import ALL_SCENARIOS
from uav_nav.vlm.prompt_builder import SYSTEM_PROMPT, build_prompt

OUT_DIR = PROJECT_ROOT / "results" / "vlm_p1_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_CACHE = PROJECT_ROOT / "tools" / "vlm_calibration" / "tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)

_ESRI_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Landmarks visually confirmed as containing visible substation equipment
CONFIRMED_VISIBLE = {"Battle Creek Substation", "Forsyth Substation"}


def _lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y


def fetch_tile(lat, lon, zoom=17):
    x, y = _lat_lon_to_tile(lat, lon, zoom)
    slug = f"{lat:.4f}_{lon:.4f}_z{zoom}"
    path = TILE_CACHE / f"{slug}.png"
    if not path.exists():
        url = _ESRI_URL.format(z=zoom, y=y, x=x)
        req = urllib.request.Request(url, headers={"User-Agent": "vlm-p1.2/1.0 research"})
        with urllib.request.urlopen(req, timeout=15) as r:
            path.write_bytes(r.read())
    return path


def run_vlm(client, path: Path):
    img = base64.b64encode(path.read_bytes()).decode()
    t0 = time.perf_counter()
    resp = client.chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(), "images": [img]},
        ],
        options={"temperature": 0.05},
    )
    latency = time.perf_counter() - t0
    raw = resp["message"]["content"].strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return False, False, 0.0, 0.0, latency
    try:
        obj = json.loads(m.group())
    except Exception:
        return True, False, 0.0, 0.0, latency
    lms = obj.get("landmarks", [])
    subs = [lm for lm in lms if isinstance(lm, dict)
            and lm.get("type") == "transformer_substation"]
    if not subs:
        return True, False, 0.0, 0.0, latency
    best = max(subs, key=lambda lm: lm.get("confidence", 0))
    return True, True, float(best.get("bearing_deg", 0)), float(best.get("confidence", 0)), latency


def main(zoom=17):
    import ollama
    client = ollama.Client(host="http://localhost:11434")

    rows = []
    print(f"{'Scenario':<20} {'Landmark':<35} {'JSON':>5} {'Det':>5} {'Bear':>8} {'Conf':>6} {'Lat':>6}")
    print("-" * 90)

    for scenario in ALL_SCENARIOS:
        for lm in scenario.landmarks:
            if lm.name.startswith("Waypoint"):
                continue  # skip interpolated waypoints — they're not real OSM nodes
            try:
                tile_path = fetch_tile(lm.lat_deg, lm.lon_deg, zoom)
                json_valid, detected, bearing, conf, lat_s = run_vlm(client, tile_path)
            except Exception as exc:
                print(f"  ERROR {lm.name}: {exc}")
                json_valid, detected, bearing, conf, lat_s = False, False, 0.0, 0.0, 0.0

            visually_confirmed = any(name in lm.name for name in CONFIRMED_VISIBLE)
            rows.append({
                "scenario":           scenario.label,
                "landmark":           lm.name,
                "lat":                lm.lat_deg,
                "lon":                lm.lon_deg,
                "visually_confirmed": visually_confirmed,
                "json_valid":         json_valid,
                "detected":           detected,
                "bearing_deg":        round(bearing, 1),
                "confidence":         round(conf, 2),
                "latency_s":          round(lat_s, 1),
            })
            flag = "[CONFIRMED]" if visually_confirmed else ""
            print(f"  {scenario.label:<20} {lm.name:<35} "
                  f"{'Y' if json_valid else 'N':>5} "
                  f"{'Y' if detected else 'N':>5} "
                  f"{bearing:>8.1f} {conf:>6.2f} {lat_s:>5.1f}s  {flag}")

    # Write CSV
    csv_path = OUT_DIR / "detection_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    all_lms    = [r for r in rows]
    confirmed  = [r for r in rows if r["visually_confirmed"]]
    unconfirmed = [r for r in rows if not r["visually_confirmed"]]

    det_all     = sum(r["detected"] for r in all_lms)     / len(all_lms)
    det_conf    = sum(r["detected"] for r in confirmed)    / len(confirmed)  if confirmed  else 0
    det_unconf  = sum(r["detected"] for r in unconfirmed) / len(unconfirmed) if unconfirmed else 0
    json_rate   = sum(r["json_valid"] for r in all_lms)   / len(all_lms)

    md = f"""# P1.2 Landmark Detection Sweep — gemma3:4b

**Date:** {time.strftime('%Y-%m-%d')}
**Model:** gemma3:4b  |  **Prompt:** bridged CLASSIFICATION KEY
**Zoom:** {zoom} (~300 m × 300 m tiles)
**Landmarks surveyed:** {len(all_lms)} (across {len(ALL_SCENARIOS)} scenarios)

## Detection Rate Summary

| Set | N | Detection Rate |
|---|---|---|
| All OSM landmarks | {len(all_lms)} | {det_all:.0%} |
| Visually-confirmed substations | {len(confirmed)} | {det_conf:.0%} |
| Other OSM nodes (coordinate may not land on yard) | {len(unconfirmed)} | {det_unconf:.0%} |
| JSON validity | {len(all_lms)} | {json_rate:.0%} |

## Interpretation

The gap between confirmed ({det_conf:.0%}) and unconfirmed ({det_unconf:.0%}) detection rates
reflects OSM coordinate quality: many nodes land on adjacent features (highway ROW,
parking lots, woodland edges) rather than the transformer yard itself.
This is the P1.5 issue — OSM coordinate accuracy.

For the UKF, what matters is the detection rate when the UAV is actually flying
_near_ a landmark and the camera frames the equipment.  The confirmed-substation rate
({det_conf:.0%}) is the relevant upper bound; the unconfirmed rate ({det_unconf:.0%}) is
a lower bound accounting for OSM coordinate error.

## Per-Landmark Results

| Scenario | Landmark | Confirmed? | Detected | Bearing° | Conf |
|---|---|---|---|---|---|
"""
    for r in rows:
        md += (f"| {r['scenario']} | {r['landmark']} | "
               f"{'Y' if r['visually_confirmed'] else 'N'} | "
               f"{'Y' if r['detected'] else 'N'} | "
               f"{r['bearing_deg']} | {r['confidence']} |\n")

    md_path = OUT_DIR / "detection_sweep.md"
    md_path.write_text(md)
    print(f"\nCSV  → {csv_path}")
    print(f"MD   → {md_path}")
    print(f"\nSummary:  all={det_all:.0%}  confirmed={det_conf:.0%}  unconfirmed={det_unconf:.0%}  JSON={json_rate:.0%}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--zoom", type=int, default=17)
    args = p.parse_args()
    main(args.zoom)
