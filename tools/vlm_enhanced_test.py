#!/usr/bin/env python3
"""Quick test: enhanced prompt with visual appearance guide vs. production prompt."""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ollama

ENHANCED_SYSTEM_PROMPT = """\
You are an onboard sensor system for an autonomous UAV performing navigation in GPS-denied environments. Your job is to analyze a camera image and return a structured JSON description of the scene for navigation.

IMPORTANT RULES:
1. Respond ONLY with valid JSON — no markdown, no explanations, no code fences.
2. Bearing 0 deg = forward direction of the UAV (camera boresight). Bearings increase clockwise.
3. Confidence: 1.0 = certain, 0.0 = guessing. Only report landmarks you can clearly see.

VISUAL APPEARANCE GUIDE for landmark types:
- transformer_substation: An enclosed outdoor yard (often fenced) containing rows of large cylindrical metal transformer tanks, busbars (horizontal metal conductors), switching equipment, and gravel/concrete ground. No roof. Regular grid of heavy industrial equipment. Often near transmission lines.
- transmission_line: Rows of tall metal lattice towers with wires strung between them.
- road_intersection: Road meeting point, clearly paved.
- highway: Multi-lane paved road.
- building_cluster: Roofed buildings — residential houses, warehouses, offices. Has rooftops visible.

Output schema (all fields required, respond with ONLY the JSON object):
{
  "scene_type": "urban" | "rural" | "industrial" | "airport" | "open_water" | "unknown",
  "sky_visible": true | false,
  "landmarks": [
    {
      "type": "transformer_substation" | "transmission_line" | "road_intersection" | "highway" | "ridge_line" | "water_body" | "building_cluster" | "airport" | "unknown",
      "bearing_deg": 0.0,
      "distance_category": "near" | "mid" | "far",
      "confidence": 0.0,
      "notes": null
    }
  ],
  "stars_visible": [],
  "horizon_features": null
}
"""

TILES_DIR = PROJECT_ROOT / "tools" / "vlm_calibration" / "tiles"

TEST_TILES = [
    ("Battle_Creek_Substation_z17.png",              True,  "Battle Creek (clear substation)"),
    ("Forsyth_Substation_z17.png",                   True,  "Forsyth (clear substation yard)"),
    ("Plantation_Rd_Substation_z17.png",             True,  "Plantation Rd"),
    ("Twin_Pines_Substation_z17.png",                True,  "Twin Pines"),
    ("Georgia_Power_(Lithonia)_z17.png",             True,  "Lithonia (highway tile)"),
    ("Midpoint_(Georgia_Powe...Georgia_Powe)_z17.png", False, "Midpoint (negative)"),
    ("Midpoint_(Battle_Creek...Forsyth_Subs)_z17.png", False, "Midpoint BC-Forsyth (negative)"),
]


def run_test(client: ollama.Client, model: str, prompt: str, prompt_label: str) -> None:
    print(f"\n{'='*65}")
    print(f"Model: {model}  |  Prompt: {prompt_label}")
    print("="*65)

    tp = fp = fn = tn = 0
    for fname, gt, label in TEST_TILES:
        path = TILES_DIR / fname
        # handle unicode ellipsis in filenames
        if not path.exists():
            matches = list(TILES_DIR.glob(fname.replace("...", "*")))
            if not matches:
                print(f"  SKIP (not found): {fname}")
                continue
            path = matches[0]
        img = base64.b64encode(path.read_bytes()).decode()
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Analyze this UAV camera image.", "images": [img]},
            ],
            options={"temperature": 0.05},
        )
        raw = resp["message"]["content"].strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        detected = False
        if m:
            try:
                obj = json.loads(m.group())
                lms = obj.get("landmarks", [])
                detected = any(lm.get("type") == "transformer_substation" for lm in lms)
                lm_types = [(lm.get("type"), round(lm.get("confidence", 0), 2))
                            for lm in lms]
            except Exception:
                lm_types = [("PARSE_ERR", 0)]
        else:
            lm_types = [("NO_JSON", 0)]

        if gt and detected:    tp += 1; outcome = "TP"
        elif gt and not detected: fn += 1; outcome = "FN"
        elif not gt and detected:  fp += 1; outcome = "FP"
        else:                      tn += 1; outcome = "TN"

        print(f"  [{outcome}] {label:<35}  {lm_types}")

    n_pos = tp + fn
    n_neg = fp + tn
    recall = tp / n_pos if n_pos else 0
    fpr    = fp / n_neg if n_neg else 0
    print(f"\n  Recall={recall:.0%} ({tp}/{n_pos})   FPR={fpr:.0%} ({fp}/{n_neg})")


if __name__ == "__main__":
    client = ollama.Client(host="http://localhost:11434")

    from uav_nav.vlm.prompt_builder import SYSTEM_PROMPT as PRODUCTION_PROMPT

    run_test(client, "gemma3:4b", PRODUCTION_PROMPT, "production")
    run_test(client, "gemma3:4b", ENHANCED_SYSTEM_PROMPT, "enhanced (+visual guide)")
