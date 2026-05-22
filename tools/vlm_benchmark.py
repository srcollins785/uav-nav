#!/usr/bin/env python3
"""
tools/vlm_benchmark.py — P1.1 VLM model selection benchmark.

Fetches aerial ESRI World Imagery tiles centred on known-substation
coordinates (positive) and on corridor midpoints (negative), then
runs each installed Ollama vision model against our production JSON
prompt.  Reports:
  • JSON parse rate
  • Substation detection recall (positive tiles)
  • Substation false-positive rate (negative tiles)
  • Mean confidence on TP detections
  • Mean inference latency (s)

Usage:
    conda run -n base python tools/vlm_benchmark.py [--models m1 m2 ...] [--zoom Z]

Results saved to results/vlm_benchmark/benchmark_results.json and a
Markdown summary table to results/vlm_benchmark/benchmark_summary.md.
"""
from __future__ import annotations

import base64
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
CALIB_DIR = PROJECT_ROOT / "tools" / "vlm_calibration"
OUT_DIR = PROJECT_ROOT / "results" / "vlm_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tile-fetch helpers  (ESRI World Imagery, public service)
# ---------------------------------------------------------------------------
_ESRI_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_TILE_SIZE = 256  # pixels


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def fetch_tile_png(lat: float, lon: float, zoom: int = 17) -> bytes:
    """Return raw PNG bytes for the ESRI tile containing (lat, lon)."""
    x, y = _lat_lon_to_tile(lat, lon, zoom)
    url = _ESRI_URL.format(z=zoom, y=y, x=x)
    req = urllib.request.Request(url, headers={"User-Agent": "vlm-benchmark/1.0 research"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def save_tile(png: bytes, path: Path) -> None:
    path.write_bytes(png)


# ---------------------------------------------------------------------------
# Calibration image catalogue
# ---------------------------------------------------------------------------
# Positive examples: real OSM substation coordinates from scenario configs.
# Negative examples: midpoints between consecutive positive landmarks — open
# rural terrain with no substation.

_POSITIVES: list[dict] = [
    # Very Short corridor
    {"label": "Georgia Power (Lithonia)",  "lat": 33.6549, "lon": -84.3946},
    {"label": "Georgia Power (Snellville)", "lat": 33.7093, "lon": -84.2684},
    {"label": "Georgia Power (Stone Mtn)", "lat": 33.7704, "lon": -84.2104},
    # Short corridor extras
    {"label": "Plantation Rd Substation",  "lat": 33.9108, "lon": -84.0314},
    # Baseline corridor
    {"label": "Battle Creek Substation",   "lat": 33.5496, "lon": -84.3430},
    {"label": "Forsyth Substation",        "lat": 33.0475, "lon": -83.9396},
    {"label": "Twin Pines Substation",     "lat": 32.8356, "lon": -83.7458},
]

def _midpoint(a: dict, b: dict) -> dict:
    return {
        "label": f"Midpoint ({a['label'][:12]}…{b['label'][:12]})",
        "lat": (a["lat"] + b["lat"]) / 2,
        "lon": (a["lon"] + b["lon"]) / 2,
    }

_NEGATIVES: list[dict] = [
    _midpoint(_POSITIVES[0], _POSITIVES[1]),
    _midpoint(_POSITIVES[1], _POSITIVES[2]),
    _midpoint(_POSITIVES[2], _POSITIVES[3]),
    _midpoint(_POSITIVES[3], _POSITIVES[4]),
    _midpoint(_POSITIVES[4], _POSITIVES[5]),
    _midpoint(_POSITIVES[5], _POSITIVES[6]),
]


# ---------------------------------------------------------------------------
# Prompt — use production prompt_builder directly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT))
from uav_nav.vlm.prompt_builder import SYSTEM_PROMPT, build_prompt


def _build_messages(image_b64: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(), "images": [image_b64]},
    ]


# ---------------------------------------------------------------------------
# Inference + parse
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict | None:
    """Try to extract a valid JSON dict from model output."""
    raw = raw.strip()
    # Layer 1: direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Layer 2: find first {...} block
    import re
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _has_substation(parsed: dict | None) -> tuple[bool, float]:
    """Return (detected, max_confidence) for transformer_substation entries."""
    if not parsed:
        return False, 0.0
    landmarks = parsed.get("landmarks", [])
    if not isinstance(landmarks, list):
        return False, 0.0
    subs = [lm for lm in landmarks if isinstance(lm, dict)
            and lm.get("type") == "transformer_substation"]
    if not subs:
        return False, 0.0
    max_conf = max(float(lm.get("confidence", 0.0)) for lm in subs)
    return True, max_conf


def run_inference(client: Any, model: str, image_b64: str) -> dict:
    """Run one inference; return metrics dict."""
    messages = _build_messages(image_b64)
    t0 = time.perf_counter()
    resp = client.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.05},
    )
    latency = time.perf_counter() - t0
    raw = resp["message"]["content"]
    parsed = _parse_response(raw)
    detected, conf = _has_substation(parsed)
    return {
        "raw": raw[:300],   # truncate for storage
        "json_valid": parsed is not None,
        "detected": detected,
        "confidence": conf,
        "latency_s": round(latency, 2),
    }


# ---------------------------------------------------------------------------
# Fetch / load tiles
# ---------------------------------------------------------------------------

def get_tiles(zoom: int = 17) -> tuple[list[dict], list[dict]]:
    """
    Return (positives, negatives) each as list of dicts with keys:
      label, lat, lon, ground_truth (bool), image_b64
    Downloads tiles not already cached in vlm_calibration/.
    """
    tiles_dir = CALIB_DIR / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    def _load(entry: dict, gt: bool) -> dict:
        slug = entry["label"].replace(" ", "_").replace("/", "-")[:40]
        fname = tiles_dir / f"{slug}_z{zoom}.png"
        if not fname.exists():
            print(f"  fetching {entry['label']} ({entry['lat']:.4f}, {entry['lon']:.4f})…", end=" ", flush=True)
            try:
                png = fetch_tile_png(entry["lat"], entry["lon"], zoom)
                fname.write_bytes(png)
                print("ok")
            except Exception as exc:
                print(f"FAILED: {exc}")
                return None
        b64 = base64.b64encode(fname.read_bytes()).decode()
        return {**entry, "ground_truth": gt, "image_b64": b64, "path": str(fname)}

    print("Loading positive tiles (substations)…")
    pos = [r for e in _POSITIVES if (r := _load(e, True)) is not None]
    print("Loading negative tiles (non-substation terrain)…")
    neg = [r for e in _NEGATIVES if (r := _load(e, False)) is not None]

    # Also include any existing calibration images
    for existing in CALIB_DIR.glob("*.png"):
        b64 = base64.b64encode(existing.read_bytes()).decode()
        pos.append({
            "label": existing.stem,
            "lat": None, "lon": None,
            "ground_truth": True,  # all hand-selected calibration images are positives
            "image_b64": b64,
            "path": str(existing),
        })

    return pos, neg


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(models: list[str], zoom: int = 17) -> None:
    import ollama

    client = ollama.Client(host="http://localhost:11434")

    # Verify models are available
    available = {m.model for m in client.list().models}
    missing = [m for m in models if m not in available]
    if missing:
        print(f"WARNING: models not installed: {missing}")
        models = [m for m in models if m in available]
    if not models:
        print("No models available — aborting.")
        sys.exit(1)

    print(f"\nModels to benchmark: {models}")
    pos_tiles, neg_tiles = get_tiles(zoom)
    all_tiles = [(t, True) for t in pos_tiles] + [(t, False) for t in neg_tiles]
    print(f"\nTile set: {len(pos_tiles)} positive, {len(neg_tiles)} negative\n")

    results: dict[str, list[dict]] = {m: [] for m in models}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print('='*60)
        for tile, gt in all_tiles:
            print(f"  [{'+' if gt else '-'}] {tile['label'][:50]}…", end=" ", flush=True)
            try:
                metrics = run_inference(client, model, tile["image_b64"])
                record = {
                    "label": tile["label"],
                    "ground_truth": gt,
                    **metrics,
                }
                results[model].append(record)
                verdict = "✓" if metrics["json_valid"] else "✗JSON"
                det = "SUB" if metrics["detected"] else "---"
                print(f"{verdict} {det} conf={metrics['confidence']:.2f} {metrics['latency_s']:.1f}s")
            except Exception as exc:
                print(f"ERROR: {exc}")
                results[model].append({
                    "label": tile["label"], "ground_truth": gt,
                    "raw": str(exc), "json_valid": False,
                    "detected": False, "confidence": 0.0, "latency_s": 0.0,
                })

    # Save raw results
    out_json = OUT_DIR / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nRaw results → {out_json}")

    # Compute summary metrics
    summary = {}
    for model, records in results.items():
        if not records:
            continue
        pos_rec = [r for r in records if r["ground_truth"]]
        neg_rec = [r for r in records if not r["ground_truth"]]
        n_total = len(records)

        json_rate = sum(r["json_valid"] for r in records) / n_total if n_total else 0
        recall = (sum(r["detected"] for r in pos_rec) / len(pos_rec)) if pos_rec else 0
        fpr = (sum(r["detected"] for r in neg_rec) / len(neg_rec)) if neg_rec else 0
        tp_recs = [r for r in pos_rec if r["detected"]]
        mean_tp_conf = (sum(r["confidence"] for r in tp_recs) / len(tp_recs)) if tp_recs else 0
        mean_lat = sum(r["latency_s"] for r in records) / n_total if n_total else 0

        summary[model] = {
            "n_images": n_total,
            "json_rate": round(json_rate, 3),
            "recall": round(recall, 3),
            "fpr": round(fpr, 3),
            "mean_tp_conf": round(mean_tp_conf, 3),
            "mean_latency_s": round(mean_lat, 2),
        }

    # Print and save Markdown summary
    header = ("| Model | JSON% | Recall | FPR | Conf(TP) | Latency |\n"
               "|---|---|---|---|---|---|\n")
    rows = []
    for model, m in summary.items():
        rows.append(
            f"| {model} "
            f"| {m['json_rate']*100:.0f}% "
            f"| {m['recall']*100:.0f}% "
            f"| {m['fpr']*100:.0f}% "
            f"| {m['mean_tp_conf']:.2f} "
            f"| {m['mean_latency_s']:.1f}s |"
        )
    table = header + "\n".join(rows)

    md = f"""# VLM Model Benchmark — P1.1 Selection

**Date:** {time.strftime('%Y-%m-%d')}
**Zoom level:** {zoom} (~1.2 m/px ESRI World Imagery, 256×256 px tile)
**Images:** {len(pos_tiles)} positive (substation), {len(neg_tiles)} negative
**Prompt:** Production `SYSTEM_PROMPT` + `build_prompt()` from `uav_nav/vlm/prompt_builder.py`

## Results

{table}

**Columns:**
- JSON%: fraction of responses that parse as valid JSON
- Recall: fraction of positive (substation) tiles where model detected `transformer_substation`
- FPR: fraction of negative tiles where model falsely detected `transformer_substation`
- Conf(TP): mean confidence on true-positive detections
- Latency: mean inference time per image

## Raw Results
See `results/vlm_benchmark/benchmark_results.json` for per-image breakdown.
"""
    out_md = OUT_DIR / "benchmark_summary.md"
    out_md.write_text(md)

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(table)
    print(f"\nSummary → {out_md}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM model selection benchmark")
    parser.add_argument(
        "--models", nargs="+",
        default=["gemma3:4b", "moondream:latest", "bakllava:latest"],
        help="Ollama model tags to benchmark",
    )
    parser.add_argument("--zoom", type=int, default=17,
                        help="ESRI tile zoom level (default 17 ≈ 300m × 300m)")
    args = parser.parse_args()

    run_benchmark(args.models, zoom=args.zoom)
