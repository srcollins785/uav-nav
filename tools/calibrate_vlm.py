#!/usr/bin/env python3
"""
tools/calibrate_vlm.py

Empirical VLM calibration against real aerial imagery of Georgia substations.

Stage 1 (detection):
    Fetch top-down aerial tiles centered on known OSM substations, query the
    local VLM (Ollama/gemma3), record whether each substation is detected
    and with what confidence.  This triages whether the VLM is suitable for
    the task before investing in bearing calibration.

Stage 2 (bearing) — run only if Stage 1 detection rate > ~60%:
    Fetch tiles centered 500 m from each substation, so the substation
    appears off-center at a known ground-truth bearing.  Measure empirical
    bearing_deg error sigma.  Use result to calibrate
    MeasurementNoise.vlm_bearing_deg in uav_nav/config.yaml.

Imagery source: ArcGIS World Imagery (no API key, public tile service).
Output: tools/vlm_calibration/*.json + *.png

Usage:
    # Stage 1: 5-substation detection smoke test
    conda run -n base python tools/calibrate_vlm.py --stage detection --n 5

    # Full detection run (all substations from scenario_configs)
    conda run -n base python tools/calibrate_vlm.py --stage detection

    # Stage 2: bearing calibration (requires successful detection)
    conda run -n base python tools/calibrate_vlm.py --stage bearing
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

# ruff: noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uav_nav.config import get_config
from uav_nav.demo.scenario_configs import ALL_SCENARIOS, LandmarkConfig
from uav_nav.vlm.client import VLMClient
from uav_nav.vlm.schemas import LandmarkType, SemanticObservation

ARCGIS_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TILE_SIZE_PX = 256
OUT_DIR = Path(__file__).parent / "vlm_calibration"


def lonlat_to_tile(lon_deg: float, lat_deg: float, zoom: int) -> Tuple[int, int, float, float]:
    """Return (x_tile, y_tile, x_frac, y_frac) for a lon/lat at zoom."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xf = (lon_deg + 180.0) / 360.0 * n
    yf = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return int(xf), int(yf), xf - int(xf), yf - int(yf)


def fetch_tile(x: int, y: int, z: int) -> Image.Image:
    url = ARCGIS_TILE_URL.format(z=z, x=x, y=y)
    headers = {"User-Agent": "uav_nav-calibration/1.0 (academic research)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def fetch_patch(lat: float, lon: float, zoom: int = 18, size_px: int = 512) -> Image.Image:
    """
    Fetch a size_px × size_px aerial patch centered on (lat, lon) by
    stitching adjacent tiles.  zoom=18 is ~0.6 m/pixel.
    """
    xt, yt, xf, yf = lonlat_to_tile(lon, lat, zoom)
    # center pixel within the tile grid
    cx = xt * TILE_SIZE_PX + xf * TILE_SIZE_PX
    cy = yt * TILE_SIZE_PX + yf * TILE_SIZE_PX
    half = size_px // 2

    x_min = int((cx - half) // TILE_SIZE_PX)
    x_max = int((cx + half) // TILE_SIZE_PX)
    y_min = int((cy - half) // TILE_SIZE_PX)
    y_max = int((cy + half) // TILE_SIZE_PX)

    tiles_w = (x_max - x_min + 1) * TILE_SIZE_PX
    tiles_h = (y_max - y_min + 1) * TILE_SIZE_PX
    canvas = Image.new("RGB", (tiles_w, tiles_h))

    for xi in range(x_min, x_max + 1):
        for yi in range(y_min, y_max + 1):
            tile = fetch_tile(xi, yi, zoom)
            canvas.paste(
                tile,
                ((xi - x_min) * TILE_SIZE_PX, (yi - y_min) * TILE_SIZE_PX),
            )

    left = int(cx - x_min * TILE_SIZE_PX) - half
    top = int(cy - y_min * TILE_SIZE_PX) - half
    return canvas.crop((left, top, left + size_px, top + size_px))


@dataclass
class DetectionResult:
    landmark_name: str
    lat: float
    lon: float
    scenario_id: str
    image_path: str
    detected: bool
    top_confidence: float
    top_landmark_type: str
    top_bearing_deg: Optional[float]
    all_landmarks: list = field(default_factory=list)
    error: Optional[str] = None


def unique_landmarks() -> List[Tuple[LandmarkConfig, str]]:
    """Collect unique landmarks across all scenarios (deduplicated by lat/lon)."""
    seen: dict = {}
    for scen in ALL_SCENARIOS:
        for lm in scen.landmarks:
            key = (round(lm.lat_deg, 5), round(lm.lon_deg, 5))
            if key not in seen:
                seen[key] = (lm, scen.scenario_id)
    return list(seen.values())


def run_detection(n: Optional[int], zoom: int = 18) -> List[DetectionResult]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = VLMClient()
    if not client.is_available():
        print("ERROR: Ollama server not reachable.  Start with: ollama serve")
        sys.exit(1)

    all_lms = unique_landmarks()
    if n is not None:
        all_lms = all_lms[:n]

    print(f"Running detection on {len(all_lms)} substations using {client.model}")
    print(f"Imagery: ArcGIS World Imagery zoom={zoom}")
    print()

    results: List[DetectionResult] = []
    for i, (lm, scen_id) in enumerate(all_lms, 1):
        safe_name = lm.name.replace(" ", "_").replace("/", "_")
        img_path = OUT_DIR / f"{scen_id}_{i:02d}_{safe_name}.png"
        label = f"[{i:02d}/{len(all_lms)}] {lm.name} ({scen_id})"

        try:
            patch = fetch_patch(lm.lat_deg, lm.lon_deg, zoom=zoom)
            patch.save(img_path)
        except Exception as exc:
            print(f"{label}: FETCH FAIL — {exc}")
            results.append(DetectionResult(
                landmark_name=lm.name, lat=lm.lat_deg, lon=lm.lon_deg,
                scenario_id=scen_id, image_path=str(img_path),
                detected=False, top_confidence=0.0, top_landmark_type="",
                top_bearing_deg=None, error=str(exc),
            ))
            continue

        t0 = time.time()
        try:
            obs: Optional[SemanticObservation] = client.query(img_path)
        except Exception as exc:
            print(f"{label}: VLM FAIL — {exc}")
            results.append(DetectionResult(
                landmark_name=lm.name, lat=lm.lat_deg, lon=lm.lon_deg,
                scenario_id=scen_id, image_path=str(img_path),
                detected=False, top_confidence=0.0, top_landmark_type="",
                top_bearing_deg=None, error=str(exc),
            ))
            continue
        dt = time.time() - t0

        if obs is None or not obs.landmarks:
            print(f"{label}: no landmarks ({dt:.1f}s)")
            results.append(DetectionResult(
                landmark_name=lm.name, lat=lm.lat_deg, lon=lm.lon_deg,
                scenario_id=scen_id, image_path=str(img_path),
                detected=False, top_confidence=0.0, top_landmark_type="",
                top_bearing_deg=None,
            ))
            continue

        substations = [l for l in obs.landmarks
                       if l.type == LandmarkType.transformer_substation]
        detected = len(substations) > 0
        top = substations[0] if detected else obs.landmarks[0]

        marker = "✓" if detected else "✗"
        print(f"{label}: {marker} type={top.type.value} "
              f"conf={top.confidence:.2f} bear={top.bearing_deg:.0f}° "
              f"({dt:.1f}s)")

        results.append(DetectionResult(
            landmark_name=lm.name, lat=lm.lat_deg, lon=lm.lon_deg,
            scenario_id=scen_id, image_path=str(img_path),
            detected=detected, top_confidence=top.confidence,
            top_landmark_type=top.type.value, top_bearing_deg=top.bearing_deg,
            all_landmarks=[{"type": l.type.value, "bearing_deg": l.bearing_deg,
                            "confidence": l.confidence} for l in obs.landmarks],
        ))

    return results


def summarize(results: List[DetectionResult]) -> None:
    n = len(results)
    detected = sum(1 for r in results if r.detected)
    any_lm = sum(1 for r in results if r.top_landmark_type)
    mean_conf = (np.mean([r.top_confidence for r in results if r.detected])
                 if detected else 0.0)

    print("\n" + "=" * 60)
    print(f"DETECTION SUMMARY  ({n} substations)")
    print("=" * 60)
    print(f"  Correctly identified as transformer_substation: {detected}/{n} ({100*detected/n:.0f}%)")
    print(f"  Any landmark detected (any type)              : {any_lm}/{n} ({100*any_lm/n:.0f}%)")
    print(f"  Mean confidence on correct detections         : {mean_conf:.2f}")
    print()
    if detected / n >= 0.6:
        print("  → Stage 2 (bearing calibration) is viable.")
    elif any_lm / n >= 0.5:
        print("  → VLM sees SOMETHING but miscategorises substations.")
        print("     Options: prompt tuning, larger VLM, or relax to 'any landmark'.")
    else:
        print("  → VLM does not reliably see substations in aerial imagery.")
        print("     Options: switch VLM model, adjust zoom/FOV, or retrain prompt.")


@dataclass
class BearingResult:
    landmark_name: str
    lat: float
    lon: float
    scenario_id: str
    direction: str          # N / S / E / W
    offset_m: float
    beta_true_deg: float    # ground-truth camera-to-substation bearing
    beta_reported_deg: Optional[float]
    confidence: float
    beta_error_deg: Optional[float]   # circular difference: reported - true
    image_path: str
    detected: bool
    error: Optional[str] = None


def _circular_diff(a: float, b: float) -> float:
    """Signed circular difference a - b in [-180, 180)."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def _offset_latlon(lat: float, lon: float,
                   north_m: float, east_m: float) -> Tuple[float, float]:
    """Return (lat, lon) displaced by north_m metres north and east_m metres east."""
    R = 6_371_000.0
    dlat = math.degrees(north_m / R)
    dlon = math.degrees(east_m / (R * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def run_bearing_calibration(n: Optional[int] = None,
                             zoom: int = 17,
                             offset_m: float = 300.0) -> List[BearingResult]:
    """
    Stage 2 — empirical VLM bearing accuracy measurement.

    For each detected substation from Stage 1, fetch four aerial tiles where
    the camera centre is displaced offset_m in each cardinal direction from the
    substation.  The substation therefore appears at a known ground-truth
    camera-frame bearing:

        Camera N of sub  →  sub is South  →  beta_true = 180°
        Camera S of sub  →  sub is North  →  beta_true =   0°
        Camera E of sub  →  sub is West   →  beta_true = 270°
        Camera W of sub  →  sub is East   →  beta_true =  90°

    At zoom=17 (~1.2 m/px) a 512-px patch covers ~614 m; a 300 m offset places
    the substation 250 px from the image centre — clearly within frame and
    recognisable at this resolution.

    Outputs error statistics that characterise sigma_0 and test whether the
    1/c^2 confidence-adaptive noise model is empirically supported.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    det_json = OUT_DIR / "detection_results.json"
    if not det_json.exists():
        print("ERROR: detection_results.json not found — run --stage detection first.")
        sys.exit(1)

    det_data = json.loads(det_json.read_text())
    detected = [
        d for d in det_data
        if d["detected"] and "Waypoint" not in d["landmark_name"]
    ]
    if n is not None:
        detected = detected[:n]

    print(f"Stage 2: bearing calibration on {len(detected)} detected substations")
    print(f"  zoom={zoom} (~{156543.03392 * math.cos(math.radians(33.6)) / (2**zoom):.1f} m/px at 33°N)")
    print(f"  offset={offset_m:.0f} m  |  4 directions each  |  {len(detected)*4} total VLM calls")
    print()

    client = VLMClient()
    if not client.is_available():
        print("ERROR: Ollama not reachable.")
        sys.exit(1)

    DIRECTIONS = {
        "N": ( offset_m,  0.0),   # camera north  → sub south  → beta_true 180
        "S": (-offset_m,  0.0),   # camera south  → sub north  → beta_true   0
        "E": ( 0.0,  offset_m),   # camera east   → sub west   → beta_true 270
        "W": ( 0.0, -offset_m),   # camera west   → sub east   → beta_true  90
    }

    results: List[BearingResult] = []
    total = len(detected) * len(DIRECTIONS)
    idx = 0

    for det in detected:
        lat, lon = det["lat"], det["lon"]
        name = det["landmark_name"]
        sid = det["scenario_id"]

        for direction, (dn, de) in DIRECTIONS.items():
            idx += 1
            cam_lat, cam_lon = _offset_latlon(lat, lon, dn, de)

            # Ground-truth bearing from camera to substation
            # sub is at (-dn, -de) relative to camera
            beta_true = math.degrees(math.atan2(-de, -dn)) % 360.0

            safe_name = name.replace(" ", "_").replace("/", "_")
            img_path = OUT_DIR / f"bear_{sid}_{safe_name}_{direction}.png"
            label = f"[{idx:03d}/{total}] {name} dir={direction} β_true={beta_true:.0f}°"

            try:
                patch = fetch_patch(cam_lat, cam_lon, zoom=zoom, size_px=512)
                patch.save(img_path)
            except Exception as exc:
                print(f"{label}: FETCH FAIL — {exc}")
                results.append(BearingResult(
                    landmark_name=name, lat=lat, lon=lon, scenario_id=sid,
                    direction=direction, offset_m=offset_m,
                    beta_true_deg=beta_true, beta_reported_deg=None,
                    confidence=0.0, beta_error_deg=None,
                    image_path=str(img_path), detected=False, error=str(exc),
                ))
                continue

            try:
                obs = client.query(img_path)
            except Exception as exc:
                print(f"{label}: VLM FAIL — {exc}")
                results.append(BearingResult(
                    landmark_name=name, lat=lat, lon=lon, scenario_id=sid,
                    direction=direction, offset_m=offset_m,
                    beta_true_deg=beta_true, beta_reported_deg=None,
                    confidence=0.0, beta_error_deg=None,
                    image_path=str(img_path), detected=False, error=str(exc),
                ))
                continue

            if obs is None:
                print(f"{label}: no observation returned")
                results.append(BearingResult(
                    landmark_name=name, lat=lat, lon=lon, scenario_id=sid,
                    direction=direction, offset_m=offset_m,
                    beta_true_deg=beta_true, beta_reported_deg=None,
                    confidence=0.0, beta_error_deg=None,
                    image_path=str(img_path), detected=False,
                ))
                continue

            subs = [lm for lm in (obs.landmarks or [])
                    if lm.type == LandmarkType.transformer_substation]

            if subs:
                top = subs[0]
                err = _circular_diff(top.bearing_deg, beta_true)
                marker = "✓"
            else:
                top = None
                err = None
                marker = "✗"

            reported = top.bearing_deg if top else None
            conf = top.confidence if top else 0.0
            print(f"{label}: {marker} β_rep={reported if reported is not None else '--':>5} "
                  f"err={f'{err:.1f}°' if err is not None else '--':>7}  conf={conf:.2f}")

            results.append(BearingResult(
                landmark_name=name, lat=lat, lon=lon, scenario_id=sid,
                direction=direction, offset_m=offset_m,
                beta_true_deg=beta_true,
                beta_reported_deg=reported,
                confidence=conf,
                beta_error_deg=err,
                image_path=str(img_path),
                detected=top is not None,
            ))

    return results


def summarize_bearing(results: List[BearingResult]) -> None:
    valid = [r for r in results if r.beta_error_deg is not None]
    n_total = len(results)
    n_valid = len(valid)
    n_detected = sum(1 for r in results if r.detected)
    n_fetch_fail = sum(1 for r in results if r.error)

    print("\n" + "=" * 65)
    print(f"BEARING CALIBRATION SUMMARY  ({n_total} offset tiles)")
    print("=" * 65)
    print(f"  Substation detected in offset tile : {n_detected}/{n_total} "
          f"({100*n_detected/n_total:.0f}%)")
    print(f"  Valid bearing errors               : {n_valid}/{n_total}")
    print(f"  Fetch failures (network)           : {n_fetch_fail}")

    if not valid:
        print("\n  No valid bearing measurements — cannot compute sigma_0.")
        return

    errors = [r.beta_error_deg for r in valid]
    abs_errors = [abs(e) for e in errors]
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
    mae = sum(abs_errors) / len(abs_errors)

    print(f"\n  Bearing error statistics:")
    print(f"    RMSE            : {rmse:.1f}°")
    print(f"    MAE             : {mae:.1f}°")
    print(f"    Mean (signed)   : {sum(errors)/len(errors):.1f}°")
    print(f"    Std dev         : {(sum((e - sum(errors)/len(errors))**2 for e in errors)/len(errors))**0.5:.1f}°")
    print(f"    Max abs error   : {max(abs_errors):.1f}°")

    # By direction
    print(f"\n  Per-direction (true bearing / mean abs error):")
    for direction, beta_true in [("N", 180.0), ("S", 0.0), ("E", 270.0), ("W", 90.0)]:
        d_valid = [r for r in valid if r.direction == direction]
        if d_valid:
            d_mae = sum(abs(r.beta_error_deg) for r in d_valid) / len(d_valid)
            print(f"    {direction} (β_true={beta_true:.0f}°): n={len(d_valid)}  MAE={d_mae:.1f}°")

    # Confidence bins — test 1/c^2 model
    sigma0 = get_config().measurement_noise.vlm_bearing_deg
    print(f"\n  Error vs confidence (tests 1/c^2 model, σ₀={sigma0}°):")
    bins = [(0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    for lo, hi in bins:
        b_valid = [r for r in valid if lo <= r.confidence < hi]
        if b_valid:
            b_mae = sum(abs(r.beta_error_deg) for r in b_valid) / len(b_valid)
            mid_c = (lo + hi) / 2
            sigma_model = sigma0 / mid_c**2
            print(f"    c∈[{lo:.2f},{hi:.2f}): n={len(b_valid):3d}  "
                  f"MAE={b_mae:.1f}°  σ_model({sigma0}°/c²)={sigma_model:.1f}°")

    # Verdict
    print(f"\n  Current config σ₀ = {sigma0}°.  Measured RMSE = {rmse:.1f}°.")
    if abs(rmse - sigma0) > 5.0:
        print(f"  *** MISMATCH > 5° — update config vlm_bearing_deg to {rmse:.1f}° ***")
    else:
        print(f"  Config σ₀ is within 5° of measured RMSE — no update required.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["detection", "bearing"], default="detection")
    p.add_argument("--n", type=int, default=None,
                   help="limit number of substations")
    p.add_argument("--zoom", type=int, default=18,
                   help="ArcGIS tile zoom for detection (18 ~ 0.6 m/px)")
    p.add_argument("--bearing-zoom", type=int, default=17,
                   help="ArcGIS tile zoom for bearing calibration (17 ~ 1.2 m/px)")
    p.add_argument("--offset", type=float, default=300.0,
                   help="Camera offset from substation in metres (Stage 2)")
    args = p.parse_args()

    if args.stage == "bearing":
        results = run_bearing_calibration(n=args.n,
                                          zoom=args.bearing_zoom,
                                          offset_m=args.offset)
        out_json = OUT_DIR / "bearing_calibration_results.json"
        with open(out_json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults written to: {out_json}")
        summarize_bearing(results)
        return

    results = run_detection(n=args.n, zoom=args.zoom)

    out_json = OUT_DIR / "detection_results.json"
    with open(out_json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults written to: {out_json}")

    summarize(results)


if __name__ == "__main__":
    main()
