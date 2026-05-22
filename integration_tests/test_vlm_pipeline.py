"""
Integration Test: VLM Pipeline (BakLLaVA + real aerial imagery)
================================================================
Tests the REAL VLM pipeline against a satellite image of a known
transformer substation downloaded from the ESRI World Imagery service
(public, no API key required).

Pipeline under test:
  download_substation_tile()  (this module — ESRI WMTS tile)
  → VLMClient.query()         (uav_nav/vlm/client.py → Ollama → BakLLaVA)
  → SemanticObservation       (uav_nav/vlm/schemas.py)

Substation under test:
  Georgia Power Plant Wansley Substation
  Latitude: 33.581°N   Longitude: 84.727°W
  (on the ATL→Macon baseline scenario route — a valid navigation target)
  Tile zoom 17, ESRI World Imagery service

Prerequisites:
  1. Ollama installed and running: https://ollama.com/download
  2. BakLLaVA pulled:  ollama pull bakllava

This test SKIPS gracefully if Ollama is unreachable.

Acceptance criteria:
  - VLMClient.query() returns a SemanticObservation (not None)
  - The response JSON is valid (parser succeeded)
  - At least one landmark is present OR sky_visible is True
  - Confidence of best landmark >= 0.3 (non-trivial response, not pure guess)
  - bearing_deg is in [0, 360) range
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class VLMPipelineResult:
    def __init__(self):
        self.passed: bool = False
        self.skipped: bool = False
        self.skip_reason: str = ""
        self.ollama_available: bool = False
        self.image_downloaded: bool = False
        self.image_source: str = ""
        self.raw_response: str = ""
        self.landmark_type: str = ""
        self.bearing_deg: float = float("nan")
        self.confidence: float = float("nan")
        self.scene_type: str = ""
        self.num_landmarks: int = 0
        self.message: str = ""

    def print_summary(self) -> None:
        status = "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
        print(f"\n{'='*60}")
        print(f"  VLM Pipeline Test: {status}")
        print(f"{'='*60}")
        if self.skipped:
            print(f"  Reason: {self.skip_reason}")
            if not self.ollama_available:
                print()
                print("  To enable this test:")
                print("    1. Install Ollama:  https://ollama.com/download")
                print("    2. Pull BakLLaVA:   ollama pull bakllava")
                print("    3. Start server:    ollama serve")
            return
        print(f"  Image source:       {self.image_source}")
        print(f"  Scene type:         {self.scene_type}")
        print(f"  Landmarks found:    {self.num_landmarks}")
        if self.num_landmarks > 0:
            print(f"  Best landmark:      {self.landmark_type}")
            print(f"  Bearing:            {self.bearing_deg:.1f}°")
            print(f"  Confidence:         {self.confidence:.2f}")
        if self.message:
            print(f"  Note: {self.message}")
        if self.raw_response:
            preview = self.raw_response[:200].replace("\n", " ")
            print(f"  Raw response:       {preview}...")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Tile download
# ---------------------------------------------------------------------------

def _lat_lon_to_tile(lat_deg: float, lon_deg: float, zoom: int):
    """Convert WGS-84 lat/lon to slippy-map tile x, y at given zoom."""
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    lat_r = math.radians(lat_deg)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _download_esri_tile(lat_deg: float, lon_deg: float, zoom: int = 17) -> Optional[bytes]:
    """
    Download a single tile from ESRI World Imagery (public, no API key).
    Returns raw PNG bytes or None on failure.
    """
    import requests

    x, y = _lat_lon_to_tile(lat_deg, lon_deg, zoom)
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        print(f"  WARNING: Could not download ESRI tile: {exc}")
        return None


def _make_synthetic_substation_image() -> bytes:
    """
    Generate a simple synthetic aerial image that resembles an industrial
    substation layout: gray transformer pads on a green field.
    Used as fallback when the network tile download fails.

    Note: BakLLaVA's response to this synthetic image is not a reliable
    test of real-world recognition — it only validates pipeline plumbing.
    """
    import cv2

    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Green field background
    img[:, :] = (34, 100, 34)

    # Gray transformer pads (rectangular equipment layout)
    cv2.rectangle(img, (60, 60), (196, 196), (120, 120, 120), -1)  # main yard
    cv2.rectangle(img, (70, 70), (130, 110), (80, 80, 80), -1)      # transformer 1
    cv2.rectangle(img, (140, 70), (190, 110), (80, 80, 80), -1)     # transformer 2
    cv2.rectangle(img, (70, 130), (190, 185), (90, 90, 90), -1)     # switchyard

    # Access road
    cv2.rectangle(img, (115, 0), (140, 60), (150, 140, 100), -1)

    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def run() -> VLMPipelineResult:
    result = VLMPipelineResult()

    # --- Check Ollama ---
    try:
        from uav_nav.vlm.client import VLMClient, VLMUnavailableError
        client = VLMClient()
        result.ollama_available = client.is_available()
    except Exception as exc:
        result.skipped = True
        result.skip_reason = f"Could not import VLMClient: {exc}"
        return result

    if not result.ollama_available:
        result.skipped = True
        result.skip_reason = (
            "Ollama server not reachable at http://localhost:11434. "
            "Install Ollama and pull BakLLaVA to run this test."
        )
        return result

    # --- Get aerial image ---
    # Georgia Power Plant Wansley substation (on the ATL-MCN route)
    SUBSTATION_LAT = 33.581
    SUBSTATION_LON = -84.727

    print("  Downloading ESRI World Imagery tile for Wansley substation...")
    image_bytes = _download_esri_tile(SUBSTATION_LAT, SUBSTATION_LON, zoom=17)

    if image_bytes is not None:
        result.image_downloaded = True
        result.image_source = (
            f"ESRI World Imagery z17 tile at {SUBSTATION_LAT}°N, {SUBSTATION_LON}°W "
            f"(Wansley substation, GA)"
        )
        print(f"  Downloaded {len(image_bytes):,} bytes")
    else:
        image_bytes = _make_synthetic_substation_image()
        result.image_source = "Synthetic fallback image (network unavailable)"
        result.message = (
            "ESRI tile download failed — using synthetic image. "
            "VLM landmark recognition result is not meaningful on synthetic data."
        )
        print("  Using synthetic substation image as fallback")

    # --- Run VLM ---
    print(f"  Querying BakLLaVA ({client.model})...")
    try:
        obs = client.query(image_bytes)
    except Exception as exc:
        result.passed = False
        result.message = f"VLMClient.query() raised exception: {exc}"
        return result

    if obs is None:
        result.passed = False
        result.message = (
            "VLMClient.query() returned None — BakLLaVA response could not be parsed. "
            "Check that bakllava model is properly pulled (ollama pull bakllava)."
        )
        return result

    # --- Validate response ---
    result.scene_type = obs.scene_type if hasattr(obs, "scene_type") else "unknown"
    result.num_landmarks = len(obs.landmarks) if hasattr(obs, "landmarks") and obs.landmarks else 0

    if result.num_landmarks > 0:
        # Pick landmark with highest confidence
        best = max(obs.landmarks, key=lambda lm: lm.confidence)
        result.landmark_type = best.landmark_type
        result.bearing_deg = best.bearing_deg
        result.confidence = best.confidence

    # Acceptance: valid parse + at least one landmark or sky visible + bearing in range
    has_content = result.num_landmarks > 0 or (hasattr(obs, "sky_visible") and obs.sky_visible)
    bearing_valid = math.isnan(result.bearing_deg) or (0.0 <= result.bearing_deg < 360.0)
    confidence_ok = math.isnan(result.confidence) or result.confidence >= 0.3

    result.passed = has_content and bearing_valid and confidence_ok

    if not has_content:
        result.message = (
            "BakLLaVA returned an empty landmarks list and sky_visible=False. "
            "Model may not recognise aerial substation imagery. "
            "Consider fine-tuning or providing annotated training crops."
        )

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    r = run()
    r.print_summary()
    sys.exit(0 if (r.passed or r.skipped) else 1)
