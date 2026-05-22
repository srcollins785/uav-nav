"""
Integration Test: Celestial Image Pipeline (end-to-end)
========================================================
Tests the REAL OpenCV star-detection → Hungarian matching → inverse_celnav
solver pipeline.  This is the first time this code path has been exercised —
all scenario simulations use a synthetic shortcut (altaz_from_radec directly)
and never call detect_stars() or build_observations().

Pipeline under test:
  render_star_field()        (this module — Gaussian blobs via cv2)
  → detect_stars()           (uav_nav/vlm/star_detector.py — OpenCV)
  → build_observations()     (uav_nav/celestial/observation_builder.py)
  → get_celestial_fix_warmstart()  (uav_nav/celestial/solver.py)
  → CelestialFix.lat_deg / .lon_deg_east

Ground truth position: Atlanta Hartsfield-Jackson (33.641°N, 84.427°W)
Test UTC time: 2026-03-22 05:00:00 UTC  (midnight local time, clear sky)
Acceptance criterion: fix within 1.0° of true position (~111 km)

Notes
-----
* The observation_builder uses a FORWARD-FACING camera model — the vertical
  FOV determines which altitudes are visible.  We override vlm_camera_fov_deg
  to 120° for this test (VFOV ≈ 67°, covers Polaris at ~33.7° altitude).
* A 120° FOV is unrealistic for a real camera but valid for unit-testing the
  pipeline mathematics.  Narrow-FOV hardware testing is deferred to the DJI
  Matrice 300 RTK validation phase.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Output directory for rendered images
_RESULTS_DIR = Path(__file__).parent / "results"


def _est_timestamp() -> str:
    """Return current time as YYYYMMDD-HHMMSSest string."""
    est = datetime.now(tz=timezone(timedelta(hours=-5)))
    return est.strftime("%Y%m%d-%H%M%SEST")

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class CelestialPipelineResult:
    def __init__(self):
        self.passed: bool = False
        self.skipped: bool = False
        self.skip_reason: str = ""
        self.stars_rendered: int = 0
        self.stars_detected: int = 0
        self.observations_built: int = 0
        self.fix_lat: float = float("nan")
        self.fix_lon: float = float("nan")
        self.true_lat: float = 33.641
        self.true_lon: float = -84.427
        self.error_deg: float = float("nan")
        self.gdop: float = float("nan")
        self.rmse: float = float("nan")
        self.heading_used_deg: float = float("nan")
        self.fov_used_deg: float = float("nan")
        self.saved_images: List[str] = []
        self.message: str = ""

    def lat_error_deg(self) -> float:
        return abs(self.fix_lat - self.true_lat) if not math.isnan(self.fix_lat) else float("nan")

    def lon_error_deg(self) -> float:
        return abs(self.fix_lon - self.true_lon) if not math.isnan(self.fix_lon) else float("nan")

    def print_summary(self) -> None:
        status = "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
        print(f"\n{'='*60}")
        print(f"  Celestial Image Pipeline Test: {status}")
        print(f"{'='*60}")
        if self.skipped:
            print(f"  Reason: {self.skip_reason}")
            return
        print(f"  Camera FOV:         {self.fov_used_deg:.0f}°  heading={self.heading_used_deg:.0f}°")
        print(f"  Stars rendered:     {self.stars_rendered}")
        print(f"  Stars detected:     {self.stars_detected}")
        print(f"  Observations built: {self.observations_built}")
        if not math.isnan(self.fix_lat):
            print(f"  True position:      {self.true_lat:.4f}°N  {self.true_lon:.4f}°E")
            print(f"  Fix position:       {self.fix_lat:.4f}°N  {self.fix_lon:.4f}°E")
            print(f"  Error lat:          {self.lat_error_deg():.4f}° ({self.lat_error_deg()*111.0:.1f} km)")
            print(f"  Error lon:          {self.lon_error_deg():.4f}° ({self.lon_error_deg()*111.0*math.cos(math.radians(self.true_lat)):.1f} km)")
            print(f"  GDOP:               {self.gdop:.3f}°")
            print(f"  RMSE normalised:    {self.rmse:.4f}")
        if self.message:
            print(f"  Note: {self.message}")
        if self.saved_images:
            print(f"\n  Output images:")
            for p in self.saved_images:
                print(f"    {p}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Star-field renderer (cv2-based, no astropy)
# ---------------------------------------------------------------------------

def _render_star_field(
    img_w: int,
    img_h: int,
    star_positions: List[Tuple[float, float, float]],  # (px, py, magnitude)
) -> np.ndarray:
    """
    Paint stars as Gaussian blobs on a black background.

    Each star is rendered as a filled circle (radius=3px) then blurred
    with a 7×7 Gaussian kernel, producing a circular PSF suitable for
    SimpleBlobDetector.  Brightness is inversely scaled with magnitude.

    Returns a uint8 grayscale image.
    """
    import cv2

    img = np.zeros((img_h, img_w), dtype=np.float32)

    for px, py, mag in star_positions:
        ix, iy = int(round(px)), int(round(py))
        if not (3 <= ix < img_w - 3 and 3 <= iy < img_h - 3):
            continue
        # Magnitude 1 → intensity 230, magnitude 4 → intensity 100
        intensity = float(max(100, 230 - int((mag - 1.0) * 43)))
        cv2.circle(img, (ix, iy), 3, intensity, -1)

    # Gaussian PSF blur
    img = cv2.GaussianBlur(img, (7, 7), 1.5)

    # Very faint background noise helps Otsu find the bimodal boundary
    rng = np.random.default_rng(seed=0)
    img += rng.uniform(0.0, 2.0, img.shape).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Image output
# ---------------------------------------------------------------------------

def _save_images(
    raw_img: np.ndarray,
    star_pixels: List[Tuple[float, float, float]],
    observations: list,
    heading: float,
    fov_deg: float,
    fix_lat: float,
    fix_lon: float,
    true_lat: float,
    true_lon: float,
    dt_utc: datetime,
    result,          # CelestialPipelineResult — populated with saved paths
) -> None:
    """
    Save three images to integration_tests/results/:
      1. starfield_raw_<ts>.png       — rendered Gaussian blobs only
      2. starfield_annotated_<ts>.png — blobs + catalog labels + fix info
    """
    import cv2

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _est_timestamp()
    IMG_W = raw_img.shape[1]
    IMG_H = raw_img.shape[0]
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    px_per_deg = IMG_W / fov_deg

    # --- 1. Raw image ---
    raw_path = _RESULTS_DIR / f"starfield_raw_{ts}.png"
    cv2.imwrite(str(raw_path), raw_img)
    result.saved_images.append(str(raw_path))

    # --- 2. Annotated image ---
    ann = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

    # Horizon line
    cv2.line(ann, (0, int(cy)), (IMG_W, int(cy)), (60, 60, 60), 1)
    cv2.putText(ann, "horizon  alt=0", (8, int(cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

    # Star blobs: green circle + name label
    from uav_nav.celestial.catalog import get_catalog
    from uav_nav.celestial.solver import altaz_from_radec
    catalog = get_catalog()
    obs_names = {o.name for o in observations}

    for star in catalog:
        try:
            alt, az = altaz_from_radec(true_lat, true_lon, dt_utc, star.ra_deg, star.dec_deg)
        except Exception:
            continue
        rel_az = ((az - heading + 180.0) % 360.0) - 180.0
        if abs(rel_az) > fov_deg / 2.0:
            continue
        px = cx + rel_az * px_per_deg
        py = cy - alt * px_per_deg
        if not (0 <= px <= IMG_W and 0 <= py <= IMG_H):
            continue
        ix, iy = int(round(px)), int(round(py))
        matched = star.name in obs_names
        color = (0, 255, 80) if matched else (100, 100, 100)
        cv2.circle(ann, (ix, iy), 13, color, 1)
        label = f"{star.name}  {alt:.1f}\u00b0"
        cv2.putText(ann, label, (ix + 15, iy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 220, 255) if matched else (100, 100, 100),
                    1, cv2.LINE_AA)

    # Fix error annotation
    lat_err_km = abs(fix_lat - true_lat) * 111.0
    lon_err_km = abs(fix_lon - true_lon) * 111.0 * abs(math.cos(math.radians(true_lat)))
    status = "PASS" if (abs(fix_lat - true_lat) < 1.0 and abs(fix_lon - true_lon) < 1.0) else "FAIL"
    status_color = (0, 220, 0) if status == "PASS" else (0, 0, 220)

    lines = [
        f"Atlanta  {true_lat:.3f}N  {true_lon:.3f}E   heading={heading:.0f}deg   FOV={fov_deg:.0f}deg   {dt_utc.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Stars rendered: {len(star_pixels)}   detected: {len(star_pixels)}   matched: {len(observations)}",
        f"Fix: {fix_lat:.4f}N  {fix_lon:.4f}E     err lat={lat_err_km:.1f} km   lon={lon_err_km:.1f} km     [{status}]",
    ]
    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.rectangle(ann, (0, y - 16), (IMG_W, y + 6), (0, 0, 0), -1)
        col = status_color if i == 2 else (200, 200, 200)
        cv2.putText(ann, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # Legend
    cv2.circle(ann, (10, IMG_H - 30), 7, (0, 255, 80), 1)
    cv2.putText(ann, "matched to solver", (24, IMG_H - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
    cv2.circle(ann, (200, IMG_H - 30), 7, (100, 100, 100), 1)
    cv2.putText(ann, "catalog star (not matched)", (214, IMG_H - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

    ann_path = _RESULTS_DIR / f"starfield_annotated_{ts}.png"
    cv2.imwrite(str(ann_path), ann)
    result.saved_images.append(str(ann_path))


# ---------------------------------------------------------------------------
# Camera scan: find heading with most stars in frame
# ---------------------------------------------------------------------------

def _find_best_heading(
    true_lat: float,
    true_lon: float,
    dt_utc: datetime,
    fov_deg: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Scan 36 headings (0°…350°) and return the heading that puts the most
    catalog stars in the camera frame, plus the list of (px, py, magnitude)
    for those stars.
    """
    from uav_nav.celestial.catalog import get_catalog
    from uav_nav.celestial.solver import altaz_from_radec

    catalog = get_catalog()
    px_per_deg = img_w / fov_deg
    cx = img_w / 2.0
    cy = img_h / 2.0

    best_heading = 0.0
    best_stars: List[Tuple[float, float, float]] = []

    for hdg in range(0, 360, 10):
        visible = []
        for star in catalog:
            try:
                alt, az = altaz_from_radec(
                    true_lat, true_lon, dt_utc, star.ra_deg, star.dec_deg
                )
            except Exception:
                continue

            if alt < 5.0:
                continue  # below horizon or too low

            rel_az = ((az - hdg + 180.0) % 360.0) - 180.0
            if abs(rel_az) > fov_deg / 2.0:
                continue  # outside horizontal FOV

            px = cx + rel_az * px_per_deg
            py = cy - alt * px_per_deg  # camera looking horizontal: 0° alt = cy

            if 4 <= px <= img_w - 4 and 4 <= py <= img_h - 4:
                visible.append((float(px), float(py), float(star.magnitude)))

        if len(visible) > len(best_stars):
            best_stars = visible
            best_heading = float(hdg)

    return best_heading, best_stars


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def run() -> CelestialPipelineResult:
    result = CelestialPipelineResult()

    # --- Dependency checks ---
    try:
        import cv2  # noqa: F401
    except ImportError:
        result.skipped = True
        result.skip_reason = "opencv-python not installed (pip install opencv-python-headless)"
        return result

    try:
        from uav_nav.celestial.solver import is_available, get_celestial_fix_warmstart
        if not is_available():
            result.skipped = True
            result.skip_reason = "inverse_celnav module not found on sys.path"
            return result
    except Exception as exc:
        result.skipped = True
        result.skip_reason = f"Failed to import uav_nav.celestial.solver: {exc}"
        return result

    # --- Override config FOV for wide-angle test ---
    import uav_nav.config as cfg_module
    from uav_nav.config import Settings
    original_config = cfg_module._config
    test_fov = 120.0
    test_settings = Settings(vlm_camera_fov_deg=test_fov)
    cfg_module._config = test_settings

    try:
        return _run_test(result, test_fov)
    finally:
        cfg_module._config = original_config  # always restore


def _run_test(result: CelestialPipelineResult, fov_deg: float) -> CelestialPipelineResult:
    import cv2

    from uav_nav.celestial.observation_builder import build_observations
    from uav_nav.celestial.solver import get_celestial_fix_warmstart
    from uav_nav.navigation.state import UAVState
    from uav_nav.vlm.star_detector import detect_stars

    IMG_W, IMG_H = 1280, 720
    TRUE_LAT = 33.641
    TRUE_LON = -84.427
    DT_UTC = datetime(2026, 3, 22, 5, 0, 0, tzinfo=timezone.utc)

    result.fov_used_deg = fov_deg

    # --- 1. Find best heading ---
    heading, star_pixels = _find_best_heading(
        TRUE_LAT, TRUE_LON, DT_UTC, fov_deg, IMG_W, IMG_H
    )
    result.heading_used_deg = heading
    result.stars_rendered = len(star_pixels)

    if len(star_pixels) < 2:
        result.skipped = True
        result.skip_reason = (
            f"Fewer than 2 catalog stars visible in {fov_deg}° FOV at "
            f"ATL on 2026-03-22 05:00 UTC (any heading). "
            f"Found {len(star_pixels)}. Try a wider FOV or different epoch."
        )
        return result

    print(f"  Found {len(star_pixels)} stars in FOV at heading={heading:.0f}°")

    # --- 2. Render star field image ---
    star_img = _render_star_field(IMG_W, IMG_H, star_pixels)

    # --- 3. Run OpenCV star detector ---
    candidates = detect_stars(star_img, max_stars=20)
    result.stars_detected = len(candidates)

    if len(candidates) == 0:
        result.passed = False
        result.message = (
            "detect_stars() returned 0 candidates from rendered image. "
            "Check SimpleBlobDetector parameters or increase star brightness."
        )
        return result

    print(f"  detect_stars() found {len(candidates)} candidate(s)")

    # --- 4. Build observations (Hungarian matching) ---
    state = UAVState(
        lat_deg=TRUE_LAT,
        lon_deg=TRUE_LON,
        heading_deg=heading,
    )
    observations = build_observations(
        candidates=candidates,
        state=state,
        dt_utc=DT_UTC,
        image_width_px=IMG_W,
        image_height_px=IMG_H,
    )
    result.observations_built = len(observations)

    if len(observations) == 0:
        result.passed = False
        result.message = (
            "build_observations() produced 0 matched observations. "
            "Hungarian matching may have failed — detected centroids did not "
            "fall within 50px of projected catalog positions."
        )
        return result

    print(f"  build_observations() matched {len(observations)} observation(s)")
    for obs in observations:
        print(f"    {obs.name:12s}  alt={obs.alt_deg:.2f}°  az={obs.az_deg:.2f}°")

    # --- 5. Celestial fix ---
    fix = get_celestial_fix_warmstart(
        observations=observations,
        dt_utc=DT_UTC,
        init_lat_deg=TRUE_LAT,
        init_lon_deg=TRUE_LON,
        use_azimuth=True,
        search_radius_deg=2.0,
    )

    if fix is None:
        result.passed = False
        result.message = "get_celestial_fix_warmstart() returned None — solver failed."
        return result

    result.fix_lat = fix.lat_deg
    result.fix_lon = fix.lon_deg_east
    result.gdop = fix.position_gdop
    result.rmse = fix.rmse_normalized

    lat_err = abs(fix.lat_deg - TRUE_LAT)
    lon_err = abs(fix.lon_deg_east - TRUE_LON)
    result.error_deg = math.hypot(lat_err, lon_err)

    ACCEPT_DEG = 1.0  # ~111 km — generous for first real-pipeline run
    result.passed = (lat_err < ACCEPT_DEG and lon_err < ACCEPT_DEG and fix.optimizer_success)

    if not fix.optimizer_success:
        result.message = f"Solver reported non-convergence (rmse={fix.rmse_normalized:.4f})"

    # --- 6. Save rendered images ---
    _save_images(
        raw_img=star_img,
        star_pixels=star_pixels,
        observations=observations,
        heading=heading,
        fov_deg=fov_deg,
        fix_lat=fix.lat_deg,
        fix_lon=fix.lon_deg_east,
        true_lat=TRUE_LAT,
        true_lon=TRUE_LON,
        dt_utc=DT_UTC,
        result=result,
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
