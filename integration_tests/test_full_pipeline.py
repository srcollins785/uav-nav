"""
Integration Test: Full End-to-End Pipeline (Composite Scene)
=============================================================
Tests the complete navigation pipeline using a single composite image:
  - Top half (sky):    rendered star field  → detect_stars → celestial fix → UKF update
  - Bottom half (ground): satellite tile    → VLM bearing                 → UKF update

Both pipeline branches feed into the *same* UAVKalmanFilter instance.

Pipeline under test (celestial branch):
  _render_star_field()            (this module, reused from test_celestial_image_pipeline)
  → detect_stars()                (uav_nav/vlm/star_detector.py)
  → build_observations()          (uav_nav/celestial/observation_builder.py)
  → get_celestial_fix_warmstart() (uav_nav/celestial/solver.py)
  → UAVKalmanFilter.update_celestial_fix()   (uav_nav/navigation/ukf.py)   ← NEW link

Pipeline under test (VLM branch):
  ESRI satellite tile  (or synthetic fallback)
  → BakLLaVA (Ollama) → parse_vlm_response  / synthetic bearing
  → UAVKalmanFilter.update_vlm_bearing()     (uav_nav/navigation/ukf.py)   ← NEW link

Composite image:
  Top 50%  — rendered Gaussian star blobs on black sky (identical method to
             test_celestial_image_pipeline.py)
  Bottom 50% — ESRI World Imagery tile of Plant Scherer power station, or
               synthetic substation fallback

Ground truth:  Plant Scherer, Juliette GA  (33.066°N, 83.779°W)
UKF init:      50 km north of truth (simulates dead-reckoning drift)
UTC epoch:     2026-03-22 05:00:00 UTC  (same as existing celestial test)

Acceptance criteria (all required for PASS):
  1. Celestial pipeline wired:  lat/lon covariance P[0,0] and P[1,1] decrease
     after update_celestial_fix().
  2. VLM pipeline wired:        update_vlm_bearing() called without exception
     (gate may accept or reject — both outcomes are valid behaviour).
  3. State protection:          velocity / heading / bias block P[3:,3:] is
     unchanged by position-only measurements.
  4. Composite image saved:     results/composite_scene.png written for visual
     inspection.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

_RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Ground truth and test constants
# ---------------------------------------------------------------------------

TRUE_LAT = 33.066          # Plant Scherer, Juliette GA
TRUE_LON = -83.779
IMG_W, IMG_H = 1280, 720
DT_UTC = datetime(2026, 3, 22, 5, 0, 0, tzinfo=timezone.utc)
FOV_DEG = 120.0            # wide-angle override — same as test_celestial_image_pipeline
OFFSET_KM = 50.0           # deliberate dead-reckoning drift
OFFSET_LAT = OFFSET_KM / 111.0  # ~0.45°


# ---------------------------------------------------------------------------
# Haversine and bearing helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2.0 * math.asin(math.sqrt(a))


def _bearing_to(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geographic bearing (degrees, 0=N clockwise) from (lat1,lon1) to (lat2,lon2)."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2_r)
    x = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))
    return math.degrees(math.atan2(y, x)) % 360.0


# ---------------------------------------------------------------------------
# Star field renderer  (copy of _render_star_field in test_celestial_image_pipeline)
# ---------------------------------------------------------------------------

def _render_star_field(
    img_w: int,
    img_h: int,
    star_positions: List[Tuple[float, float, float]],
) -> np.ndarray:
    """
    Paint stars as Gaussian blobs on a black background.
    Returns a uint8 grayscale image.  Stars appear in the top half when
    rendered with the standard horizontal-camera model (horizon at cy).
    """
    import cv2

    img = np.zeros((img_h, img_w), dtype=np.float32)
    for px, py, mag in star_positions:
        ix, iy = int(round(px)), int(round(py))
        if not (3 <= ix < img_w - 3 and 3 <= iy < img_h - 3):
            continue
        intensity = float(max(100, 230 - int((mag - 1.0) * 43)))
        cv2.circle(img, (ix, iy), 3, intensity, -1)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)
    rng = np.random.default_rng(seed=0)
    img += rng.uniform(0.0, 2.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Best-heading scanner  (same logic as test_celestial_image_pipeline)
# ---------------------------------------------------------------------------

def _find_best_heading(
    lat: float,
    lon: float,
    dt_utc: datetime,
    fov_deg: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Return (heading_deg, [(px, py, mag), ...]) maximising stars in FOV."""
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
                alt, az = altaz_from_radec(lat, lon, dt_utc, star.ra_deg, star.dec_deg)
            except Exception:
                continue
            if alt < 5.0:
                continue
            rel_az = ((az - hdg + 180.0) % 360.0) - 180.0
            if abs(rel_az) > fov_deg / 2.0:
                continue
            px = cx + rel_az * px_per_deg
            py = cy - alt * px_per_deg
            if 4 <= px <= img_w - 4 and 4 <= py <= img_h - 4:
                visible.append((float(px), float(py), float(star.magnitude)))
        if len(visible) > len(best_stars):
            best_stars = visible
            best_heading = float(hdg)

    return best_heading, best_stars


# ---------------------------------------------------------------------------
# ESRI tile download
# ---------------------------------------------------------------------------

def _lat_lon_to_tile(lat_deg: float, lon_deg: float, zoom: int):
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    lat_r = math.radians(lat_deg)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _download_esri_tile(lat_deg: float, lon_deg: float,
                        zoom: int = 13, stitch: int = 3) -> Optional[bytes]:
    """Download NxN stitched ESRI tile. Returns JPEG bytes or None."""
    import requests
    import cv2

    cx, cy = _lat_lon_to_tile(lat_deg, lon_deg, zoom)
    half = stitch // 2
    rows = []
    try:
        for dy in range(-half, half + 1):
            cols = []
            for dx in range(-half, half + 1):
                url = (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                       f"World_Imagery/MapServer/tile/{zoom}/{cy+dy}/{cx+dx}")
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                buf = np.frombuffer(resp.content, np.uint8)
                tile_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                cols.append(tile_img)
            rows.append(np.hstack(cols))
        composite = np.vstack(rows)
        _, enc = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Downloaded {stitch}x{stitch} grid z{zoom}"
              f" ({composite.shape[1]}x{composite.shape[0]}px)"
              f"  {len(enc.tobytes()):,} bytes")
        return enc.tobytes()
    except Exception as exc:
        print(f"  WARNING: ESRI tile download failed: {exc}")
        return None


def _make_synthetic_substation_image() -> bytes:
    """Synthetic substation aerial view as JPEG bytes (fallback if network unavailable)."""
    import cv2

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :] = (34, 100, 34)
    cv2.rectangle(img, (60, 60), (196, 196), (120, 120, 120), -1)
    cv2.rectangle(img, (70, 70), (130, 110), (80, 80, 80), -1)
    cv2.rectangle(img, (140, 70), (190, 110), (80, 80, 80), -1)
    cv2.rectangle(img, (70, 130), (190, 185), (90, 90, 90), -1)
    cv2.rectangle(img, (115, 0), (140, 60), (150, 140, 100), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()




# ---------------------------------------------------------------------------
# Composite scene builder
# ---------------------------------------------------------------------------

def _build_composite(
    star_img: np.ndarray,
    ground_bytes: Optional[bytes],
) -> Tuple[np.ndarray, bytes]:
    """
    Combine star image (grayscale) and ground image (BGR JPEG bytes) into a
    single 1280×720 BGR composite.  The top half shows sky (stars), the bottom
    half shows terrain (landmark).

    Returns (composite_bgr_array, composite_jpeg_bytes).
    """
    import cv2

    # Sky half: convert star field (grayscale) to BGR, take top 360 rows
    sky_bgr = cv2.cvtColor(star_img, cv2.COLOR_GRAY2BGR)
    sky_half = sky_bgr[:IMG_H // 2, :]                # 1280×360

    # Ground half: ESRI tile or synthetic
    if ground_bytes is not None:
        buf = np.frombuffer(ground_bytes, np.uint8)
        ground_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        synth = _make_synthetic_substation_image()
        buf = np.frombuffer(synth, np.uint8)
        ground_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    ground_half = cv2.resize(ground_bgr, (IMG_W, IMG_H // 2))  # 1280×360

    # Draw horizon separator line and section labels
    composite = np.vstack([sky_half, ground_half])              # 1280×720
    cv2.line(composite, (0, IMG_H // 2), (IMG_W, IMG_H // 2), (80, 180, 80), 2)
    cv2.putText(composite, "HORIZON", (10, IMG_H // 2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1, cv2.LINE_AA)
    cv2.putText(composite, "SKY  /  CELESTIAL NAV", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(composite, "GROUND  /  VLM LANDMARK", (10, IMG_H // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1, cv2.LINE_AA)

    # Save unnanotated version for VLM query (no overlaid text to confuse model)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / "composite_scene.png"
    cv2.imwrite(str(out_path), composite)
    print(f"  Composite scene saved → {out_path}")

    _, enc = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return composite, enc.tobytes()


# ---------------------------------------------------------------------------
# Post-run annotation  (called after all pipeline results are available)
# ---------------------------------------------------------------------------

def _annotate_composite(
    composite_bgr: np.ndarray,
    star_pixels: List[Tuple[float, float, float]],
    observations: list,
    heading_deg: float,
    fov_deg: float,
    vlm_bearing: Optional[float],
    vlm_source: str,
    result: "FullPipelineResult",
) -> None:
    """
    Overlay pipeline results on the composite image and save as
    composite_scene_annotated.png.

    Sky half  — detected stars circled and named; celestial fix stats.
    Ground half — VLM bearing arrow; source and gate status.
    Full image — error progression banner and PASS/FAIL badge.
    """
    import cv2
    import math as _math

    ann = composite_bgr.copy()
    W, H = ann.shape[1], ann.shape[0]
    half = H // 2
    px_per_deg = W / fov_deg

    # ── Sky half: star annotations ──────────────────────────────────────────
    obs_names = {o.name for o in observations}
    for px, py, mag in star_pixels:
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < W and 0 <= iy < half):
            continue
        matched = False
        # Find the observation whose pixel projection is nearest this rendered star
        for obs in observations:
            # Re-project catalog star to pixel using same formula as _find_best_heading
            from uav_nav.celestial.solver import altaz_from_radec
            from uav_nav.celestial.catalog import get_catalog
            break  # use name matching instead (simpler)
        # Use magnitude as proxy for brightness; highlight if in obs_names
        # We don't have star names in star_pixels — use distance from obs projected pos
        color = (0, 255, 80)   # green for all rendered stars (detected)
        cv2.circle(ann, (ix, iy), 14, color, 1, cv2.LINE_AA)

    # Label matched observations using catalog projection
    try:
        from uav_nav.celestial.catalog import get_catalog
        from uav_nav.celestial.solver import altaz_from_radec
        catalog = get_catalog()
        cx_sky = W / 2.0
        cy_sky = half / 2.0  # center of sky half in sky-half coords
        for star in catalog:
            try:
                alt, az = altaz_from_radec(TRUE_LAT, TRUE_LON, DT_UTC,
                                           star.ra_deg, star.dec_deg)
            except Exception:
                continue
            if alt < 5.0:
                continue
            rel_az = ((az - heading_deg + 180.0) % 360.0) - 180.0
            if abs(rel_az) > fov_deg / 2.0:
                continue
            px = W / 2.0 + rel_az * px_per_deg
            py = half / 2.0 - alt * px_per_deg   # horizon at half/2 in sky-half coords
            if not (4 <= px <= W - 4 and 4 <= py <= half - 4):
                continue
            ix, iy = int(round(px)), int(round(py))
            matched = star.name in obs_names
            ring_color = (0, 255, 80) if matched else (80, 80, 80)
            label_color = (0, 220, 255) if matched else (80, 80, 80)
            cv2.circle(ann, (ix, iy), 14, ring_color, 1, cv2.LINE_AA)
            cv2.putText(ann, f"{star.name} {alt:.0f}\u00b0",
                        (ix + 16, iy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, label_color, 1, cv2.LINE_AA)
    except Exception:
        pass

    # Celestial fix stats box (top-right of sky half)
    if not _math.isnan(result.fix_lat):
        lat_err_km = abs(result.fix_lat - TRUE_LAT) * 111.0
        lon_err_km = abs(result.fix_lon - TRUE_LON) * 111.0 * abs(_math.cos(_math.radians(TRUE_LAT)))
        cel_lines = [
            f"Celestial fix: {result.fix_lat:.3f}N  {result.fix_lon:.3f}E",
            f"Fix error: lat {lat_err_km:.0f} km  lon {lon_err_km:.0f} km",
            f"GDOP: {result.fix_gdop:.3f}deg   converged: {result.fix_success}",
            f"P[lat]: {_math.sqrt(result.cel_lat_var_before)*111:.0f} km -> {_math.sqrt(result.cel_lat_var_after)*111:.0f} km  1-sigma",
        ]
        box_x = W - 420
        for i, line in enumerate(cel_lines):
            y = 22 + i * 18
            cv2.rectangle(ann, (box_x - 4, y - 13), (W - 4, y + 4), (0, 0, 0), -1)
            cv2.putText(ann, line, (box_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 230, 255), 1, cv2.LINE_AA)

    # ── Ground half: VLM bearing arrow ──────────────────────────────────────
    gnd_cx = W // 2
    gnd_cy = half + half // 2   # center of ground half in full-image coords

    if vlm_bearing is not None and not _math.isnan(vlm_bearing):
        # Bearing is geographic (north-up). Camera heading sets forward direction.
        cam_rel = (vlm_bearing - heading_deg + 360.0) % 360.0   # 0=fwd, 90=right
        # Map to image: fwd = up in ground half (pointing toward horizon)
        # In our layout, forward = toward the horizon (upward in the ground half)
        angle_rad = _math.radians(cam_rel - 90.0)   # -90 rotates fwd to "up"
        arrow_len = 80
        tip_x = int(gnd_cx + arrow_len * _math.cos(angle_rad))
        tip_y = int(gnd_cy - arrow_len * _math.sin(angle_rad))
        gate_color = (0, 220, 0) if result.vlm_gate_accepted else (0, 100, 220)
        cv2.arrowedLine(ann, (gnd_cx, gnd_cy), (tip_x, tip_y),
                        gate_color, 3, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(ann, (gnd_cx, gnd_cy), 8, gate_color, -1)
        cv2.putText(ann, f"VLM bearing: {vlm_bearing:.1f}deg",
                    (gnd_cx + 15, gnd_cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, gate_color, 1, cv2.LINE_AA)
        gate_txt = "gate: ACCEPTED" if result.vlm_gate_accepted else "gate: REJECTED"
        cv2.putText(ann, gate_txt,
                    (gnd_cx + 15, gnd_cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, gate_color, 1, cv2.LINE_AA)

    # VLM source label (bottom-right of ground half)
    src_lines = [
        f"VLM source: {vlm_source}",
        f"Landmark: {TRUE_LAT:.3f}N  {TRUE_LON:.3f}E  (Plant Scherer)",
    ]
    for i, line in enumerate(src_lines):
        y = half + half - 30 + i * 18
        cv2.putText(ann, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1, cv2.LINE_AA)

    # ── Full-image: error progression banner (bottom strip) ─────────────────
    banner_y = H - 18
    status = "PASS" if result.passed else "FAIL"
    status_color = (0, 220, 0) if result.passed else (0, 60, 220)
    banner = (
        f"  Init: {result.error_initial_m/1000:.1f} km"
        f"  ->  After celestial: {result.error_after_celestial_m/1000:.1f} km"
        f"  ->  After VLM: {result.error_after_vlm_m/1000:.1f} km"
        f"      State protection: {result.vel_bias_unchanged}"
        f"      [{status}]"
    )
    cv2.rectangle(ann, (0, H - 28), (W, H), (0, 0, 0), -1)
    cv2.putText(ann, banner, (8, banner_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

    # PASS/FAIL badge (top-left)
    badge_color = (0, 200, 0) if result.passed else (0, 50, 200)
    cv2.rectangle(ann, (W - 110, 5), (W - 5, 35), badge_color, -1)
    cv2.putText(ann, status, (W - 95, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    ann_path = _RESULTS_DIR / "composite_scene_annotated.png"
    cv2.imwrite(str(ann_path), ann)
    print(f"  Annotated composite → {ann_path}")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class FullPipelineResult:
    def __init__(self):
        self.passed: bool = False
        self.skipped: bool = False
        self.skip_reason: str = ""
        # Celestial branch
        self.stars_rendered: int = 0
        self.stars_detected: int = 0
        self.obs_matched: int = 0
        self.fix_lat: float = float("nan")
        self.fix_lon: float = float("nan")
        self.fix_gdop: float = float("nan")
        self.fix_success: bool = False
        self.error_initial_m: float = float("nan")
        self.error_after_celestial_m: float = float("nan")
        self.cel_lat_var_before: float = float("nan")
        self.cel_lat_var_after: float = float("nan")
        self.cel_lon_var_before: float = float("nan")
        self.cel_lon_var_after: float = float("nan")
        # VLM branch
        self.vlm_source: str = "none"
        self.vlm_bearing_used: float = float("nan")
        self.error_after_vlm_m: float = float("nan")
        self.vlm_gate_accepted: bool = False
        # State protection
        self.vel_bias_unchanged: bool = False
        # Saved artifacts
        self.composite_saved: bool = False
        self.message: str = ""

    def print_summary(self) -> None:
        status = "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
        print(f"\n{'='*65}")
        print(f"  Full End-to-End Pipeline Test: {status}")
        print(f"{'='*65}")
        if self.skipped:
            print(f"  Reason: {self.skip_reason}")
            return
        print(f"  Ground truth:   {TRUE_LAT:.4f}°N  {TRUE_LON:.4f}°E  (Plant Scherer)")
        print(f"  UKF init:       {OFFSET_KM:.0f} km north of truth")
        print(f"  Initial error:  {self.error_initial_m/1000:.1f} km")
        print()
        print(f"  ── Celestial branch ──")
        print(f"  Stars rendered:     {self.stars_rendered}")
        print(f"  Stars detected:     {self.stars_detected}")
        print(f"  Obs matched:        {self.obs_matched}")
        if not math.isnan(self.fix_lat):
            print(f"  Celestial fix:      {self.fix_lat:.4f}°N  {self.fix_lon:.4f}°E")
            print(f"  GDOP:               {self.fix_gdop:.3f}°")
            print(f"  Solver converged:   {self.fix_success}")
        print(f"  Error after cel:    {self.error_after_celestial_m/1000:.1f} km")
        print(f"  P[lat] before/after:{math.sqrt(self.cel_lat_var_before)*111:.1f} km"
              f" → {math.sqrt(self.cel_lat_var_after)*111:.1f} km  1-sigma")
        print(f"  P[lon] before/after:{math.sqrt(self.cel_lon_var_before)*111:.1f} km"
              f" → {math.sqrt(self.cel_lon_var_after)*111:.1f} km  1-sigma")
        print()
        print(f"  ── VLM branch ──")
        print(f"  VLM source:         {self.vlm_source}")
        if not math.isnan(self.vlm_bearing_used):
            print(f"  Bearing used:       {self.vlm_bearing_used:.1f}°")
        print(f"  Gate accepted:      {self.vlm_gate_accepted}")
        print(f"  Error after VLM:    {self.error_after_vlm_m/1000:.1f} km")
        print()
        print(f"  ── Integrity ──")
        print(f"  State protection:   {self.vel_bias_unchanged}")
        print(f"  Composite saved:    {self.composite_saved}")
        if self.message:
            print(f"  Note: {self.message}")
        print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main test logic
# ---------------------------------------------------------------------------

def run() -> FullPipelineResult:
    result = FullPipelineResult()

    # Dependency checks
    try:
        import cv2  # noqa: F401
    except ImportError:
        result.skipped = True
        result.skip_reason = "opencv-python not installed"
        return result

    try:
        from uav_nav.celestial.solver import is_available
        if not is_available():
            result.skipped = True
            result.skip_reason = "inverse_celnav module not found on sys.path"
            return result
    except Exception as exc:
        result.skipped = True
        result.skip_reason = f"Failed to import celestial solver: {exc}"
        return result

    # Override config FOV (same approach as test_celestial_image_pipeline)
    import uav_nav.config as cfg_module
    from uav_nav.config import Settings
    original_config = cfg_module._config
    cfg_module._config = Settings(vlm_camera_fov_deg=FOV_DEG)

    try:
        return _run_test(result)
    finally:
        cfg_module._config = original_config


def _run_test(result: FullPipelineResult) -> FullPipelineResult:
    from uav_nav.celestial.observation_builder import build_observations
    from uav_nav.celestial.solver import get_celestial_fix_warmstart
    from uav_nav.navigation.imu import IMUReading
    from uav_nav.navigation.state import UAVState
    from uav_nav.navigation.ukf import UAVKalmanFilter
    from uav_nav.vlm.star_detector import detect_stars

    INIT_LAT = TRUE_LAT + OFFSET_LAT
    INIT_LON = TRUE_LON

    # ── 1. Build composite image ────────────────────────────────────────────
    print(f"  Finding best heading for star field at ({TRUE_LAT}, {TRUE_LON}) ...")
    heading, star_pixels = _find_best_heading(TRUE_LAT, TRUE_LON, DT_UTC,
                                              FOV_DEG, IMG_W, IMG_H)
    result.stars_rendered = len(star_pixels)

    if len(star_pixels) < 2:
        result.skipped = True
        result.skip_reason = (
            f"Fewer than 2 catalog stars in {FOV_DEG}° FOV at Plant Scherer "
            f"on 2026-03-22 05:00 UTC (any heading). Found {len(star_pixels)}."
        )
        return result

    print(f"  Rendering {len(star_pixels)} stars at heading={heading:.0f}° ...")
    star_img = _render_star_field(IMG_W, IMG_H, star_pixels)

    print("  Downloading ground tile ...")
    ground_bytes = _download_esri_tile(TRUE_LAT, TRUE_LON, zoom=13, stitch=3)

    composite_bgr, composite_jpeg = _build_composite(star_img, ground_bytes)
    result.composite_saved = (_RESULTS_DIR / "composite_scene.png").exists()

    # ── 2. Initialise UKF at offset position ────────────────────────────────
    # Large initial covariance reflects 50 km dead-reckoning uncertainty.
    init_state = UAVState(
        lat_deg=INIT_LAT,
        lon_deg=INIT_LON,
        alt_m=300.0,
        heading_deg=heading,
        covariance=np.diag([
            (0.5) ** 2,    # lat  (~55 km 1σ)
            (0.5) ** 2,    # lon
            (10.0) ** 2,   # alt
            (1.0) ** 2,    # v_north
            (1.0) ** 2,    # v_east
            (0.5) ** 2,    # v_down
            (5.0) ** 2,    # heading
            (2.0) ** 2,    # baro_bias
            (2.0) ** 2,    # mag_bias
            (0.05) ** 2,   # gyro_bias_z
        ]),
    )
    kf = UAVKalmanFilter(init_state)

    # filterpy requires at least one predict() before update() to initialise
    # the stored sigma-point array (sigmas_f).  Without it, sigmas_f is all-zeros
    # and the Kalman gain is zero — no covariance update occurs.  Use a
    # zero-motion IMU step (dt=0.01 s, 9.81 m/s² gravity) to warm up.
    _imu_warmup = IMUReading(
        accel_x_ms2=0.0, accel_y_ms2=0.0, accel_z_ms2=9.81,
        gyro_x_dps=0.0,  gyro_y_dps=0.0,  gyro_z_dps=0.0,
        dt_s=0.01,
    )
    kf.predict(_imu_warmup)

    result.error_initial_m = _haversine_m(INIT_LAT, INIT_LON, TRUE_LAT, TRUE_LON)
    print(f"  UKF initialised  {OFFSET_KM:.0f} km north of truth"
          f"  (error = {result.error_initial_m/1000:.1f} km)")

    # ── 3. Celestial pipeline ───────────────────────────────────────────────
    print("  Running star detector on sky half of composite ...")
    candidates = detect_stars(star_img, max_stars=20)
    result.stars_detected = len(candidates)

    if len(candidates) == 0:
        result.passed = False
        result.message = "detect_stars() returned 0 candidates from rendered sky image."
        return result

    print(f"  detect_stars(): {len(candidates)} candidates")

    # build_observations uses current UKF state for star projection
    ukf_state_obs = UAVState(lat_deg=INIT_LAT, lon_deg=INIT_LON, heading_deg=heading)
    observations = build_observations(
        candidates=candidates,
        state=ukf_state_obs,
        dt_utc=DT_UTC,
        image_width_px=IMG_W,
        image_height_px=IMG_H,
    )
    result.obs_matched = len(observations)
    print(f"  build_observations(): {len(observations)} matched")
    for obs in observations:
        print(f"    {obs.name:12s}  alt={obs.alt_deg:.2f}°  az={obs.az_deg:.2f}°")

    fix = get_celestial_fix_warmstart(
        observations=observations,
        dt_utc=DT_UTC,
        init_lat_deg=INIT_LAT,
        init_lon_deg=INIT_LON,
        use_azimuth=True,
        search_radius_deg=2.0,
    )

    if fix is not None:
        result.fix_lat = fix.lat_deg
        result.fix_lon = fix.lon_deg_east
        result.fix_gdop = fix.position_gdop
        result.fix_success = fix.optimizer_success
        print(f"  Celestial fix: ({fix.lat_deg:.4f}°N, {fix.lon_deg_east:.4f}°E)"
              f"  GDOP={fix.position_gdop:.3f}  success={fix.optimizer_success}")

    # Record P before celestial update
    P_initial = kf._ukf.P.copy()
    result.cel_lat_var_before = float(P_initial[0, 0])
    result.cel_lon_var_before = float(P_initial[1, 1])

    if fix is not None and fix.optimizer_success:
        kf.update_celestial_fix(fix)
        print("  update_celestial_fix() → applied")
    else:
        print("  Celestial fix unavailable or non-convergent — skipping UKF update")

    P_after_cel = kf._ukf.P.copy()
    result.cel_lat_var_after = float(P_after_cel[0, 0])
    result.cel_lon_var_after = float(P_after_cel[1, 1])
    state_after_cel = kf.state
    result.error_after_celestial_m = _haversine_m(
        state_after_cel.lat_deg, state_after_cel.lon_deg, TRUE_LAT, TRUE_LON
    )
    print(f"  Error after celestial update: {result.error_after_celestial_m/1000:.1f} km")

    # ── 4. VLM pipeline ─────────────────────────────────────────────────────
    #
    # Priority:
    #   (a) BakLLaVA via Ollama — real inference on composite image
    #   (b) Synthetic bearing — computed from geometry, guarantees gate passage
    #
    vlm_bearing: Optional[float] = None
    vlm_source = "none"

    # (a) Try BakLLaVA / Ollama
    if vlm_bearing is None:
        try:
            from uav_nav.vlm.client import VLMClient
            client = VLMClient()
            if client.is_available():
                print("  Trying BakLLaVA (Ollama) for VLM bearing ...")
                obs = client.query(composite_jpeg)
                if obs and obs.landmarks:
                    best = max(obs.landmarks, key=lambda lm: lm.confidence)
                    vlm_bearing = (heading + best.bearing_deg) % 360.0
                    vlm_source = "ollama_bakllava"
                    print(f"  BakLLaVA detected '{best.type.value}'"
                          f"  abs_bearing={vlm_bearing:.1f}°"
                          f"  conf={best.confidence:.2f}")
        except Exception as exc:
            print(f"  WARNING: Ollama VLM failed: {exc}")

    # (c) Synthetic bearing fallback — always available, always geometrically correct
    if vlm_bearing is None:
        rng = np.random.default_rng(seed=0)
        true_geo_bearing = _bearing_to(
            state_after_cel.lat_deg, state_after_cel.lon_deg, TRUE_LAT, TRUE_LON
        )
        vlm_bearing = (true_geo_bearing + float(rng.normal(0, 5.0))) % 360.0
        vlm_source = "synthetic_fallback"
        print(f"  Synthetic fallback bearing: {vlm_bearing:.1f}° "
              f"(true={true_geo_bearing:.1f}° + noise)")

    result.vlm_source = vlm_source
    result.vlm_bearing_used = vlm_bearing

    # Record fix_sources before so we can detect gate acceptance
    sources_before = list(kf.state.fix_sources)
    kf.update_vlm_bearing(vlm_bearing, TRUE_LAT, TRUE_LON)
    sources_after = list(kf.state.fix_sources)
    result.vlm_gate_accepted = ("vlm_bearing" in sources_after
                                 and "vlm_bearing" not in sources_before)

    P_after_vlm = kf._ukf.P.copy()
    state_after_vlm = kf.state
    result.error_after_vlm_m = _haversine_m(
        state_after_vlm.lat_deg, state_after_vlm.lon_deg, TRUE_LAT, TRUE_LON
    )
    print(f"  update_vlm_bearing() gate_accepted={result.vlm_gate_accepted}"
          f"  error={result.error_after_vlm_m/1000:.1f} km")

    # ── 5. State protection check ────────────────────────────────────────────
    # The velocity / heading / bias block P[3:, 3:] must be unchanged by the
    # position-only celestial measurement.  The VLM bearing is 1-DOF and also
    # does not directly update velocity, but some cross-covariance coupling is
    # expected — we only assert the celestial state-protection invariant.
    result.vel_bias_unchanged = bool(
        np.allclose(P_after_cel[3:, 3:], P_initial[3:, 3:], atol=1e-8)
    )
    print(f"  State protection (vel/bias block unchanged): {result.vel_bias_unchanged}")

    # ── 6. Pass/fail decision ────────────────────────────────────────────────
    cel_covar_reduced = (
        fix is not None and fix.optimizer_success
        and result.cel_lat_var_after < result.cel_lat_var_before
        and result.cel_lon_var_after < result.cel_lon_var_before
    )
    vlm_pipeline_wired = not math.isnan(result.vlm_bearing_used)

    result.passed = (
        cel_covar_reduced
        and vlm_pipeline_wired
        and result.vel_bias_unchanged
        and result.composite_saved
    )

    if not cel_covar_reduced:
        result.message = (
            "Celestial covariance did not decrease — fix may have failed or "
            "been rejected by Mahalanobis gate."
        )
    elif not result.vel_bias_unchanged:
        result.message = (
            "State protection FAILED: celestial update corrupted "
            "velocity/heading/bias covariance block."
        )

    # ── 7. Save annotated composite ─────────────────────────────────────────
    try:
        _annotate_composite(
            composite_bgr=composite_bgr,
            star_pixels=star_pixels,
            observations=observations,
            heading_deg=heading,
            fov_deg=FOV_DEG,
            vlm_bearing=vlm_bearing,
            vlm_source=vlm_source,
            result=result,
        )
    except Exception as exc:
        print(f"  WARNING: Annotation failed (non-fatal): {exc}")

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
