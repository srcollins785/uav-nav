"""
Celestial observation builder: converts detected star centroids to
inverse_celnav Observation objects using catalog matching.

Algorithm:
  1. Use the UKF's current lat/lon estimate + IMU heading to project each
     catalog star to expected pixel coordinates (pinhole camera model).
  2. Run the Hungarian algorithm (scipy) to optimally match detected
     centroids to projected catalog stars within a search radius.
  3. For each matched pair, compute alt_deg and az_deg from pixel geometry.
  4. Apply atmospheric refraction correction.
  5. Return List[Observation] ready for celestial/solver.py.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from uav_nav.celestial.catalog import StarRecord, get_catalog
from uav_nav.celestial.corrections import correct_altitude
from uav_nav.config import get_config
from uav_nav.navigation.state import UAVState
from uav_nav.vlm.star_detector import StarCandidate

log = logging.getLogger(__name__)

# Maximum pixel distance between a detected centroid and a projected catalog star
_MAX_MATCH_RADIUS_PX = 50.0


def build_observations(
    candidates: List[StarCandidate],
    state: UAVState,
    dt_utc: datetime,
    image_width_px: int = 1280,
    image_height_px: int = 720,
) -> List:
    """
    Match detected star centroids to catalog stars and build Observation objects.

    Args:
        candidates:      Detected star centroids from star_detector.py.
        state:           Current UKF state estimate (lat, lon, heading).
        dt_utc:          UTC time of the image capture.
        image_width_px:  Camera image width in pixels.
        image_height_px: Camera image height in pixels.

    Returns:
        List of inverse_celnav.Observation objects (may be empty).
    """
    from uav_nav.celestial.solver import Observation, altaz_from_radec, is_available

    if not is_available():
        log.debug("inverse_celnav unavailable — skipping observation builder.")
        return []

    if not candidates:
        return []

    cfg = get_config()
    fov_deg = cfg.vlm_camera_fov_deg
    catalog = get_catalog()

    # --- Project catalog stars to pixel coordinates ---
    projected = _project_catalog_to_pixels(
        catalog=catalog,
        state=state,
        dt_utc=dt_utc,
        fov_deg=fov_deg,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
    )

    if not projected:
        log.debug("No catalog stars project into camera FOV.")
        return []

    # --- Hungarian matching ---
    matched_pairs = _hungarian_match(candidates, projected)
    if not matched_pairs:
        log.debug("No centroid-to-catalog matches within radius %.0f px.", _MAX_MATCH_RADIUS_PX)
        return []

    # --- Build Observation objects ---
    observations = []
    for cand, star, proj_az, proj_alt in matched_pairs:
        # Pixel offset from image center → angular offset
        cx = image_width_px / 2.0
        cy = image_height_px / 2.0
        px_per_deg = image_width_px / fov_deg

        # Azimuth: image x offset from center → bearing offset from heading
        az_offset = (cand.pixel_x - cx) / px_per_deg
        # Altitude: image y offset from center → elevation angle
        # (y increases downward → upward in image = higher alt)
        alt_offset = -(cand.pixel_y - cy) / px_per_deg

        # Heading-relative az → true az
        true_az = (state.heading_deg + az_offset) % 360.0

        # Vertical FOV estimated from aspect ratio
        vfov_deg = fov_deg * (image_height_px / image_width_px)
        # Camera pitch is not measured, so we cannot use the pixel y-offset
        # as an absolute altitude reference.  The catalog-projected altitude
        # (proj_alt) is the primary source of truth for this star.
        # alt_offset (pixel-based) equals proj_alt when the centroid sits
        # exactly at its projected position — adding both would double-count.
        raw_alt = proj_alt
        corrected_alt = correct_altitude(raw_alt)

        obs = Observation(
            name=star.name,
            ra_deg=star.ra_deg,
            dec_deg=star.dec_deg,
            alt_deg=corrected_alt,
            az_deg=true_az,
            sigma_alt_deg=cfg.sigma_alt_deg,
            sigma_az_deg=cfg.sigma_az_deg,
        )
        observations.append(obs)
        log.debug(
            "Matched %s: pixel(%.1f, %.1f) → alt=%.2f° az=%.2f°",
            star.name, cand.pixel_x, cand.pixel_y, corrected_alt, true_az,
        )

    log.info("Built %d celestial observations from %d candidates.", len(observations), len(candidates))
    return observations


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

@dataclass
class _ProjectedStar:
    star: StarRecord
    pixel_x: float
    pixel_y: float
    alt_deg: float
    az_deg: float


def _project_catalog_to_pixels(
    catalog: List[StarRecord],
    state: UAVState,
    dt_utc: datetime,
    fov_deg: float,
    image_width_px: int,
    image_height_px: int,
) -> List[_ProjectedStar]:
    """Project catalog stars to image pixels using the current state estimate."""
    from uav_nav.celestial.solver import altaz_from_radec

    cx = image_width_px / 2.0
    cy = image_height_px / 2.0
    px_per_deg = image_width_px / fov_deg
    vfov_deg = fov_deg * (image_height_px / image_width_px)

    projected = []
    for star in catalog:
        try:
            alt, az = altaz_from_radec(
                lat_deg=state.lat_deg,
                lon_deg_east=state.lon_deg,
                dt_utc=dt_utc,
                ra_deg=star.ra_deg,
                dec_deg=star.dec_deg,
            )
        except Exception:
            continue

        # Skip stars below horizon or more than 45° off-zenith
        if alt < 5.0:
            continue

        # Bearing relative to camera heading
        rel_az = ((az - state.heading_deg + 180.0) % 360.0) - 180.0
        if abs(rel_az) > fov_deg / 2.0:
            continue

        # Project to pixel
        px = cx + rel_az * px_per_deg
        # Alt relative to horizontal → vertical pixel
        py = cy - alt * px_per_deg  # higher alt → lower pixel index

        if 0 <= px <= image_width_px and 0 <= py <= image_height_px:
            projected.append(_ProjectedStar(
                star=star,
                pixel_x=float(px),
                pixel_y=float(py),
                alt_deg=alt,
                az_deg=az,
            ))

    return projected


def _hungarian_match(
    candidates: List[StarCandidate],
    projected: List[_ProjectedStar],
) -> List[Tuple[StarCandidate, StarRecord, float, float]]:
    """
    Optimally match detected centroids to projected catalog stars.

    Returns list of (candidate, star_record, projected_az, projected_alt) tuples
    for pairs within _MAX_MATCH_RADIUS_PX.
    """
    from scipy.optimize import linear_sum_assignment

    if not candidates or not projected:
        return []

    n_cand = len(candidates)
    n_proj = len(projected)

    # Build cost matrix (pixel distance)
    cost = np.full((n_cand, n_proj), fill_value=_MAX_MATCH_RADIUS_PX * 10.0)
    for i, c in enumerate(candidates):
        for j, p in enumerate(projected):
            dist = math.hypot(c.pixel_x - p.pixel_x, c.pixel_y - p.pixel_y)
            if dist < _MAX_MATCH_RADIUS_PX:
                cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < _MAX_MATCH_RADIUS_PX:
            proj = projected[c]
            matches.append((candidates[r], proj.star, proj.az_deg, proj.alt_deg))

    return matches
