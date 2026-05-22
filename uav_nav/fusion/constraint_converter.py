"""
Converts VLM SemanticObservation landmarks into UKF bearing constraints.

Key concept — SemanticMap:
  - First time a landmark type is seen: save a PendingLandmark (bearing ray).
  - When the same type is seen again after the UAV has moved > 200 m:
    triangulate the landmark position and promote to a known SemanticEntry.
  - Known landmark → directly call kf.update_vlm_bearing().

This moving-baseline triangulation extracts position information without
any pre-loaded map.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from uav_nav.config import get_config
from uav_nav.navigation.state import UAVState
from uav_nav.navigation.ukf import UAVKalmanFilter
from uav_nav.vlm.schemas import LandmarkObservation, LandmarkType, SemanticObservation

log = logging.getLogger(__name__)


@dataclass
class _PendingLandmark:
    """First sighting of a landmark — awaiting triangulation."""
    obs_lat: float
    obs_lon: float
    bearing_deg: float
    landmark_type: LandmarkType
    confidence: float


@dataclass
class _SemanticEntry:
    """Triangulated (or high-confidence assumed) landmark position."""
    lat_deg: float
    lon_deg: float
    landmark_type: LandmarkType
    confidence: float = 1.0
    physical_height_m: float = None   # enables monocular range estimate when set


def _bearing_to_ray(
    origin_lat: float, origin_lon: float, bearing_deg: float, distance_m: float = 50_000.0
) -> Tuple[float, float]:
    """Project a point along a bearing for `distance_m` metres from origin."""
    R = 6_371_000.0
    bearing_rad = math.radians(bearing_deg)
    lat1 = math.radians(origin_lat)
    lon1 = math.radians(origin_lon)
    d_over_R = distance_m / R

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_over_R)
        + math.cos(lat1) * math.sin(d_over_R) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(d_over_R) * math.cos(lat1),
        math.cos(d_over_R) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _true_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward-azimuth (true compass bearing) from (lat1, lon1) to (lat2, lon2)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlon))
    return math.degrees(math.atan2(y, x)) % 360.0


def _angular_diff_deg(a: float, b: float) -> float:
    """Shortest absolute angular difference in degrees, range [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _triangulate(
    obs1_lat: float, obs1_lon: float, bearing1_deg: float,
    obs2_lat: float, obs2_lon: float, bearing2_deg: float,
    sigma_bearing_deg: float = 15.0,
) -> Optional[Tuple[float, float, float]]:
    """
    Intersect two bearing rays to estimate landmark position.

    Uses small-angle approximation (valid within ~100 km).
    Returns (lat, lon, pos_sigma_deg) or None if rays are nearly parallel.

    pos_sigma_deg is an analytical estimate of the landmark position
    uncertainty propagated from bearing measurement noise:
        σ_pos ≈ D · σ_bearing / sin(Δβ)
    where D is the range from obs1 to the intersection and Δβ is the
    angular separation between the two rays.
    """
    # Convert to local ENU coordinates (meters) with obs1 as origin
    R = 6_371_000.0
    cos_lat = math.cos(math.radians(obs1_lat))

    # obs2 in local ENU
    e2 = (obs2_lon - obs1_lon) * math.radians(1) * R * cos_lat
    n2 = (obs2_lat - obs1_lat) * math.radians(1) * R

    b1 = math.radians(bearing1_deg)  # from obs1
    b2 = math.radians(bearing2_deg)  # from obs2

    # Direction vectors
    d1 = np.array([math.sin(b1), math.cos(b1)])
    d2 = np.array([math.sin(b2), math.cos(b2)])

    # Solve: obs1 + t1*d1 = obs2 + t2*d2
    # [d1 | -d2] * [t1, t2]^T = obs2 - obs1
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([e2, n2])

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-4:  # rays nearly parallel (unit direction vectors → max det=1)
        return None

    t = np.linalg.solve(A, b)
    if t[0] < 0 or t[1] < 0:  # intersection behind observer
        return None

    # Intersection in ENU
    ix_e = d1[0] * t[0]
    ix_n = d1[1] * t[0]

    lm_lat = obs1_lat + ix_n / (math.radians(1) * R)
    lm_lon = obs1_lon + ix_e / (math.radians(1) * R * cos_lat)

    # Propagate bearing noise to landmark position uncertainty
    # Angular separation between the two observation rays
    bearing_diff_rad = abs(b1 - b2)
    bearing_diff_rad = min(bearing_diff_rad, math.pi - bearing_diff_rad)
    bearing_diff_rad = max(bearing_diff_rad, math.radians(5.0))   # floor at 5°
    dist_m = float(t[0])
    pos_sigma_m = dist_m * math.radians(sigma_bearing_deg) / math.sin(bearing_diff_rad)
    pos_sigma_deg = pos_sigma_m / 111_000.0

    return lm_lat, lm_lon, pos_sigma_deg


def _range_bearing_to_position_fix(
    bearing_deg: float,
    range_m: float,
    sigma_bearing_deg: float,
    sigma_range_m: float,
    lm_lat: float,
    lm_lon: float,
) -> tuple:
    """
    Compute UAV (lat, lon) and 2×2 covariance matrix from range+bearing to a
    known landmark.

    bearing_deg: true bearing FROM UAV TO landmark (degrees, north-referenced).
    range_m:     estimated range from UAV to landmark (metres).

    The UAV position is the landmark position minus the range vector:
        lat_uav = lat_lm - range_m * cos(bearing) / 111_000
        lon_uav = lon_lm - range_m * sin(bearing) / (111_000 * cos(lat_lm))

    Returns (lat_fix_deg, lon_fix_deg, R_2x2) where R is in degrees².
    """
    b = math.radians(bearing_deg)
    cos_lat = max(math.cos(math.radians(lm_lat)), 1e-9)

    lat_fix = lm_lat - range_m * math.cos(b) / 111_000.0
    lon_fix = lm_lon - range_m * math.sin(b) / (111_000.0 * cos_lat)

    # Jacobian: d(lat,lon)/d(range_m, bearing_rad)
    sb, cb = math.sin(b), math.cos(b)
    J = np.array([
        [-cb / 111_000.0,           range_m * sb / 111_000.0],
        [-sb / (111_000.0 * cos_lat), -range_m * cb / (111_000.0 * cos_lat)],
    ])
    sigma_b_rad = math.radians(sigma_bearing_deg)
    R_rb = np.diag([sigma_range_m ** 2, sigma_b_rad ** 2])
    R_fix = J @ R_rb @ J.T
    return lat_fix, lon_fix, R_fix


class ConstraintConverter:
    """
    Maintains a SemanticMap of observed landmarks and converts VLM
    observations into UKF bearing updates.
    """

    def __init__(self):
        self._cfg = get_config()
        self._pending: Dict[LandmarkType, _PendingLandmark] = {}
        self._known: Dict[LandmarkType, List[_SemanticEntry]] = {}
        # Instrumentation counters — accumulated across all process() calls
        self.n_vlm_obs_total: int = 0           # all landmarks seen (before confidence filter)
        self.n_vlm_passed_confidence: int = 0   # passed confidence threshold
        self.n_vlm_rejected_confidence: int = 0 # failed confidence threshold
        self.n_vlm_rejected_bearing: int = 0    # failed absolute bearing gate
        self.n_vlm_rejected_ambiguity: int = 0  # failed ambiguity gate
        self.n_vlm_ukf_updates: int = 0         # UKF updates applied
        self.n_triangulation_attempts: int = 0  # attempts with sufficient baseline
        self.n_triangulation_successes: int = 0 # successful triangulations

    def register_known_landmark(
        self,
        landmark_type: LandmarkType,
        lat_deg: float,
        lon_deg: float,
        confidence: float = 1.0,
        physical_height_m: float = None,
    ) -> None:
        """
        Pre-load a landmark with a known geographic position.

        Bypasses triangulation for this landmark type — any VLM observation
        of this type will immediately generate a bearing update (or a
        range+bearing position fix when physical_height_m is provided).
        Suitable for landmarks whose positions are in a pre-loaded geographic
        database (power substations, water towers, airports, etc.).
        """
        entry = _SemanticEntry(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            landmark_type=landmark_type,
            confidence=confidence,
            physical_height_m=physical_height_m,
        )
        if landmark_type not in self._known:
            self._known[landmark_type] = []
        self._known[landmark_type].append(entry)
        log.debug(
            "Pre-loaded known landmark: %s at (%.4f°N, %.4f°E) [%d total of this type]",
            landmark_type.value, lat_deg, lon_deg, len(self._known[landmark_type]),
        )

    def process(
        self,
        obs: SemanticObservation,
        kf: UAVKalmanFilter,
    ) -> int:
        """
        Process one SemanticObservation and apply bearing updates to the UKF.

        Returns the number of UKF updates applied.
        """
        state = kf.state
        updates = 0

        all_lms = obs.landmarks
        hc_lms = obs.high_confidence_landmarks(self._cfg.landmark_min_confidence)
        self.n_vlm_obs_total += len(all_lms)
        self.n_vlm_passed_confidence += len(hc_lms)
        self.n_vlm_rejected_confidence += len(all_lms) - len(hc_lms)

        for lm in hc_lms:
            # Convert bearing from camera-relative (0=forward) to true bearing
            true_bearing = (state.heading_deg + lm.bearing_deg) % 360.0

            base_sigma = self._cfg.measurement_noise.vlm_bearing_deg

            if lm.type in self._known:
                # Known landmark → associate by BEARING.  Among candidates of
                # this type, pick the one whose predicted bearing from the UAV
                # best matches the observed bearing.  Nearest-by-position is
                # wrong when two same-type landmarks are simultaneously in view
                # (the nearer one hijacks an observation aimed at the farther
                # one, feeding the filter a geometrically inconsistent update
                # and tripping NEES divergence alarms).
                #
                # Two gates guard the pick:
                # 1. Absolute gate: best mismatch must be within
                #    vlm_assoc_max_bearing_mismatch_deg.
                # 2. Ambiguity gate: when ≥2 candidates exist, the best must
                #    beat the second-best by vlm_assoc_min_margin_deg.  If
                #    two landmarks both match the observation closely, the
                #    pick is a coin flip — drop the obs rather than guess.
                entries = self._known[lm.type]

                def _mismatch(e):
                    pred = _true_bearing_deg(
                        state.lat_deg, state.lon_deg, e.lat_deg, e.lon_deg,
                    )
                    return _angular_diff_deg(true_bearing, pred)

                ranked = sorted(entries, key=_mismatch)
                entry = ranked[0]
                best_mismatch = _mismatch(entry)
                if best_mismatch > self._cfg.vlm_assoc_max_bearing_mismatch_deg:
                    log.debug(
                        "VLM obs dropped: bearing mismatch %.1f° > %.1f° "
                        "gate (type=%s)",
                        best_mismatch,
                        self._cfg.vlm_assoc_max_bearing_mismatch_deg,
                        lm.type.value,
                    )
                    self.n_vlm_rejected_bearing += 1
                    continue
                if len(ranked) >= 2:
                    second_mismatch = _mismatch(ranked[1])
                    margin = second_mismatch - best_mismatch
                    if margin < self._cfg.vlm_assoc_min_margin_deg:
                        log.debug(
                            "VLM obs dropped: ambiguous (best=%.1f°, "
                            "second=%.1f°, margin=%.1f° < %.1f°, type=%s)",
                            best_mismatch, second_mismatch, margin,
                            self._cfg.vlm_assoc_min_margin_deg, lm.type.value,
                        )
                        self.n_vlm_rejected_ambiguity += 1
                        continue
                sigma_deg = base_sigma / (lm.confidence ** 2)

                # Prefer range+bearing 2-DOF fix when pixel height is reliable.
                # Below ~8 pixels apparent height the landmark is >3.75 km away
                # and monocular ranging becomes unreliable (sub-pixel noise →
                # hundreds of metres range error).  Fall back to bearing-only.
                _MIN_RANGE_BEARING_HEIGHT_PX = 8.0
                if (lm.apparent_height_px is not None
                        and lm.apparent_height_px >= _MIN_RANGE_BEARING_HEIGHT_PX
                        and entry.physical_height_m is not None):
                    f_px = self._cfg.focal_length_px
                    range_m = f_px * entry.physical_height_m / lm.apparent_height_px
                    sigma_range_m = range_m * self._cfg.vlm_range_sigma_frac
                    lat_fix, lon_fix, R_fix = _range_bearing_to_position_fix(
                        bearing_deg=true_bearing,
                        range_m=range_m,
                        sigma_bearing_deg=sigma_deg,
                        sigma_range_m=sigma_range_m,
                        lm_lat=entry.lat_deg,
                        lm_lon=entry.lon_deg,
                    )
                    kf.update_position_fix(lat_fix, lon_fix, R_fix)
                    log.debug(
                        "VLM range+bearing fix: %s range=%.0f m sigma_r=%.0f m",
                        lm.type.value, range_m, sigma_range_m,
                    )
                else:
                    # Fall back to 1-DOF bearing update (no height data)
                    kf.update_vlm_bearing(
                        bearing_deg=true_bearing,
                        landmark_lat_deg=entry.lat_deg,
                        landmark_lon_deg=entry.lon_deg,
                        sigma_deg=sigma_deg,
                    )
                    log.debug("VLM bearing update: %s at %.1f° (sigma=%.1f°)",
                              lm.type.value, true_bearing, sigma_deg)
                updates += 1
                self.n_vlm_ukf_updates += 1

            elif lm.type in self._pending:
                # Second sighting — attempt triangulation
                pending = self._pending[lm.type]
                baseline = _haversine_m(
                    pending.obs_lat, pending.obs_lon, state.lat_deg, state.lon_deg
                )
                if baseline >= self._cfg.landmark_min_baseline_m:
                    bearing1 = (state.heading_deg + pending.bearing_deg) % 360.0
                    # Minimum bearing change: reject triangulation when rays are nearly
                    # parallel (small angular difference → intersection poorly determined).
                    # With 8° measurement noise we need at least 10° angular separation
                    # to get a useful triangulation.
                    bearing_diff = abs((true_bearing - bearing1 + 180.0) % 360.0 - 180.0)
                    if bearing_diff < 10.0:
                        log.debug(
                            "Skipping triangulation of %s: bearing change %.1f° < 10° (poor geometry)",
                            lm.type.value, bearing_diff,
                        )
                        # Keep the ORIGINAL pending entry — don't reset it.
                        # The angular separation from the first observation grows as
                        # the UAV moves further; eventually bearing_diff >= 10°.
                        continue
                    self.n_triangulation_attempts += 1
                    result = _triangulate(
                        obs1_lat=pending.obs_lat, obs1_lon=pending.obs_lon,
                        bearing1_deg=bearing1,
                        obs2_lat=state.lat_deg, obs2_lon=state.lon_deg,
                        bearing2_deg=true_bearing,
                    )
                    if result is not None:
                        self.n_triangulation_successes += 1
                        lm_lat, lm_lon, pos_sigma_deg = result
                        new_entry = _SemanticEntry(
                            lat_deg=lm_lat,
                            lon_deg=lm_lon,
                            landmark_type=lm.type,
                        )
                        if lm.type not in self._known:
                            self._known[lm.type] = []
                        self._known[lm.type].append(new_entry)
                        del self._pending[lm.type]
                        # Combine bearing measurement noise with landmark position
                        # uncertainty propagated through the triangulation geometry.
                        avg_conf = (lm.confidence + pending.confidence) / 2.0
                        sigma_meas = base_sigma / (avg_conf ** 2)
                        dist_m = max(_haversine_m(
                            state.lat_deg, state.lon_deg, lm_lat, lm_lon), 100.0)
                        pos_bearing_sigma_deg = math.degrees(
                            math.atan(pos_sigma_deg * 111_000.0 / dist_m))
                        sigma_deg = math.sqrt(sigma_meas ** 2 + pos_bearing_sigma_deg ** 2)
                        kf.update_vlm_bearing(true_bearing, lm_lat, lm_lon,
                                              sigma_deg=sigma_deg)
                        updates += 1
                        self.n_vlm_ukf_updates += 1
                        log.info(
                            "Triangulated %s at (%.4f°N, %.4f°E) from %.0f m baseline "
                            "(pos_sigma=%.1f km, sigma_deg=%.1f°).",
                            lm.type.value, lm_lat, lm_lon, baseline,
                            pos_sigma_deg * 111.0, sigma_deg,
                        )
                    else:
                        log.debug("Triangulation of %s failed (parallel rays).", lm.type.value)
                else:
                    log.debug(
                        "Baseline %.0f m < %.0f m for %s — waiting for more movement.",
                        baseline, self._cfg.landmark_min_baseline_m, lm.type.value,
                    )
            else:
                # First sighting — save as pending
                self._pending[lm.type] = _PendingLandmark(
                    obs_lat=state.lat_deg,
                    obs_lon=state.lon_deg,
                    bearing_deg=lm.bearing_deg,
                    landmark_type=lm.type,
                    confidence=lm.confidence,
                )
                log.debug("New pending landmark: %s at %.1f°", lm.type.value, true_bearing)

        return updates

    @property
    def known_landmarks(self) -> Dict[LandmarkType, List[_SemanticEntry]]:
        return dict(self._known)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def vlm_stats(self) -> dict:
        """Return a snapshot of VLM gating counters for CSV/report output."""
        return {
            "vlm_obs_total": self.n_vlm_obs_total,
            "vlm_passed_confidence": self.n_vlm_passed_confidence,
            "vlm_rejected_confidence": self.n_vlm_rejected_confidence,
            "vlm_rejected_bearing": self.n_vlm_rejected_bearing,
            "vlm_rejected_ambiguity": self.n_vlm_rejected_ambiguity,
            "vlm_ukf_updates": self.n_vlm_ukf_updates,
            "triangulation_attempts": self.n_triangulation_attempts,
            "triangulation_successes": self.n_triangulation_successes,
        }
