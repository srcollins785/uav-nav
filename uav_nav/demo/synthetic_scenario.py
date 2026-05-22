"""
Synthetic scenario generator: Atlanta → Macon, GA night flight.

Generates all sensor data for a 30-minute flight from
Hartsfield-Jackson (33.641°N, 84.427°W) to Middle Georgia Regional (32.693°N, 83.649°W)
at 50 m/s and 150 m AGL.

Sensor streams produced:
  - IMU at 100 Hz (biased accelerometer + noisy gyro)
  - Barometer at 10 Hz (slow drift +0.5 m/min)
  - Magnetometer at 10 Hz (constant 3° E bias on top of -4.9° declination)
  - Synthetic VLM observations at configured VLM events
  - Celestial observations every 5 minutes (Polaris + Betelgeuse)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np

from uav_nav.celestial.solver import Observation as CelNavObservation
from uav_nav.navigation.barometer import BaroReading
from uav_nav.navigation.imu import IMUReading
from uav_nav.navigation.magnetometer import MagReading
from uav_nav.vlm.schemas import (
    DistanceCategory, LandmarkObservation, LandmarkType,
    SceneType, SemanticObservation,
)

# --- Flight parameters ---
ORIGIN_LAT = 33.641
ORIGIN_LON = -84.427
ORIGIN_ALT = 313.0 + 150.0   # 150 m AGL over ATL elevation
DEST_LAT = 32.693
DEST_LON = -83.649
AIRSPEED_MS = 50.0
T_START_UTC = datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc)
IMU_RATE_HZ = 100.0
IMU_DT = 1.0 / IMU_RATE_HZ

# --- Known landmark (Georgia Power McDonough Substation area) ---
SUBSTATION_LAT = 33.44
SUBSTATION_LON = -84.15

# Noise seeds for reproducibility
_RNG = random.Random(42)
_NP_RNG = np.random.default_rng(42)


@dataclass
class FlightTruth:
    """Ground-truth position at each second."""
    t_s: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    heading_deg: float
    v_north_ms: float
    v_east_ms: float
    timestamp_utc: datetime


@dataclass
class SyntheticFrame:
    """All sensor data for one IMU step."""
    t_s: float
    timestamp_utc: datetime
    imu: IMUReading
    baro: Optional[BaroReading]
    mag: Optional[MagReading]
    vlm_obs: Optional[SemanticObservation]
    celestial_obs: Optional[List[CelNavObservation]]
    truth: FlightTruth


def _compute_heading(origin_lat, origin_lon, dest_lat, dest_lon) -> float:
    """Initial bearing from origin to destination."""
    lat1 = math.radians(origin_lat)
    lat2 = math.radians(dest_lat)
    dlon = math.radians(dest_lon - origin_lon)
    y = math.sin(dlon) * math.cos(lat2)
    x = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return math.degrees(math.atan2(y, x)) % 360.0


def _propagate_position(lat, lon, v_north, v_east, dt):
    R = 6_371_000.0
    new_lat = lat + (v_north * dt) / R * (180.0 / math.pi)
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    new_lon = lon + (v_east * dt) / (R * cos_lat) * (180.0 / math.pi)
    return new_lat, new_lon


def _bearing_to_landmark(obs_lat, obs_lon, lm_lat, lm_lon) -> float:
    lat1 = math.radians(obs_lat)
    lat2 = math.radians(lm_lat)
    dlon = math.radians(lm_lon - obs_lon)
    y = math.sin(dlon) * math.cos(lat2)
    x = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return math.degrees(math.atan2(y, x)) % 360.0


def generate_scenario(duration_s: float = 1800.0) -> List[SyntheticFrame]:
    """
    Generate the full Atlanta → Macon synthetic flight.

    Returns list of SyntheticFrame at IMU rate (100 Hz).
    Total frames: duration_s * IMU_RATE_HZ = 180,000 for 30 min.
    """
    heading = _compute_heading(ORIGIN_LAT, ORIGIN_LON, DEST_LAT, DEST_LON)
    heading_rad = math.radians(heading)
    v_north = AIRSPEED_MS * math.cos(heading_rad)
    v_east = AIRSPEED_MS * math.sin(heading_rad)

    lat, lon, alt = ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT
    t = 0.0
    baro_drift_m = 0.0
    baro_accumulator = 0.0
    mag_accumulator = 0.0
    vlm_accumulator = 0.0
    celestial_accumulator = 0.0

    # Baro rate counters (10 Hz = every 0.1 s)
    BARO_PERIOD = 0.1
    MAG_PERIOD = 0.1
    VLM_PERIOD = 10.0    # 0.1 Hz
    CEL_PERIOD = 300.0   # every 5 min

    frames: List[SyntheticFrame] = []

    while t <= duration_s:
        timestamp = T_START_UTC + timedelta(seconds=t)

        # --- True state ---
        truth = FlightTruth(
            t_s=t,
            lat_deg=lat,
            lon_deg=lon,
            alt_m=alt,
            heading_deg=heading,
            v_north_ms=v_north,
            v_east_ms=v_east,
            timestamp_utc=timestamp,
        )

        # --- IMU (biased, noisy) — calibrated consumer-grade MEMS IMU ---
        # Accel bias 0.003 m/s² (typical post-calibration MEMS): causes ~0.6 m/s
        # velocity drift per minute, ~27 m position drift per minute uncorrected.
        # Gyro bias 0.005 deg/s: causes ~0.3° heading drift per minute uncorrected.
        # Both are controlled by the airspeed + magnetometer constraints in the UKF.
        ACCEL_BIAS = 0.003  # m/s² — calibrated MEMS
        GYRO_BIAS = 0.005   # deg/s — calibrated MEMS
        imu = IMUReading(
            accel_x_ms2=ACCEL_BIAS + _NP_RNG.normal(0, 0.05),
            accel_y_ms2=0.0 + _NP_RNG.normal(0, 0.05),
            accel_z_ms2=9.81 + _NP_RNG.normal(0, 0.05),
            gyro_x_dps=_NP_RNG.normal(0, 0.1),
            gyro_y_dps=_NP_RNG.normal(0, 0.1),
            gyro_z_dps=GYRO_BIAS + _NP_RNG.normal(0, 0.1),  # small yaw bias
            dt_s=IMU_DT,
        )

        # --- Barometer (10 Hz) ---
        baro_accumulator += IMU_DT
        baro: Optional[BaroReading] = None
        if baro_accumulator >= BARO_PERIOD:
            baro_drift_m += 0.5 / 60.0 * BARO_PERIOD   # +0.5 m/min drift
            baro = BaroReading(
                alt_m=alt + baro_drift_m + _NP_RNG.normal(0, 1.5),
                sigma_m=2.0,
            )
            baro_accumulator = 0.0

        # --- Magnetometer (10 Hz) ---
        mag_accumulator += IMU_DT
        mag: Optional[MagReading] = None
        if mag_accumulator >= MAG_PERIOD:
            mag = MagReading(
                heading_deg=(heading - (-4.9) + 3.0 + _NP_RNG.normal(0, 2.0)) % 360.0,
                sigma_deg=3.0,
                declination_deg=-4.9,
            )
            mag_accumulator = 0.0

        # --- VLM semantic observation (0.1 Hz) ---
        vlm_accumulator += IMU_DT
        vlm_obs: Optional[SemanticObservation] = None
        if vlm_accumulator >= VLM_PERIOD:
            vlm_obs = _make_vlm_obs(lat, lon, heading, t)
            vlm_accumulator = 0.0

        # --- Celestial observation (every 5 min) ---
        celestial_accumulator += IMU_DT
        cel_obs: Optional[List[CelNavObservation]] = None
        if celestial_accumulator >= CEL_PERIOD:
            cel_obs = _make_celestial_obs(lat, lon, timestamp)
            celestial_accumulator = 0.0

        frames.append(SyntheticFrame(
            t_s=t,
            timestamp_utc=timestamp,
            imu=imu,
            baro=baro,
            mag=mag,
            vlm_obs=vlm_obs,
            celestial_obs=cel_obs,
            truth=truth,
        ))

        # Advance position
        lat, lon = _propagate_position(lat, lon, v_north, v_east, IMU_DT)
        t += IMU_DT

    return frames


def _make_vlm_obs(lat: float, lon: float, heading: float, t: float) -> SemanticObservation:
    """Generate a synthetic VLM observation with landmark bearing to substation."""
    landmarks = []

    # Check if the substation is within ~15 km (roughly visible)
    R = 6_371_000.0
    dlat = math.radians(SUBSTATION_LAT - lat)
    dlon = math.radians(SUBSTATION_LON - lon)
    dist = R * math.sqrt(dlat ** 2 + (dlon * math.cos(math.radians(lat))) ** 2)

    if dist < 15_000:
        # True bearing to substation
        true_bear = _bearing_to_landmark(lat, lon, SUBSTATION_LAT, SUBSTATION_LON)
        # Camera-relative bearing (subtract heading)
        cam_bear = (true_bear - heading) % 360.0
        # Add noise
        cam_bear_noisy = (cam_bear + _NP_RNG.normal(0, 8.0)) % 360.0

        dist_cat = DistanceCategory.near if dist < 500 else (
            DistanceCategory.mid if dist < 2000 else DistanceCategory.far
        )
        landmarks.append(LandmarkObservation(
            type=LandmarkType.transformer_substation,
            bearing_deg=cam_bear_noisy,
            distance_category=dist_cat,
            confidence=max(0.70, min(0.95, 0.95 - dist / 50_000.0)),
        ))

    return SemanticObservation(
        scene_type=SceneType.rural,
        sky_visible=True,
        landmarks=landmarks,
        stars_visible=["Polaris", "Betelgeuse"],
        horizon_features="Rolling forested hills",
    )


def _make_celestial_obs(
    lat: float, lon: float, dt_utc: datetime
) -> List[CelNavObservation]:
    """
    Generate synthetic celestial observations (Polaris + Betelgeuse)
    computed from the true position with measurement noise.
    """
    try:
        from uav_nav.celestial.solver import altaz_from_radec, is_available
        if not is_available():
            return []

        stars = [
            ("Polaris",    37.95456067, 89.26410897),
            ("Betelgeuse", 88.792939,    7.407064),
        ]
        observations = []
        for name, ra, dec in stars:
            alt, az = altaz_from_radec(lat, lon, dt_utc, ra, dec)
            if alt < 5.0:
                continue
            obs = CelNavObservation(
                name=name,
                ra_deg=ra,
                dec_deg=dec,
                alt_deg=alt + _NP_RNG.normal(0, 0.05),
                az_deg=az + _NP_RNG.normal(0, 0.3),
                sigma_alt_deg=0.05,
                sigma_az_deg=0.3,
            )
            observations.append(obs)
        return observations
    except Exception:
        return []
