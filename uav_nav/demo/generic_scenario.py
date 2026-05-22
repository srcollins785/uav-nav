"""
Generic synthetic scenario generator.

Works identically to synthetic_scenario.py but accepts a ScenarioConfig
instead of hardcoded ATL→MCN parameters.  All sensor models are the
same: MEMS IMU noise, baro drift, magnetometer anomaly, VLM observations
for each registered landmark within range, and Polaris+Betelgeuse celestial
observations every 5 minutes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional

import numpy as np

from uav_nav.celestial.solver import Observation as CelNavObservation
from uav_nav.demo.scenario_configs import LandmarkConfig, ScenarioConfig
from uav_nav.navigation.barometer import BaroReading
from uav_nav.navigation.imu import IMUReading
from uav_nav.navigation.magnetometer import MagReading
from uav_nav.solar.observation_builder import make_simulated_solar_obs
from uav_nav.solar.solver import SolarObservation
from uav_nav.vlm.schemas import (
    DistanceCategory, LandmarkObservation, LandmarkType,
    SceneType, SemanticObservation,
)

IMU_RATE_HZ = 100.0
IMU_DT = 1.0 / IMU_RATE_HZ
BARO_PERIOD = 0.1
MAG_PERIOD = 0.1
VLM_PERIOD = 10.0
CEL_PERIOD = 300.0

# Camera model for monocular range estimation (DJI Zenmuse X7 equivalent)
FOCAL_LENGTH_PX = 2000.0   # 24 mm lens, 4/3" sensor → f ≈ 2000 px at full res
# Simulation injects 2° noise; filter R uses 5° (conservative) to account for
# real-VLM performance variability. This intentional gap tests filter robustness.
VLM_BEARING_NOISE_DEG = 2.0


@dataclass
class FlightTruth:
    t_s: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    heading_deg: float
    v_north_ms: float
    v_east_ms: float
    timestamp_utc: object


@dataclass
class SyntheticFrame:
    t_s: float
    timestamp_utc: object
    imu: IMUReading
    baro: Optional[BaroReading]
    mag: Optional[MagReading]
    vlm_obs: Optional[SemanticObservation]
    celestial_obs: Optional[List[CelNavObservation]]
    truth: FlightTruth
    solar_obs: Optional[SolarObservation] = None


def _compute_heading(lat1, lon1, lat2, lon2) -> float:
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _propagate(lat, lon, v_north, v_east, dt):
    R = 6_371_000.0
    new_lat = lat + (v_north * dt) / R * (180.0 / math.pi)
    cos_lat = max(math.cos(math.radians(lat)), 1e-9)
    new_lon = lon + (v_east * dt) / (R * cos_lat) * (180.0 / math.pi)
    return new_lat, new_lon


def _bearing_to(obs_lat, obs_lon, lm_lat, lm_lon) -> float:
    la1, la2 = math.radians(obs_lat), math.radians(lm_lat)
    dlon = math.radians(lm_lon - obs_lon)
    y = math.sin(dlon) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def generate_scenario_from_config(cfg: ScenarioConfig) -> List[SyntheticFrame]:
    """
    Generate a full synthetic flight from a ScenarioConfig.

    Returns a list of SyntheticFrame at IMU rate (100 Hz).
    Duration is computed from the straight-line distance at cfg.airspeed_ms.
    """
    rng = np.random.default_rng(cfg.seed)

    heading = _compute_heading(cfg.origin_lat, cfg.origin_lon,
                               cfg.dest_lat, cfg.dest_lon)
    heading_rad = math.radians(heading)
    v_north = cfg.airspeed_ms * math.cos(heading_rad)
    v_east  = cfg.airspeed_ms * math.sin(heading_rad)

    # Duration from great-circle distance
    dist_m = _haversine_m(cfg.origin_lat, cfg.origin_lon,
                          cfg.dest_lat, cfg.dest_lon)
    duration_s = dist_m / cfg.airspeed_ms

    lat, lon, alt = cfg.origin_lat, cfg.origin_lon, cfg.origin_alt_m
    t = 0.0
    baro_drift_m = 0.0
    baro_acc = mag_acc = vlm_acc = cel_acc = sol_acc = 0.0

    frames: List[SyntheticFrame] = []

    while t <= duration_s:
        ts = cfg.t_start_utc + timedelta(seconds=t)

        truth = FlightTruth(
            t_s=t, lat_deg=lat, lon_deg=lon, alt_m=alt,
            heading_deg=heading, v_north_ms=v_north, v_east_ms=v_east,
            timestamp_utc=ts,
        )

        # --- IMU ---
        imu = IMUReading(
            accel_x_ms2=0.003 + rng.normal(0, 0.05),
            accel_y_ms2=rng.normal(0, 0.05),
            accel_z_ms2=9.81 + rng.normal(0, 0.05),
            gyro_x_dps=rng.normal(0, 0.1),
            gyro_y_dps=rng.normal(0, 0.1),
            gyro_z_dps=0.005 + rng.normal(0, 0.1),
            dt_s=IMU_DT,
        )

        # --- Barometer (10 Hz) ---
        baro_acc += IMU_DT
        baro: Optional[BaroReading] = None
        if baro_acc >= BARO_PERIOD:
            baro_drift_m += 0.5 / 60.0 * BARO_PERIOD
            baro = BaroReading(
                alt_m=alt + baro_drift_m + rng.normal(0, 1.5),
                sigma_m=2.0,
            )
            baro_acc = 0.0

        # --- Magnetometer (10 Hz) ---
        mag_acc += IMU_DT
        mag: Optional[MagReading] = None
        if mag_acc >= MAG_PERIOD:
            mag = MagReading(
                heading_deg=(heading - cfg.mag_declination_deg
                             + cfg.mag_anomaly_deg
                             + rng.normal(0, 2.0)) % 360.0,
                sigma_deg=3.0,
                declination_deg=cfg.mag_declination_deg,
            )
            mag_acc = 0.0

        # --- VLM (0.1 Hz) ---
        vlm_acc += IMU_DT
        vlm_obs: Optional[SemanticObservation] = None
        if vlm_acc >= VLM_PERIOD:
            vlm_obs = _make_vlm_obs(lat, lon, heading, cfg.landmarks, rng)
            vlm_acc = 0.0

        # --- Celestial / solar (every 5 min, shared cadence) ---
        cel_acc += IMU_DT
        sol_acc += IMU_DT
        cel_obs = None
        sol_obs = None
        if cel_acc >= CEL_PERIOD:
            cel_obs = _make_celestial_obs(lat, lon, ts, rng)
            cel_acc = 0.0
        if sol_acc >= CEL_PERIOD:
            sol_obs = make_simulated_solar_obs(lat, lon, ts, rng)
            sol_acc = 0.0

        frames.append(SyntheticFrame(
            t_s=t, timestamp_utc=ts,
            imu=imu, baro=baro, mag=mag,
            vlm_obs=vlm_obs, celestial_obs=cel_obs,
            truth=truth,
            solar_obs=sol_obs,
        ))

        lat, lon = _propagate(lat, lon, v_north, v_east, IMU_DT)
        t += IMU_DT

    return frames


def _make_vlm_obs(
    lat: float, lon: float, heading: float,
    landmarks: List[LandmarkConfig],
    rng: np.random.Generator,
) -> SemanticObservation:
    """Generate a synthetic VLM observation for all visible landmarks."""
    lm_list = []
    for lm in landmarks:
        dist = _haversine_m(lat, lon, lm.lat_deg, lm.lon_deg)
        if dist > lm.visible_range_km * 1000:
            continue
        true_bear = _bearing_to(lat, lon, lm.lat_deg, lm.lon_deg)
        cam_bear = (true_bear - heading + rng.normal(0, VLM_BEARING_NOISE_DEG)) % 360.0
        dist_cat = (DistanceCategory.near if dist < 500 else
                    DistanceCategory.mid if dist < 2000 else
                    DistanceCategory.far)
        confidence = max(0.71, min(0.95, 0.95 - dist / 50_000.0))

        # Monocular range: simulate apparent pixel height from known physical height
        apparent_height_px = None
        if lm.physical_height_m is not None and dist > 1.0:
            h_px_true = FOCAL_LENGTH_PX * lm.physical_height_m / dist
            # 5% pixel measurement noise (centroid detection uncertainty)
            apparent_height_px = float(h_px_true * (1.0 + rng.normal(0, 0.05)))
            apparent_height_px = max(apparent_height_px, 0.1)  # guard against negatives

        lm_list.append(LandmarkObservation(
            type=lm.landmark_type,
            bearing_deg=cam_bear,
            distance_category=dist_cat,
            confidence=confidence,
            apparent_height_px=apparent_height_px,
        ))

    return SemanticObservation(
        scene_type=SceneType.rural,
        sky_visible=True,
        landmarks=lm_list,
        stars_visible=["Polaris", "Betelgeuse"],
        horizon_features="Rolling forested terrain",
    )


def _make_celestial_obs(lat, lon, dt_utc, rng):
    try:
        from uav_nav.celestial.solver import altaz_from_radec, is_available
        if not is_available():
            return []
        # Five-star set chosen for azimuth diversity from the SE United States:
        #   Polaris (~360°N), Castor (~81°ENE), Procyon (~117°ESE),
        #   Betelgeuse (~147°SE), Aldebaran (~193°SSW).
        # Spread across 5 quadrants eliminates the degenerate longitude geometry
        # that caused false local minima with the original N+SE+SE triangle.
        stars = [
            ("Polaris",    37.95456067,  89.26410897),
            ("Castor",    113.649426,    31.888283),
            ("Procyon",   114.825493,     5.224988),
            ("Betelgeuse", 88.792939,     7.407064),
            ("Aldebaran",  68.980163,    16.509302),
        ]
        obs = []
        for name, ra, dec in stars:
            alt, az = altaz_from_radec(lat, lon, dt_utc, ra, dec)
            if alt < 5.0:
                continue
            from uav_nav.celestial.solver import Observation as CelObs
            obs.append(CelObs(
                name=name, ra_deg=ra, dec_deg=dec,
                alt_deg=alt + rng.normal(0, 0.05),
                az_deg=az  + rng.normal(0, 0.3),
                sigma_alt_deg=0.05, sigma_az_deg=0.3,
            ))
        return obs
    except Exception:
        return []
