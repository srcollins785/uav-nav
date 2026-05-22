"""
Solar observation construction.

Much simpler than the celestial/observation_builder.py — the sun is a single
object that needs no catalog matching or Hungarian algorithm.

Two paths:
  - Simulation: caller provides true lat/lon + UTC; this module adds noise.
  - Hardware  : detect_sun_stub() is a placeholder for real centroid detection.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from uav_nav.solar.ephemeris import IlluminationMode, get_illumination_mode, solar_position
from uav_nav.solar.solver import SolarObservation, _MIN_SIGMA_AZ, _MIN_SIGMA_ELEV


def build_solar_observation(
    elevation_deg: float,
    azimuth_deg: float,
    sigma_elev_deg: float = _MIN_SIGMA_ELEV,
    sigma_az_deg: float   = _MIN_SIGMA_AZ,
    dt_utc: Optional[datetime] = None,
) -> SolarObservation:
    """
    Construct a SolarObservation from measured (or simulated) values.

    Args:
        elevation_deg  : Atmospherically-corrected sun altitude (degrees).
        azimuth_deg    : True azimuth 0=N clockwise=E (degrees).
        sigma_elev_deg : 1-sigma elevation uncertainty (degrees).
        sigma_az_deg   : 1-sigma azimuth uncertainty (degrees).
        dt_utc         : UTC epoch (optional; used for logging only).

    Returns:
        SolarObservation ready for get_solar_fix_warmstart().
    """
    return SolarObservation(
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        sigma_elev_deg=sigma_elev_deg,
        sigma_az_deg=sigma_az_deg,
        dt_utc=dt_utc,
    )


def make_simulated_solar_obs(
    true_lat_deg: float,
    true_lon_deg: float,
    dt_utc: datetime,
    rng: np.random.Generator,
    sigma_elev_deg: float = _MIN_SIGMA_ELEV,
    sigma_az_deg: float   = _MIN_SIGMA_AZ,
) -> Optional[SolarObservation]:
    """
    Generate a synthetic solar observation from true position and UTC time.

    Returns None if the sun is not in DAYTIME mode (elevation ≤ 6°).

    Args:
        true_lat_deg   : True latitude (degrees).
        true_lon_deg   : True longitude (degrees East).
        dt_utc         : UTC observation epoch.
        rng            : NumPy random generator for reproducible noise.
        sigma_elev_deg : 1-sigma elevation noise (degrees).
        sigma_az_deg   : 1-sigma azimuth noise (degrees).

    Returns:
        Noisy SolarObservation or None if not daytime.
    """
    pos = solar_position(true_lat_deg, true_lon_deg, dt_utc)
    if get_illumination_mode(pos.elevation_deg) != IlluminationMode.DAYTIME:
        return None

    return SolarObservation(
        elevation_deg=pos.elevation_deg + rng.normal(0.0, sigma_elev_deg),
        azimuth_deg=(pos.azimuth_deg  + rng.normal(0.0, sigma_az_deg)) % 360.0,
        sigma_elev_deg=sigma_elev_deg,
        sigma_az_deg=sigma_az_deg,
        dt_utc=dt_utc,
    )


def detect_sun_stub(image_source: object) -> Optional[Tuple[float, float]]:
    """
    Hardware placeholder for real-time sun centroid detection.

    In a hardware implementation this would process the camera image (with ND
    filter) to find the sun disk centroid and return (elevation_deg, azimuth_deg)
    in the UAV body frame converted to world coordinates via IMU attitude.

    Currently returns None always — the simulation path uses
    make_simulated_solar_obs() or SensorBundle.solar_obs directly.

    Returns:
        (elevation_deg, azimuth_deg) or None if sun not detected.
    """
    return None
