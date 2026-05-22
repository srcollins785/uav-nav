"""
UKF measurement functions h(x) for each sensor type.

Each function takes the 10-dim state vector x and returns a measurement
prediction vector z_pred. These are passed as `hx` to ukf.update().
"""
from __future__ import annotations

import math

import numpy as np

from uav_nav.navigation.state import (
    IDX_ALT, IDX_BARO_BIAS, IDX_HDG, IDX_LAT, IDX_LON, IDX_MAG_BIAS,
    IDX_VN, IDX_VE,
)


def hx_barometer(x: np.ndarray) -> np.ndarray:
    """Barometer measures altitude plus (unknown) bias."""
    return np.array([x[IDX_ALT] + x[IDX_BARO_BIAS]])


def hx_magnetometer(x: np.ndarray) -> np.ndarray:
    """Magnetometer measures heading plus (unknown) bias."""
    return np.array([(x[IDX_HDG] + x[IDX_MAG_BIAS]) % 360.0])


def hx_celestial(x: np.ndarray) -> np.ndarray:
    """Celestial fix provides a direct lat/lon measurement."""
    return np.array([x[IDX_LAT], x[IDX_LON]])


def hx_vlm_bearing(landmark_lat_deg: float, landmark_lon_deg: float):
    """
    Factory: returns h(x) that computes the expected bearing from the
    current position estimate to a known landmark at (landmark_lat, landmark_lon).

    Returns bearing in [0, 360) degrees clockwise from north.
    """
    def hx(x: np.ndarray) -> np.ndarray:
        obs_lat = math.radians(x[IDX_LAT])
        obs_lon = math.radians(x[IDX_LON])
        lm_lat = math.radians(landmark_lat_deg)
        lm_lon = math.radians(landmark_lon_deg)

        dlon = lm_lon - obs_lon
        y = math.sin(dlon) * math.cos(lm_lat)
        x_comp = (math.cos(obs_lat) * math.sin(lm_lat)
                  - math.sin(obs_lat) * math.cos(lm_lat) * math.cos(dlon))
        bearing = math.degrees(math.atan2(y, x_comp)) % 360.0
        return np.array([bearing])

    return hx


def hx_airspeed(x: np.ndarray) -> np.ndarray:
    """Airspeed constraint: measures horizontal speed magnitude (m/s)."""
    v_north = x[IDX_VN]
    v_east = x[IDX_VE]
    speed = math.sqrt(v_north ** 2 + v_east ** 2)
    return np.array([max(speed, 1e-6)])  # avoid divide-by-zero in sigma-point derivatives


def hx_velocity_ned(x: np.ndarray) -> np.ndarray:
    """
    Zero-sideslip velocity constraint: measures [v_north, v_east] directly.

    Paired with z = [airspeed * cos(heading), airspeed * sin(heading)], this
    constrains both the magnitude AND direction of the horizontal velocity,
    preventing accel-bias from rotating the velocity vector between fixes.
    """
    return np.array([x[IDX_VN], x[IDX_VE]])


def residual_heading(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Heading residual wrapped to [-180, 180]."""
    diff = (a[0] - b[0] + 180.0) % 360.0 - 180.0
    return np.array([diff])


def residual_bearing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bearing residual wrapped to [-180, 180]."""
    diff = (a[0] - b[0] + 180.0) % 360.0 - 180.0
    return np.array([diff])
