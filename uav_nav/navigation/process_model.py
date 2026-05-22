"""
UKF process model f(x, dt): IMU-based dead reckoning.

State vector x (10-dim):
    [lat_deg, lon_deg, alt_m, v_north_ms, v_east_ms, v_down_ms,
     heading_deg, baro_bias_m, mag_bias_deg, gyro_bias_z_dps]

The process model propagates the state forward by dt seconds using
the current velocity and IMU readings. Bias states follow a random-walk
model (unchanged in prediction; driven only by process noise Q).
"""
from __future__ import annotations

import math

import numpy as np

from uav_nav.navigation.imu import IMUReading
from uav_nav.navigation.state import (
    IDX_ALT, IDX_BARO_BIAS, IDX_GYRO_BIAS_Z, IDX_HDG, IDX_LAT, IDX_LON,
    IDX_MAG_BIAS, IDX_VD, IDX_VE, IDX_VN,
)

# Earth radius for dead reckoning (m)
R_EARTH_M = 6_371_000.0


def state_transition(x: np.ndarray, dt: float, imu: IMUReading) -> np.ndarray:
    """
    Propagate state x forward by dt seconds.

    Args:
        x:   Current 10-dim state vector.
        dt:  Time step (seconds).
        imu: IMU reading for this interval.

    Returns:
        New 10-dim state vector.
    """
    x_new = x.copy()

    lat = float(x[IDX_LAT])
    v_north = float(x[IDX_VN])
    v_east = float(x[IDX_VE])
    v_down = float(x[IDX_VD])
    heading = float(x[IDX_HDG])

    # --- Position update via dead reckoning ---
    # Approximate: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
    cos_lat = math.cos(math.radians(lat))
    cos_lat = max(cos_lat, 1e-6)  # avoid division by zero near poles

    x_new[IDX_LAT] = lat + (v_north * dt) / (R_EARTH_M / 1.0) * (180.0 / math.pi)
    # lon change in degrees = (v_east * dt) / (R_earth * cos(lat)) * (180/pi)
    x_new[IDX_LON] = float(x[IDX_LON]) + (v_east * dt) / (R_EARTH_M * cos_lat) * (180.0 / math.pi)
    x_new[IDX_ALT] = float(x[IDX_ALT]) - v_down * dt  # positive alt = up

    # --- Velocity update from accelerometer (body frame → NED) ---
    # Simple flat-earth, level-flight assumption:
    # Body x-axis (forward) aligned with heading.
    hdg_rad = math.radians(heading)
    ax = imu.accel_x_ms2  # forward
    ay = imu.accel_y_ms2  # right
    # Gravity partially handled: assume near-level flight, az_body ≈ gravity - lift
    az = imu.accel_z_ms2  # down (+down in NED body frame when level)

    # Rotate body-frame accel to NED
    a_north = ax * math.cos(hdg_rad) - ay * math.sin(hdg_rad)
    a_east = ax * math.sin(hdg_rad) + ay * math.cos(hdg_rad)
    a_down = az - 9.81  # subtract gravity (sensor reads +9.81 at rest when pointing up)

    x_new[IDX_VN] = v_north + a_north * dt
    x_new[IDX_VE] = v_east + a_east * dt
    x_new[IDX_VD] = v_down + a_down * dt

    # --- Heading update from gyro z (bias-corrected) ---
    gyro_bias_z = float(x[IDX_GYRO_BIAS_Z])
    x_new[IDX_HDG] = (heading + (imu.gyro_z_dps - gyro_bias_z) * dt) % 360.0

    # Bias states: random walk (no deterministic update; driven by Q noise)
    # x_new[IDX_BARO_BIAS] unchanged
    # x_new[IDX_MAG_BIAS] unchanged
    # x_new[IDX_GYRO_BIAS_Z] unchanged

    return x_new


def fx_factory(imu: IMUReading):
    """Return a closure suitable as the UKF f(x, dt) argument."""
    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        return state_transition(x, dt, imu)
    return fx
