"""
UKF noise covariance matrices.

Q  — process noise covariance (10×10)
R_* — measurement noise covariances per sensor type
"""
from __future__ import annotations

import numpy as np

from uav_nav.config import get_config
from uav_nav.navigation.state import STATE_DIM


def build_Q(dt: float) -> np.ndarray:
    """
    Build the 10×10 process noise matrix Q for a predict step of length dt.

    Each σ in config is interpreted as a continuous-time white-noise density
    (units: value/√s), so the variance accumulated over dt is σ²·dt — not
    (σ·dt)², which scales quadratically and severely underweights Q at the
    100 Hz IMU rate (by a factor of dt = 0.01, i.e. 100×).  A too-small Q
    makes the filter overconfident between measurement updates, which in
    turn makes it underweight subsequent measurements (K = P·Hᵀ·S⁻¹ shrinks
    when P is small), producing the "more landmarks → more error" anomaly.

    Units: lat/lon in degrees, alt/baro in m, velocity in m/s, heading/mag
    bias in degrees, gyro_bias_z in deg/s.
    """
    cfg = get_config().process_noise

    # Convert m/s position noise to degrees/s (approx)
    pos_noise_deg = cfg.pos_m / 111_000.0

    q = np.zeros((STATE_DIM, STATE_DIM))
    q[0, 0] = (pos_noise_deg ** 2) * dt         # lat
    q[1, 1] = (pos_noise_deg ** 2) * dt         # lon
    q[2, 2] = (cfg.pos_m ** 2) * dt              # alt (m)
    q[3, 3] = (cfg.vel_ms ** 2) * dt             # v_north
    q[4, 4] = (cfg.vel_ms ** 2) * dt             # v_east
    q[5, 5] = (cfg.vel_ms ** 2) * dt             # v_down
    q[6, 6] = (cfg.heading_deg ** 2) * dt        # heading
    q[7, 7] = (cfg.baro_bias_m ** 2) * dt        # baro bias
    q[8, 8] = (cfg.mag_bias_deg ** 2) * dt       # mag bias
    q[9, 9] = (cfg.gyro_bias_z_dps ** 2) * dt   # gyro z bias random walk
    return q


def build_R_baro() -> np.ndarray:
    cfg = get_config().measurement_noise
    return np.array([[cfg.baro_m ** 2]])


def build_R_mag() -> np.ndarray:
    cfg = get_config().measurement_noise
    return np.array([[cfg.mag_deg ** 2]])


def build_R_vlm_bearing() -> np.ndarray:
    cfg = get_config().measurement_noise
    return np.array([[cfg.vlm_bearing_deg ** 2]])


def build_R_celestial() -> np.ndarray:
    cfg = get_config().measurement_noise
    return np.diag([
        cfg.celestial_lat_deg ** 2,
        cfg.celestial_lon_deg ** 2,
    ])


def build_R_solar() -> np.ndarray:
    """
    Baseline 2×2 measurement noise covariance for a solar position fix.

    Larger than the star-based celestial baseline (0.10° vs 0.05°) because
    a single-object solar fix has less geometric redundancy than a multi-star fix.
    The GDOP-weighted scaling in update_solar_fix() inflates this further when
    solar geometry is poor.
    """
    cfg = get_config().measurement_noise
    return np.diag([
        cfg.solar_lat_deg ** 2,
        cfg.solar_lon_deg ** 2,
    ])
