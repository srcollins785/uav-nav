"""
UKF unit test: synthetic straight-line flight.

A UAV flies due north at 50 m/s for 60 seconds (3 km) with perfect IMU.
Without any external corrections the UKF position should track within
100 m/min error (dominated by configured process noise, not accumulated drift).
"""
import math
import pytest
from datetime import datetime, timezone

import numpy as np

from uav_nav.navigation.imu import IMUReading
from uav_nav.navigation.state import UAVState
from uav_nav.navigation.ukf import UAVKalmanFilter


def _make_initial_state():
    s = UAVState(
        lat_deg=33.641,
        lon_deg=-84.427,
        alt_m=313.0,
        v_north_ms=50.0,
        v_east_ms=0.0,
        v_down_ms=0.0,
        heading_deg=0.0,
    )
    return s


def _north_imu(dt: float = 0.01) -> IMUReading:
    """IMU reading for straight, level, northward flight at constant speed."""
    return IMUReading(
        accel_x_ms2=0.0,
        accel_y_ms2=0.0,
        accel_z_ms2=9.81,   # level flight: body-down = gravity
        gyro_x_dps=0.0,
        gyro_y_dps=0.0,
        gyro_z_dps=0.0,
        dt_s=dt,
    )


def test_ukf_initializes():
    state = _make_initial_state()
    kf = UAVKalmanFilter(state)
    assert kf.state.lat_deg == pytest.approx(33.641, abs=1e-6)


def test_ukf_predict_moves_north():
    """After 1 second at 50 m/s north, lat should increase."""
    state = _make_initial_state()
    kf = UAVKalmanFilter(state)
    initial_lat = kf.state.lat_deg

    # 100 IMU steps = 1 second at 100 Hz
    for _ in range(100):
        kf.predict(_north_imu(0.01))

    assert kf.state.lat_deg > initial_lat, "Latitude should increase flying north"


def test_ukf_60s_position_error():
    """
    Pure IMU dead reckoning for 60 seconds at 50 m/s north.
    Expected travel ≈ 3 km = 0.02695° latitude.
    Estimated position should be within 200 m of true position
    (error driven only by process noise, not bias accumulation with perfect IMU).
    """
    state = _make_initial_state()
    kf = UAVKalmanFilter(state)

    n_steps = 6000  # 60 s at 100 Hz
    for _ in range(n_steps):
        kf.predict(_north_imu(0.01))

    final = kf.state
    true_lat = 33.641 + (50.0 * 60.0) / 6_371_000.0 * (180.0 / math.pi)
    true_lon = -84.427

    dlat = math.radians(final.lat_deg - true_lat)
    dlon = math.radians(final.lon_deg - true_lon)
    dist_m = 6_371_000.0 * math.sqrt(dlat ** 2 + dlon ** 2)

    assert dist_m < 200.0, (
        f"IMU dead reckoning error {dist_m:.1f} m after 60 s (limit 200 m)"
    )


def test_covariance_positive_definite():
    """P matrix must stay positive-definite after 100 predict steps."""
    state = _make_initial_state()
    kf = UAVKalmanFilter(state)
    for _ in range(100):
        kf.predict(_north_imu(0.01))
    eigs = np.linalg.eigvalsh(kf.state.covariance)
    assert eigs.min() > 0, "Covariance matrix must remain positive-definite"


def test_barometer_update_reduces_alt_uncertainty():
    from uav_nav.navigation.barometer import BaroReading
    state = _make_initial_state()
    kf = UAVKalmanFilter(state)
    kf.predict(_north_imu(0.01))
    P_before = kf.state.covariance[2, 2]  # alt variance
    kf.update_barometer(BaroReading(alt_m=313.0, sigma_m=2.0))
    P_after = kf.state.covariance[2, 2]
    assert P_after <= P_before, "Barometer update should reduce altitude uncertainty"
