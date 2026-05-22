"""
UKF integration tests for solar navigation (update_solar_fix).

Verifies that:
  1. A valid solar fix reduces position uncertainty.
  2. The selective-P update protects P[2:,2:] (velocity/heading/bias block).
  3. Non-position state values are preserved exactly.
  4. An outlier solar fix is rejected by the Mahalanobis gate.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from uav_nav.navigation.imu import IMUReading
from uav_nav.navigation.state import UAVState
from uav_nav.navigation.ukf import UAVKalmanFilter
from uav_nav.solar.solver import SolarFix


ATL_LAT = 33.641
ATL_LON = -84.427

_LEVEL_IMU = IMUReading(
    accel_x_ms2=0.0, accel_y_ms2=0.0, accel_z_ms2=9.81,
    gyro_x_dps=0.0, gyro_y_dps=0.0, gyro_z_dps=0.0, dt_s=0.01,
)


def _make_kf(lat=ATL_LAT, lon=ATL_LON) -> UAVKalmanFilter:
    state = UAVState(
        lat_deg=lat,
        lon_deg=lon,
        alt_m=313.0,
        v_north_ms=50.0,
        v_east_ms=0.0,
        v_down_ms=0.0,
        heading_deg=45.0,
    )
    return UAVKalmanFilter(state)


def _inflate_position_uncertainty(kf: UAVKalmanFilter) -> None:
    """
    Inflate P[0,0] and P[1,1] and refresh sigma points via predict().

    filterpy's UKF uses sigma points computed in predict() for the measurement
    update.  To make the measurement update reduce P, the sigma points must
    reflect the inflated P — so we inflate first, then call predict().
    """
    kf._ukf.P[0, 0] = (0.5) ** 2   # 0.5° std dev lat
    kf._ukf.P[1, 1] = (0.5) ** 2   # 0.5° std dev lon
    # Refresh sigma points so the update step sees the large P
    kf.predict(_LEVEL_IMU)


def _good_solar_fix(lat=ATL_LAT, lon=ATL_LON) -> SolarFix:
    return SolarFix(
        lat_deg=lat,
        lon_deg_east=lon,
        rmse_normalized=0.8,
        optimizer_success=True,
        nfev=12,
        position_gdop=0.20,
    )


class TestSolarFixReducesUncertainty:
    def test_lat_uncertainty_decreases(self):
        """Solar fix on the true lat/lon should reduce P[0,0]."""
        kf = _make_kf()
        _inflate_position_uncertainty(kf)
        p00_before = kf._ukf.P[0, 0]

        kf.update_solar_fix(_good_solar_fix())

        assert kf._ukf.P[0, 0] < p00_before, "P[0,0] should decrease after solar fix"

    def test_lon_uncertainty_decreases(self):
        """Solar fix on the true lat/lon should reduce P[1,1]."""
        kf = _make_kf()
        _inflate_position_uncertainty(kf)
        p11_before = kf._ukf.P[1, 1]

        kf.update_solar_fix(_good_solar_fix())

        assert kf._ukf.P[1, 1] < p11_before, "P[1,1] should decrease after solar fix"


class TestSelectivePUpdate:
    def test_inner_P_block_protected(self):
        """P[2:,2:] (velocity/heading/bias sub-matrix) must be unchanged."""
        kf = _make_kf()
        _inflate_position_uncertainty(kf)
        # Capture P_inner AFTER predict (sigma refresh) but BEFORE the solar fix
        P_inner_before = kf._ukf.P[2:, 2:].copy()

        kf.update_solar_fix(_good_solar_fix())

        P_inner_after = kf._ukf.P[2:, 2:]
        np.testing.assert_allclose(
            P_inner_after, P_inner_before, atol=1e-10,
            err_msg="P[2:,2:] should be unchanged by a solar fix",
        )

    def test_position_block_updated(self):
        """P[0:2,0:2] should change (decrease) after a close solar fix."""
        kf = _make_kf()
        _inflate_position_uncertainty(kf)
        P_pos_before = kf._ukf.P[0:2, 0:2].copy()

        kf.update_solar_fix(_good_solar_fix())

        P_pos_after = kf._ukf.P[0:2, 0:2]
        # At least one diagonal should have decreased
        assert (P_pos_after[0, 0] < P_pos_before[0, 0] or
                P_pos_after[1, 1] < P_pos_before[1, 1])


class TestStateVectorProtection:
    def test_non_position_states_preserved(self):
        """
        Velocity, heading, and bias states must be exactly preserved
        after a solar fix (fix has zero information about these states).
        """
        kf = _make_kf()
        # Record non-position states before update
        x_before = kf._ukf.x[3:10].copy()

        kf.update_solar_fix(_good_solar_fix())

        x_after = kf._ukf.x[3:10]
        np.testing.assert_array_equal(
            x_after, x_before,
            err_msg="Non-position states must be unchanged by a solar fix",
        )


class TestMahalanobisGate:
    def test_outlier_fix_rejected(self):
        """
        A fix 5° away from the true position should be gated out.
        The fix_sources list should not contain 'solar' after the rejected update.
        """
        kf = _make_kf()
        kf.predict(_LEVEL_IMU)   # initialize sigmas_f
        # Solar fix 5° off in lat — wildly outside the Mahalanobis gate
        bad_fix = SolarFix(
            lat_deg=ATL_LAT + 5.0,
            lon_deg_east=ATL_LON,
            rmse_normalized=0.8,
            optimizer_success=True,
            nfev=10,
            position_gdop=0.20,
        )
        state_before = kf._ukf.x.copy()
        kf.update_solar_fix(bad_fix)

        assert "solar" not in kf.state.fix_sources, (
            "Outlier solar fix should be rejected by Mahalanobis gate"
        )
        np.testing.assert_allclose(
            kf._ukf.x, state_before, atol=1e-10,
            err_msg="State vector should be unchanged after rejected fix",
        )

    def test_non_converged_fix_skipped(self):
        """A fix with optimizer_success=False must be silently ignored."""
        kf = _make_kf()
        x_before = kf._ukf.x.copy()
        P_before  = kf._ukf.P.copy()

        bad_fix = SolarFix(
            lat_deg=ATL_LAT,
            lon_deg_east=ATL_LON,
            rmse_normalized=5.0,
            optimizer_success=False,   # <-- did not converge
            nfev=500,
            position_gdop=0.5,
        )
        kf.update_solar_fix(bad_fix)

        np.testing.assert_array_equal(kf._ukf.x, x_before)
        np.testing.assert_array_equal(kf._ukf.P, P_before)


class TestFixSourceTracking:
    def test_good_fix_recorded_in_fix_sources(self):
        """A valid solar fix should add 'solar' to fix_sources."""
        kf = _make_kf()
        _inflate_position_uncertainty(kf)
        kf.update_solar_fix(_good_solar_fix())
        assert "solar" in kf.state.fix_sources
