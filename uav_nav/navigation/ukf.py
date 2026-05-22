"""
Unscented Kalman Filter wrapper for UAV navigation.

State vector (10-dim):
    [lat_deg, lon_deg, alt_m, v_north_ms, v_east_ms, v_down_ms,
     heading_deg, baro_bias_m, mag_bias_deg, gyro_bias_z_dps]

Uses filterpy.kalman.UnscentedKalmanFilter with Merwe-scaled sigma points.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

from uav_nav.celestial.solver import CelestialFix
from uav_nav.solar.solver import SolarFix
from uav_nav.config import get_config
from uav_nav.navigation.barometer import BaroReading
from uav_nav.navigation.imu import IMUReading
from uav_nav.navigation.magnetometer import MagReading
from uav_nav.navigation.observation_models import (
    hx_airspeed,
    hx_barometer,
    hx_celestial,
    hx_magnetometer,
    hx_velocity_ned,
    hx_vlm_bearing,
    residual_bearing,
    residual_heading,
)
from uav_nav.navigation.process_model import fx_factory
from uav_nav.navigation.state import STATE_DIM, UAVState
from uav_nav.navigation.uncertainty import (
    build_Q,
    build_R_baro,
    build_R_celestial,
    build_R_mag,
    build_R_vlm_bearing,
)

log = logging.getLogger(__name__)

_P_EPSILON = 1e-9  # regularization added to keep P positive-definite


class UAVKalmanFilter:
    """
    UKF-based state estimator for offline UAV navigation.

    Usage::

        kf = UAVKalmanFilter(initial_state)
        for each sensor epoch:
            kf.predict(imu_reading)
            kf.update_barometer(baro_reading)
            kf.update_magnetometer(mag_reading)
            if vlm_obs:
                kf.update_vlm_bearing(bearing_deg, lm_lat, lm_lon)
            if celestial_fix:
                kf.update_celestial_fix(celestial_fix)
            state = kf.state
    """

    def __init__(self, initial_state: UAVState):
        cfg = get_config()

        sigma_pts = MerweScaledSigmaPoints(
            n=STATE_DIM,
            alpha=cfg.ukf_alpha,
            beta=cfg.ukf_beta,
            kappa=cfg.ukf_kappa,
        )

        dt0 = 1.0 / cfg.imu_rate_hz

        self._ukf = UnscentedKalmanFilter(
            dim_x=STATE_DIM,
            dim_z=1,            # overridden per update call
            dt=dt0,
            fx=lambda x, dt: x,  # placeholder, replaced per predict call
            hx=hx_barometer,     # placeholder, replaced per update call
            points=sigma_pts,
        )

        self._ukf.x = initial_state.to_vector()
        self._ukf.P = initial_state.covariance.copy()
        self._ukf.Q = build_Q(dt0)

        self._dt = dt0
        self._fix_sources: List[str] = []
        self._timestamp: datetime = initial_state.timestamp_utc

        # NIS (Normalized Innovation Squared) divergence monitor.
        # Tracks d²/dof = (z - h(x̂))ᵀ S⁻¹ (z - h(x̂)) / dof.
        # NOTE: this is NIS, not NEES. True NEES = (x_true - x̂)ᵀ P⁻¹ (x_true - x̂)
        # requires the ground-truth state; use compute_true_nees() in simulation.
        self._nees_window: deque = deque(maxlen=self._NEES_WINDOW_SIZE)
        self._nees_alarm: bool = False
        self._nees_alarm_count: int = 0  # cumulative NIS divergence events

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, imu: IMUReading) -> None:
        """Propagate state using IMU dead reckoning."""
        dt = imu.dt_s
        self._ukf.Q = build_Q(dt)
        self._ukf.fx = fx_factory(imu)
        self._ukf.predict(dt=dt)
        self._regularize_P()
        self._fix_sources = ["imu"]

    def update_barometer(self, baro: BaroReading) -> None:
        """Fuse barometric altitude measurement."""
        z = np.array([baro.alt_m])
        R = np.array([[baro.sigma_m ** 2]])
        try:
            self._ukf.update(z, hx=hx_barometer, R=R)
            self._regularize_P()
            self._fix_sources.append("baro")
        except Exception as exc:
            log.warning("Barometer update failed: %s", exc)

    def update_magnetometer(self, mag: MagReading) -> None:
        """Fuse magnetometer heading measurement."""
        true_heading = (mag.heading_deg + mag.declination_deg) % 360.0
        z = np.array([true_heading])
        R = np.array([[mag.sigma_deg ** 2]])
        old_res = self._ukf.residual_z
        self._ukf.residual_z = residual_heading
        try:
            self._ukf.update(z, hx=hx_magnetometer, R=R)
            self._regularize_P()
            self._fix_sources.append("mag")
        except Exception as exc:
            log.warning("Magnetometer update failed: %s", exc)
        finally:
            self._ukf.residual_z = old_res

    def update_airspeed(
        self,
        airspeed_ms: float,
        sigma_ms: float = 2.0,
    ) -> None:
        """
        Fuse a horizontal airspeed magnitude constraint.

        For a constant-airspeed UAV this prevents unconstrained velocity drift
        from accelerometer bias between absolute position fixes.
        """
        z = np.array([airspeed_ms])
        R = np.array([[sigma_ms ** 2]])
        try:
            self._ukf.update(z, hx=hx_airspeed, R=R)
            self._regularize_P()
        except Exception as exc:
            log.debug("Airspeed update failed: %s", exc)

    def update_velocity_from_heading(
        self,
        airspeed_ms: float,
        sigma_ms: float = 2.0,
    ) -> None:
        """
        Zero-sideslip velocity constraint: constrain [v_north, v_east] to
        match [airspeed * cos(heading), airspeed * sin(heading)].

        This prevents accel-bias from rotating the velocity vector, keeping
        the estimated trajectory aligned with the heading state.
        Heading and bias states are protected from velocity cross-covariance
        corruption to prevent unstable feedback loops.
        """
        import math as _math
        hdg_rad = _math.radians(float(self._ukf.x[6]))  # IDX_HDG = 6
        z = np.array([
            airspeed_ms * _math.cos(hdg_rad),
            airspeed_ms * _math.sin(hdg_rad),
        ])
        R = np.diag([sigma_ms ** 2, sigma_ms ** 2])
        # Protect heading and bias states — only velocity should be updated
        _hdg_pre = float(self._ukf.x[6])
        _baro_pre = float(self._ukf.x[7])
        _mag_pre = float(self._ukf.x[8])
        _gyro_bias_pre = float(self._ukf.x[9])
        try:
            self._ukf.update(z, hx=hx_velocity_ned, R=R)
            self._ukf.x[6] = _hdg_pre
            self._ukf.x[7] = _baro_pre
            self._ukf.x[8] = _mag_pre
            self._ukf.x[9] = _gyro_bias_pre
            self._regularize_P()
        except Exception as exc:
            log.debug("Velocity-from-heading update failed: %s", exc)

    _CHI2_1DOF_99 = 6.63   # chi-squared 99th percentile, 1 degree of freedom
    _CHI2_2DOF_99 = 9.21   # chi-squared 99th percentile, 2 degrees of freedom

    # NEES divergence monitor parameters
    _NEES_WINDOW_SIZE = 10            # rolling window of gate evaluations
    _NEES_DIVERGENCE_THRESHOLD = 5.0  # mean d²/dof above this → alarm
    _NEES_RECOVERY_THRESHOLD = 2.0    # mean d²/dof below this → clear alarm

    def update_vlm_bearing(
        self,
        bearing_deg: float,
        landmark_lat_deg: float,
        landmark_lon_deg: float,
        sigma_deg: Optional[float] = None,
    ) -> None:
        """Fuse a VLM-derived bearing to a known landmark position."""
        if sigma_deg is None:
            sigma_deg = get_config().measurement_noise.vlm_bearing_deg
        z = np.array([bearing_deg % 360.0])
        R = np.array([[sigma_deg ** 2]])
        hx = hx_vlm_bearing(landmark_lat_deg, landmark_lon_deg)
        if not self._mahalanobis_gate(z, hx, R, self._CHI2_1DOF_99,
                                      residual_fn=residual_bearing):
            log.warning("VLM bearing outlier rejected (gate): bearing=%.1f°", bearing_deg)
            return
        old_res = self._ukf.residual_z
        self._ukf.residual_z = residual_bearing
        try:
            self._ukf.update(z, hx=hx, R=R)
            self._regularize_P()
            self._fix_sources.append("vlm_bearing")
        except Exception as exc:
            log.warning("VLM bearing update failed: %s", exc)
        finally:
            self._ukf.residual_z = old_res

    def update_celestial_fix(self, fix: CelestialFix) -> None:
        """
        Fuse a celestial navigation position fix (lat/lon) into the UKF.

        The measurement uncertainty is set conservatively based on the
        solver's rmse_normalized — worse fits get larger R.
        """
        if not fix.optimizer_success:
            log.debug("Skipping non-converged celestial fix (rmse=%.3f)", fix.rmse_normalized)
            return

        cfg = get_config().measurement_noise

        # GDOP-aware sigma: use whichever is larger — the configured baseline
        # or the fix's own geometry-derived accuracy estimate.  Poor star
        # geometry (stars clustered in one quadrant) produces a high GDOP that
        # automatically reduces the Kalman gain, preventing a single bad fix
        # from corrupting a well-converged filter state.
        scale = max(1.0, fix.rmse_normalized)
        gdop_floor = fix.position_gdop          # degrees; 0.05° = 5.5 km
        lat_sigma = max(cfg.celestial_lat_deg, gdop_floor) * scale
        lon_sigma = max(cfg.celestial_lon_deg, gdop_floor) * scale

        z = np.array([fix.lat_deg, fix.lon_deg_east])
        R = np.diag([lat_sigma ** 2, lon_sigma ** 2])

        if not self._mahalanobis_gate(z, hx_celestial, R, self._CHI2_2DOF_99):
            log.warning(
                "Celestial fix outlier rejected (gate): lat=%.4f lon=%.4f rmse=%.3f",
                fix.lat_deg, fix.lon_deg_east, fix.rmse_normalized,
            )
            return

        try:
            self._ukf.update(z, hx=hx_celestial, R=R)
            self._regularize_P()
            self._fix_sources.append("celestial")
            log.debug(
                "Celestial fix applied: lat=%.4f lon=%.4f rmse=%.3f",
                fix.lat_deg, fix.lon_deg_east, fix.rmse_normalized,
            )
        except Exception as exc:
            log.warning("Celestial update failed: %s", exc)

    def update_solar_fix(self, fix: SolarFix) -> None:
        """
        Fuse a solar navigation position fix (lat/lon) into the UKF.

        Identical logic to update_celestial_fix() — same 2-DOF lat/lon
        measurement model, same GDOP-weighted noise, same selective-P update
        protecting P[2:,2:] and non-position states.  Only the baseline sigma
        values and log tag differ.
        """
        if not fix.optimizer_success:
            log.debug("Skipping non-converged solar fix (rmse=%.3f)", fix.rmse_normalized)
            return

        cfg = get_config().measurement_noise

        scale = max(1.0, fix.rmse_normalized)
        gdop_floor = fix.position_gdop
        lat_sigma = max(cfg.solar_lat_deg, gdop_floor) * scale
        lon_sigma = max(cfg.solar_lon_deg, gdop_floor) * scale

        z = np.array([fix.lat_deg, fix.lon_deg_east])
        R = np.diag([lat_sigma ** 2, lon_sigma ** 2])

        if not self._mahalanobis_gate(z, hx_celestial, R, self._CHI2_2DOF_99):
            log.warning(
                "Solar fix outlier rejected (gate): lat=%.4f lon=%.4f rmse=%.3f",
                fix.lat_deg, fix.lon_deg_east, fix.rmse_normalized,
            )
            return

        try:
            self._ukf.update(z, hx=hx_celestial, R=R)
            self._regularize_P()
            self._fix_sources.append("solar")
            log.debug(
                "Solar fix applied: lat=%.4f lon=%.4f rmse=%.3f",
                fix.lat_deg, fix.lon_deg_east, fix.rmse_normalized,
            )
        except Exception as exc:
            log.warning("Solar update failed: %s", exc)

    def update_position_fix(
        self,
        lat_deg: float,
        lon_deg: float,
        R: np.ndarray,
    ) -> None:
        """
        Fuse a direct 2-DOF lat/lon position fix with a caller-supplied 2×2
        covariance matrix R (degrees²).

        Used by range+bearing VLM fixes where the measurement covariance is
        computed analytically from the sensor geometry before this call.
        Applies the same selective-P update strategy as update_celestial_fix():
        lat/lon rows and columns of P are updated; P[2:,2:] and non-position
        states are protected.
        """
        z = np.array([lat_deg, lon_deg])

        if not self._mahalanobis_gate(z, hx_celestial, R, self._CHI2_2DOF_99):
            log.debug(
                "Range+bearing fix rejected by Mahalanobis gate: lat=%.4f lon=%.4f",
                lat_deg, lon_deg,
            )
            return

        try:
            self._ukf.update(z, hx=hx_celestial, R=R)
            self._regularize_P()
            self._fix_sources.append("range_bearing")
            log.debug(
                "Range+bearing position fix applied: lat=%.4f lon=%.4f",
                lat_deg, lon_deg,
            )
        except Exception as exc:
            log.warning("Range+bearing position fix failed: %s", exc)

    @property
    def state(self) -> UAVState:
        return UAVState.from_vector(
            x=self._ukf.x,
            covariance=self._ukf.P,
            timestamp=self._timestamp,
            fix_sources=list(self._fix_sources),
        )

    def set_timestamp(self, ts: datetime) -> None:
        self._timestamp = ts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mahalanobis_gate(
        self,
        z: np.ndarray,
        hx,
        R: np.ndarray,
        chi2_threshold: float,
        residual_fn=None,
    ) -> bool:
        """
        Return True if the innovation passes the chi-squared gate.

        Computes S ≈ H·P·H^T + R via finite-difference Jacobian, then checks
        d² = innov^T · S⁻¹ · innov < chi2_threshold.

        Recommended thresholds (chi-squared CDF at 99%):
            dim_z=1  →  6.63
            dim_z=2  →  9.21
        """
        z_pred = np.asarray(hx(self._ukf.x), dtype=float)
        if residual_fn is not None:
            innov = np.asarray(residual_fn(z, z_pred), dtype=float)
        else:
            innov = z - z_pred

        eps = 1e-5
        dim_z = len(z)
        dim_x = len(self._ukf.x)
        H = np.zeros((dim_z, dim_x))
        for i in range(dim_x):
            x_plus = self._ukf.x.copy();  x_plus[i]  += eps
            x_minus = self._ukf.x.copy(); x_minus[i] -= eps
            H[:, i] = (np.asarray(hx(x_plus)) - np.asarray(hx(x_minus))) / (2.0 * eps)

        S = H @ self._ukf.P @ H.T + R
        try:
            d2 = float(innov @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            return True  # degenerate S — allow update
        passed = d2 < chi2_threshold
        self._update_nees(d2, len(z))
        if not passed:
            log.debug("Mahalanobis gate rejected: d²=%.2f threshold=%.2f", d2, chi2_threshold)
        return passed

    def _update_nees(self, d2: float, dof: int) -> None:
        """
        Record a normalised d²/dof sample in the rolling NEES window and
        manage the divergence alarm.

        Alarm ON  — mean d²/dof > _NEES_DIVERGENCE_THRESHOLD:
            • Sets nees_alarm = True and increments nees_alarm_count.
            • The alarm is a diagnostic signal: it indicates the filter has
              diverged and the navigation solution should not be trusted.
            • Recovery from severe divergence (d² >> threshold) requires
              improved landmark infrastructure, not just a wider gate.
              Empirical analysis of rejected measurements showed median
              d²/dof ≈ 47 in the failure case — far beyond recoverable.

        Alarm OFF — mean d²/dof < _NEES_RECOVERY_THRESHOLD:
            • Clears the alarm; filter has recovered via accepted measurements.
        """
        self._nees_window.append(d2 / max(dof, 1))
        if len(self._nees_window) < 3:
            return  # need at least 3 samples before judging

        mean_nees = sum(self._nees_window) / len(self._nees_window)

        if not self._nees_alarm and mean_nees > self._NEES_DIVERGENCE_THRESHOLD:
            self._nees_alarm = True
            self._nees_alarm_count += 1
            log.warning(
                "NEES divergence detected (mean d²/dof=%.2f > %.1f): "
                "navigation solution unreliable (event #%d)",
                mean_nees, self._NEES_DIVERGENCE_THRESHOLD,
                self._nees_alarm_count,
            )
        elif self._nees_alarm and mean_nees < self._NEES_RECOVERY_THRESHOLD:
            self._nees_alarm = False
            log.info(
                "NEES recovery: mean d²/dof=%.2f — gate and Q returned to nominal",
                mean_nees,
            )

    @property
    def nees_alarm_count(self) -> int:
        """Number of NIS divergence events since filter initialization."""
        return self._nees_alarm_count

    @property
    def nees_diverged(self) -> bool:
        """True if the filter is currently in a NIS-alarm state."""
        return self._nees_alarm

    def compute_true_nees(self, x_true: np.ndarray) -> float:
        """
        Compute the true Normalized Estimation Error Squared (NEES).

        NEES = (x_true - x̂)ᵀ P⁻¹ (x_true - x̂)

        For a well-tuned filter, the expected value is equal to the state
        dimension (n=10).  Values significantly larger indicate the filter
        is overconfident (P too small relative to actual error).

        Only meaningful in simulation where x_true is available.
        Heading and bearing states use circular difference for indices 6, 8.

        Args:
            x_true: Ground-truth state vector (10-dimensional, same order as
                    the UKF state vector).

        Returns:
            NEES scalar.  Returns NaN if P is singular.
        """
        delta = x_true - self._ukf.x
        # Circular wrap for heading (idx 6) and mag_bias (idx 8)
        for idx in (6, 8):
            delta[idx] = (delta[idx] + 180.0) % 360.0 - 180.0
        try:
            return float(delta @ np.linalg.solve(self._ukf.P, delta))
        except np.linalg.LinAlgError:
            return float("nan")

    def _regularize_P(self) -> None:
        """
        Ensure covariance matrix P stays positive-definite.
        Adds a tiny diagonal term if any eigenvalue goes non-positive.
        """
        P = self._ukf.P
        # Symmetrize to prevent numerical drift
        P = (P + P.T) / 2.0
        min_eig = np.linalg.eigvalsh(P).min()
        if min_eig < _P_EPSILON:
            P += (_P_EPSILON - min_eig) * np.eye(STATE_DIM)
        self._ukf.P = P
