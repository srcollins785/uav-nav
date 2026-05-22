"""
Bridges the solar observation → solver → UKF update pipeline.

Mirrors uav_nav/fusion/celestial_updater.py but for daytime solar navigation.
Much simpler than the star pipeline — no image processing or catalog matching.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from uav_nav.navigation.ukf import UAVKalmanFilter
from uav_nav.solar.solver import SolarFix, SolarObservation, get_solar_fix_warmstart

log = logging.getLogger(__name__)


class SolarUpdater:
    """
    Orchestrates the solar navigation pipeline:
      SolarObservation → position fix → UKF update
    """

    def process(
        self,
        solar_obs: SolarObservation,
        kf: UAVKalmanFilter,
        dt_utc: datetime,
    ) -> Optional[SolarFix]:
        """
        Solve for position from a solar observation and fuse into the UKF.

        Args:
            solar_obs : Measured (or simulated) solar elevation and azimuth.
            kf        : The active Kalman filter (provides warm-start position).
            dt_utc    : UTC epoch of the observation.

        Returns:
            The SolarFix applied to the UKF, or None if no fix was produced.
        """
        state = kf.state

        fix = get_solar_fix_warmstart(
            obs=solar_obs,
            dt_utc=dt_utc,
            init_lat_deg=state.lat_deg,
            init_lon_deg=state.lon_deg,
        )
        if fix is None:
            log.debug("Solar solver returned no fix.")
            return None

        kf.update_solar_fix(fix)
        log.debug(
            "Solar fix applied: lat=%.4f lon=%.4f rmse=%.3f gdop=%.3f",
            fix.lat_deg, fix.lon_deg_east, fix.rmse_normalized, fix.position_gdop,
        )
        return fix
