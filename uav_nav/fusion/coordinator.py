"""
Navigation coordinator — main event loop.

Pulls sensor readings from a SensorBundle at each epoch and dispatches
to the appropriate sub-systems in the correct order:

  1. IMU predict (always, 100 Hz)
  2. Barometer update (10 Hz)
  3. Magnetometer update (10 Hz)
  4. VLM semantic update (0.1 Hz) → ConstraintConverter
  5. Celestial fix update (0.017 Hz) → CelestialUpdater

All steps are conditional on data availability — the system degrades
gracefully when any sensor stream is absent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from uav_nav.celestial.solver import CelestialFix
from uav_nav.config import get_config
from uav_nav.fusion.celestial_updater import CelestialUpdater
from uav_nav.fusion.constraint_converter import ConstraintConverter
from uav_nav.fusion.sensor_scheduler import SensorScheduler
from uav_nav.fusion.solar_updater import SolarUpdater
from uav_nav.navigation.barometer import BaroReading
from uav_nav.navigation.imu import IMUReading
from uav_nav.navigation.magnetometer import MagReading
from uav_nav.navigation.state import UAVState
from uav_nav.navigation.ukf import UAVKalmanFilter
from uav_nav.solar.ephemeris import IlluminationMode, get_illumination_mode, solar_position
from uav_nav.solar.solver import SolarFix, SolarObservation
from uav_nav.vlm.client import VLMClient
from uav_nav.vlm.schemas import SemanticObservation

log = logging.getLogger(__name__)


@dataclass
class SensorBundle:
    """All sensor data available at one navigation epoch."""
    imu: IMUReading
    timestamp_utc: datetime
    baro: Optional[BaroReading] = None
    mag: Optional[MagReading] = None
    image: Optional[Union[str, Path, bytes]] = None   # camera frame
    image_width_px: int = 1280
    image_height_px: int = 720
    solar_obs: Optional[SolarObservation] = None      # pre-computed for simulation path
    airspeed_ms: float = 50.0                         # cruise airspeed for zero-sideslip constraint


@dataclass
class EpochResult:
    """Output from one coordinator epoch."""
    state: UAVState
    celestial_fix: Optional[CelestialFix] = None
    solar_fix: Optional[SolarFix] = None
    vlm_obs: Optional[SemanticObservation] = None
    vlm_updates_applied: int = 0


class NavigationCoordinator:
    """
    Top-level navigation coordinator.

    Args:
        initial_state:  Starting UAV state (lat/lon from pre-launch GPS fix).
        vlm_client:     VLM client instance (or None to disable VLM).
        enable_celestial: Whether to attempt celestial fixes (requires clear sky images).
    """

    def __init__(
        self,
        initial_state: UAVState,
        vlm_client: Optional[VLMClient] = None,
        enable_celestial: bool = True,
    ):
        self._kf = UAVKalmanFilter(initial_state)
        self._vlm = vlm_client or VLMClient()
        self._celestial = CelestialUpdater()
        self._solar = SolarUpdater()
        self._converter = ConstraintConverter()
        self._scheduler = SensorScheduler()
        self._enable_celestial = enable_celestial

    def process_epoch(self, bundle: SensorBundle) -> EpochResult:
        """Process one sensor epoch and return updated state."""
        dt = bundle.imu.dt_s
        self._scheduler.tick(dt)
        self._kf.set_timestamp(bundle.timestamp_utc)

        result = EpochResult(state=self._kf.state)

        # 1. IMU predict (always)
        self._kf.predict(bundle.imu)

        # 2. Barometer update
        if bundle.baro and self._scheduler.baro_due:
            self._kf.update_barometer(bundle.baro)

        # 3. Magnetometer update
        if bundle.mag and self._scheduler.mag_due:
            self._kf.update_magnetometer(bundle.mag)

        # 3b. Zero-sideslip velocity constraint (1 Hz via airspeed_due)
        if self._scheduler.airspeed_due:
            self._kf.update_velocity_from_heading(bundle.airspeed_ms, sigma_ms=2.0)

        # 4. VLM semantic update
        if bundle.image and self._scheduler.vlm_due:
            try:
                obs = self._vlm.query(bundle.image)
                if obs is not None:
                    result.vlm_obs = obs
                    n_updates = self._converter.process(obs, self._kf)
                    result.vlm_updates_applied = n_updates
            except Exception as exc:
                log.warning("VLM query failed in coordinator: %s", exc)

        # 5. Celestial / solar update (gated by illumination mode)
        if self._enable_celestial and self._scheduler.celestial_due:
            _sun_pos = solar_position(
                self._kf.state.lat_deg,
                self._kf.state.lon_deg,
                bundle.timestamp_utc,
            )
            _illum = get_illumination_mode(_sun_pos.elevation_deg)

            # 5a. Daytime: solar fix (sun elevation > 6°)
            if _illum == IlluminationMode.DAYTIME:
                try:
                    solar_obs = bundle.solar_obs   # simulation path
                    if solar_obs is None and bundle.image:
                        # Hardware path — stub returns None until implemented
                        from uav_nav.solar.observation_builder import detect_sun_stub
                        raw = detect_sun_stub(bundle.image)
                        if raw is not None:
                            from uav_nav.solar.observation_builder import build_solar_observation
                            solar_obs = build_solar_observation(*raw, dt_utc=bundle.timestamp_utc)
                    if solar_obs is not None:
                        fix = self._solar.process(solar_obs, self._kf, bundle.timestamp_utc)
                        result.solar_fix = fix
                except Exception as exc:
                    log.warning("Solar update failed in coordinator: %s", exc)

            # 5b. Nighttime: star-based celestial fix (sun elevation < −12°)
            elif _illum == IlluminationMode.NIGHTTIME and bundle.image:
                if result.vlm_obs and result.vlm_obs.sky_visible:
                    try:
                        fix = self._celestial.process(
                            image_source=bundle.image,
                            kf=self._kf,
                            dt_utc=bundle.timestamp_utc,
                            image_width_px=bundle.image_width_px,
                            image_height_px=bundle.image_height_px,
                        )
                        result.celestial_fix = fix
                    except Exception as exc:
                        log.warning("Celestial update failed in coordinator: %s", exc)
            # Twilight: neither source active — graceful degradation to IMU/baro/VLM

        result.state = self._kf.state
        return result

    @property
    def state(self) -> UAVState:
        return self._kf.state

    @property
    def constraint_converter(self) -> ConstraintConverter:
        return self._converter
