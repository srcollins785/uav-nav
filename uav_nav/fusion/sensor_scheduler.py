"""
Rate-aware sensor scheduler.

Tracks elapsed simulated/real time and determines when each sensor
is due for an update based on its configured rate.
"""
from __future__ import annotations

from uav_nav.config import get_config


class SensorScheduler:
    """
    Tracks which sensors are due for an update at the current time step.

    Usage::
        sched = SensorScheduler()
        sched.tick(dt=0.01)
        if sched.vlm_due:
            vlm_client.query(...)
        if sched.celestial_due:
            ...
    """

    def __init__(self):
        cfg = get_config()
        self._vlm_period = 1.0 / cfg.vlm_rate_hz
        self._celestial_period = 1.0 / cfg.celestial_rate_hz
        self._baro_period = 1.0 / cfg.baro_rate_hz
        self._mag_period = 1.0 / cfg.mag_rate_hz
        self._airspeed_period = 1.0   # 1 Hz — velocity-drift constraint

        self._vlm_accum = 0.0
        self._celestial_accum = 0.0
        self._baro_accum = 0.0
        self._mag_accum = 0.0
        self._airspeed_accum = 0.0

        self.vlm_due = False
        self.celestial_due = False
        self.baro_due = False
        self.mag_due = False
        self.airspeed_due = False

    def tick(self, dt: float) -> None:
        """Advance time by dt seconds and update due flags."""
        self._vlm_accum += dt
        self._celestial_accum += dt
        self._baro_accum += dt
        self._mag_accum += dt
        self._airspeed_accum += dt

        self.vlm_due = self._vlm_accum >= self._vlm_period
        self.celestial_due = self._celestial_accum >= self._celestial_period
        self.baro_due = self._baro_accum >= self._baro_period
        self.mag_due = self._mag_accum >= self._mag_period
        self.airspeed_due = self._airspeed_accum >= self._airspeed_period

        if self.vlm_due:
            self._vlm_accum = 0.0
        if self.celestial_due:
            self._celestial_accum = 0.0
        if self.baro_due:
            self._baro_accum = 0.0
        if self.mag_due:
            self._mag_accum = 0.0
        if self.airspeed_due:
            self._airspeed_accum = 0.0

    def reset(self) -> None:
        self._vlm_accum = 0.0
        self._celestial_accum = 0.0
        self._baro_accum = 0.0
        self._mag_accum = 0.0
        self._airspeed_accum = 0.0
