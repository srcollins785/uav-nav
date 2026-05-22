"""UAV navigation state dataclass and utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import numpy as np

# State vector index constants — keep in sync with ukf.py
IDX_LAT = 0
IDX_LON = 1
IDX_ALT = 2
IDX_VN = 3   # velocity north (m/s)
IDX_VE = 4   # velocity east  (m/s)
IDX_VD = 5   # velocity down  (m/s)
IDX_HDG = 6  # heading (degrees, 0=N, clockwise)
IDX_BARO_BIAS = 7    # barometer altitude bias (m)
IDX_MAG_BIAS = 8     # magnetometer heading bias (degrees)
IDX_GYRO_BIAS_Z = 9  # gyro z-axis rate bias (deg/s)

STATE_DIM = 10


@dataclass
class UAVState:
    lat_deg: float = 33.641
    lon_deg: float = -84.427
    alt_m: float = 313.0
    v_north_ms: float = 0.0
    v_east_ms: float = 0.0
    v_down_ms: float = 0.0
    heading_deg: float = 0.0
    baro_bias_m: float = 0.0
    mag_bias_deg: float = 0.0
    gyro_bias_z_dps: float = 0.0
    covariance: np.ndarray = field(
        default_factory=lambda: np.diag([
            (0.001) ** 2,   # lat (deg) — ~100 m
            (0.001) ** 2,   # lon (deg)
            (10.0) ** 2,    # alt (m)
            (1.0) ** 2,     # v_north (m/s)
            (1.0) ** 2,     # v_east  (m/s)
            (0.5) ** 2,     # v_down  (m/s)
            (5.0) ** 2,     # heading (deg)
            (2.0) ** 2,     # baro_bias (m)
            (2.0) ** 2,     # mag_bias  (deg)
            (0.05) ** 2,    # gyro_bias_z (deg/s) — ~0.05 deg/s initial uncertainty
        ])
    )
    timestamp_utc: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    fix_sources: List[str] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.lat_deg,
            self.lon_deg,
            self.alt_m,
            self.v_north_ms,
            self.v_east_ms,
            self.v_down_ms,
            self.heading_deg,
            self.baro_bias_m,
            self.mag_bias_deg,
            self.gyro_bias_z_dps,
        ], dtype=float)

    @classmethod
    def from_vector(
        cls,
        x: np.ndarray,
        covariance: np.ndarray,
        timestamp: datetime,
        fix_sources: List[str],
    ) -> "UAVState":
        return cls(
            lat_deg=float(x[IDX_LAT]),
            lon_deg=float(x[IDX_LON]),
            alt_m=float(x[IDX_ALT]),
            v_north_ms=float(x[IDX_VN]),
            v_east_ms=float(x[IDX_VE]),
            v_down_ms=float(x[IDX_VD]),
            heading_deg=float(x[IDX_HDG]) % 360.0,
            baro_bias_m=float(x[IDX_BARO_BIAS]),
            mag_bias_deg=float(x[IDX_MAG_BIAS]),
            gyro_bias_z_dps=float(x[IDX_GYRO_BIAS_Z]),
            covariance=covariance.copy(),
            timestamp_utc=timestamp,
            fix_sources=list(fix_sources),
        )

    def position_sigma_km(self) -> float:
        """Rough 1-sigma position uncertainty in km (average of lat/lon sigmas)."""
        lat_sigma_km = np.sqrt(self.covariance[IDX_LAT, IDX_LAT]) * 111.0
        lon_sigma_km = (
            np.sqrt(self.covariance[IDX_LON, IDX_LON])
            * 111.0
            * abs(np.cos(np.radians(self.lat_deg)))
        )
        return float((lat_sigma_km + lon_sigma_km) / 2.0)
