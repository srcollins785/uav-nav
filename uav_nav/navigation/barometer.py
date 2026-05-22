"""Barometer reading dataclass and noise model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaroReading:
    alt_m: float          # Barometric altitude (m MSL)
    sigma_m: float = 2.0  # 1-sigma uncertainty (m)


# Typical MEMS barometer noise
BARO_NOISE_M = 1.5        # 1-sigma white noise (m)
BARO_DRIFT_M_PER_S = 0.01 # slow bias random walk
