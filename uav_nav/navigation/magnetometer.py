"""Magnetometer reading dataclass and noise model."""
from __future__ import annotations

from dataclasses import dataclass

# Atlanta, GA magnetic declination (West = negative)
ATLANTA_DECLINATION_DEG = -4.9


@dataclass
class MagReading:
    heading_deg: float          # Raw magnetic heading (degrees, 0=N, clockwise)
    sigma_deg: float = 3.0      # 1-sigma uncertainty
    declination_deg: float = ATLANTA_DECLINATION_DEG  # magnetic → true north correction


MAG_NOISE_DEG = 2.0
MAG_BIAS_WALK_DEG_PER_S = 0.005
