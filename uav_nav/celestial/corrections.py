"""
Atmospheric and geometric corrections for celestial observations.

All corrections are small (< 1 degree) but matter for high-accuracy work.
"""
from __future__ import annotations

import math


def saemundsson_refraction_deg(alt_deg: float) -> float:
    """
    Atmospheric refraction correction (Saemundsson 1986).

    Returns the amount to ADD to the apparent altitude to get true altitude.
    At alt=0° (horizon): ~0.53°. At alt=90° (zenith): ~0°.

    Args:
        alt_deg: Apparent altitude above horizon (degrees).

    Returns:
        Refraction correction in degrees (always positive, subtract from obs to correct).
    """
    if alt_deg <= -1.0:
        return 0.0
    # Saemundsson formula (arcminutes)
    denom = math.tan(math.radians(alt_deg + 10.3 / (alt_deg + 5.11)))
    refraction_arcmin = 1.02 / denom
    return refraction_arcmin / 60.0  # convert to degrees


def correct_altitude(observed_alt_deg: float) -> float:
    """
    Apply atmospheric refraction to observed altitude.

    Returns the corrected (true) altitude.
    """
    refraction = saemundsson_refraction_deg(observed_alt_deg)
    return observed_alt_deg + refraction


def dip_correction_deg(observer_height_m: float) -> float:
    """
    Dip of the visible horizon due to observer height above ground.

    Returns degrees to subtract from observed altitude (always positive).

    Args:
        observer_height_m: Observer eye height above sea level (m).
    """
    if observer_height_m <= 0:
        return 0.0
    return 0.0293 * math.sqrt(observer_height_m)  # degrees
