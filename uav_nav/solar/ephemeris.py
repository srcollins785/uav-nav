"""
Solar position ephemeris — NOAA simplified algorithm.

Computes the sun's altitude and azimuth for a given observer position and UTC
time.  Accurate to ±0.01° for the years 2000–2100, which exceeds the ~0.15°
measurement precision of a sun-disk centroid.

No external dependencies — uses only Python stdlib `math`.

References:
  NOAA Solar Calculator (https://gml.noaa.gov/grad/solcalc/)
  Blanco-Muriel et al. (2001) — Solar Energy 70(5) pp.431-441
  Meeus, J. (1998) — Astronomical Algorithms, 2nd ed., Willmann-Bell
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from uav_nav.celestial.corrections import saemundsson_refraction_deg

# Illumination thresholds (degrees)
DAYTIME_MIN_ELEVATION_DEG: float = 6.0    # sun reliably above horizon, low refraction error
NIGHTTIME_MAX_ELEVATION_DEG: float = -12.0  # astronomical twilight boundary


class IlluminationMode(Enum):
    """Day/night/twilight classification used to gate celestial sources."""
    DAYTIME = "daytime"
    TWILIGHT = "twilight"
    NIGHTTIME = "nighttime"


@dataclass(frozen=True)
class SolarPosition:
    """
    Sun position at a specific observer location and UTC time.

    elevation_deg         : True altitude above horizon (degrees), after refraction correction.
    azimuth_deg           : True azimuth (degrees, 0=N, clockwise=E).
    declination_deg       : Sun's geocentric declination (degrees).
    equation_of_time_min  : Difference between apparent and mean solar time (minutes).
    """
    elevation_deg: float
    azimuth_deg: float
    declination_deg: float
    equation_of_time_min: float


def _julian_day(dt_utc: datetime) -> float:
    """
    Convert UTC datetime to Julian Day Number.

    Matches the formula used in inverse_celnav.py (verified to agree to 1e-9).
    """
    y = dt_utc.year
    m = dt_utc.month
    d = (dt_utc.day
         + dt_utc.hour / 24.0
         + dt_utc.minute / 1440.0
         + dt_utc.second / 86400.0
         + getattr(dt_utc, "microsecond", 0) / 86_400_000_000.0)

    if m <= 2:
        y -= 1
        m += 12

    A = int(y / 100)
    B = 2 - A + int(A / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5


def solar_position(
    lat_deg: float,
    lon_deg_east: float,
    dt_utc: datetime,
) -> SolarPosition:
    """
    Compute the sun's position for an observer at (lat_deg, lon_deg_east) at dt_utc.

    Args:
        lat_deg       : Observer latitude (degrees, positive N).
        lon_deg_east  : Observer longitude (degrees East, negative W).
        dt_utc        : UTC observation time (timezone-aware or naive UTC).

    Returns:
        SolarPosition with refraction-corrected elevation and true azimuth.
    """
    # Ensure we have a UTC-equivalent value even if naive
    if dt_utc.tzinfo is not None:
        dt_utc = dt_utc.astimezone(timezone.utc).replace(tzinfo=None)

    # Step 1 — Julian Day and Julian centuries from J2000.0
    JD = _julian_day(dt_utc)
    T = (JD - 2_451_545.0) / 36_525.0

    # Step 2 — Geometric mean longitude and mean anomaly of the sun (degrees)
    L0 = (280.46646 + 36_000.76983 * T + 0.0003032 * T * T) % 360.0
    M  = (357.52911 + 35_999.05029 * T - 0.0001537 * T * T) % 360.0
    M_rad = math.radians(M)

    # Step 3 — Equation of center C
    C = ((1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(M_rad)
         + (0.019993 - 0.000101 * T) * math.sin(2.0 * M_rad)
         + 0.000289 * math.sin(3.0 * M_rad))

    # Step 4 — Sun's true longitude and apparent longitude
    sun_lon = L0 + C
    omega = 125.04 - 1934.136 * T                    # moon ascending node (degrees)
    apparent_lon = sun_lon - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # Step 5 — Obliquity of ecliptic and sun's declination
    epsilon0 = (23.0
                + 26.0 / 60.0
                + 21.448 / 3600.0
                - (46.8150 * T + 0.00059 * T * T - 0.001813 * T * T * T) / 3600.0)
    epsilon = epsilon0 + 0.00256 * math.cos(math.radians(omega))   # apparent obliquity

    sin_dec = math.sin(math.radians(epsilon)) * math.sin(math.radians(apparent_lon))
    sin_dec = max(-1.0, min(1.0, sin_dec))   # numerical clamp
    dec_rad = math.asin(sin_dec)
    dec_deg = math.degrees(dec_rad)

    # Step 6 — Equation of time (minutes)
    # Orbital eccentricity
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T
    y_eot = math.tan(math.radians(epsilon / 2.0)) ** 2
    L0_rad = math.radians(L0)
    EqT_rad = (y_eot * math.sin(2.0 * L0_rad)
               - 2.0 * e * math.sin(M_rad)
               + 4.0 * e * y_eot * math.sin(M_rad) * math.cos(2.0 * L0_rad)
               - 0.5 * y_eot * y_eot * math.sin(4.0 * L0_rad)
               - 1.25 * e * e * math.sin(2.0 * M_rad))
    EqT_min = 4.0 * math.degrees(EqT_rad)

    # Step 7 — True Solar Time and Hour Angle
    utc_min = (dt_utc.hour * 60.0
               + dt_utc.minute
               + dt_utc.second / 60.0
               + getattr(dt_utc, "microsecond", 0) / 60_000_000.0)
    true_solar_time = (utc_min + EqT_min + 4.0 * lon_deg_east) % 1440.0
    hour_angle_deg = true_solar_time / 4.0 - 180.0   # negative in morning

    # Step 8 — Solar elevation (geometric) and azimuth
    lat_rad = math.radians(lat_deg)
    ha_rad  = math.radians(hour_angle_deg)

    cos_zenith = (math.sin(lat_rad) * math.sin(dec_rad)
                  + math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad))
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith_rad = math.acos(cos_zenith)
    elevation_geom = 90.0 - math.degrees(zenith_rad)

    # Azimuth: 0=N, clockwise=E (matches inverse_celnav convention)
    sin_zenith = math.sin(zenith_rad)
    if sin_zenith < 1e-9:
        # Sun at or very near zenith — azimuth undefined; set to 0
        azimuth_deg = 0.0
    else:
        sin_az = (-math.cos(dec_rad) * math.sin(ha_rad)) / sin_zenith
        cos_az = (math.sin(dec_rad) - math.sin(lat_rad) * cos_zenith) / (math.cos(lat_rad) * sin_zenith)
        azimuth_deg = math.degrees(math.atan2(sin_az, cos_az)) % 360.0

    # Step 9 — Atmospheric refraction correction (reuse Saemundsson formula)
    refraction = saemundsson_refraction_deg(elevation_geom)
    elevation_corrected = elevation_geom + refraction

    return SolarPosition(
        elevation_deg=elevation_corrected,
        azimuth_deg=azimuth_deg,
        declination_deg=dec_deg,
        equation_of_time_min=EqT_min,
    )


def get_illumination_mode(elevation_deg: float) -> IlluminationMode:
    """
    Classify sky illumination based on solar elevation.

    Returns:
        DAYTIME   if elevation > DAYTIME_MIN_ELEVATION_DEG (6°)
        NIGHTTIME if elevation < NIGHTTIME_MAX_ELEVATION_DEG (−12°)
        TWILIGHT  otherwise (civil/nautical/astronomical twilight band)
    """
    if elevation_deg > DAYTIME_MIN_ELEVATION_DEG:
        return IlluminationMode.DAYTIME
    if elevation_deg < NIGHTTIME_MAX_ELEVATION_DEG:
        return IlluminationMode.NIGHTTIME
    return IlluminationMode.TWILIGHT
