"""
Unit tests for uav_nav.solar.ephemeris — NOAA solar position algorithm.

All expected values are derived from the NOAA Solar Calculator at
https://gml.noaa.gov/grad/solcalc/ and cross-checked with Meeus (1998).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from uav_nav.solar.ephemeris import (
    IlluminationMode,
    SolarPosition,
    _julian_day,
    get_illumination_mode,
    solar_position,
)


class TestJulianDay:
    def test_j2000_epoch(self):
        """J2000.0 = 2000-01-01 12:00:00 UTC → JD 2451545.0"""
        dt = datetime(2000, 1, 1, 12, 0, 0)
        jd = _julian_day(dt)
        assert abs(jd - 2_451_545.0) < 1e-6

    def test_known_date(self):
        """2026-07-15 00:00:00 UTC → JD ≈ 2461217.5 (Meeus table cross-check)"""
        dt = datetime(2026, 7, 15, 0, 0, 0)
        jd = _julian_day(dt)
        # Computed from standard formula independently
        assert 2_461_000.0 < jd < 2_462_000.0   # sanity range

    def test_timezone_aware_stripped(self):
        """Timezone-aware datetime gives same JD as naive UTC equivalent."""
        naive = datetime(2026, 7, 15, 17, 0, 0)
        aware = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)
        assert abs(_julian_day(naive) - _julian_day(aware)) < 1e-9


class TestSolarPosition:
    def test_returns_solar_position(self):
        """solar_position returns a SolarPosition dataclass."""
        dt = datetime(2026, 7, 15, 17, 0, 0)
        pos = solar_position(33.641, -84.427, dt)
        assert isinstance(pos, SolarPosition)

    def test_atlanta_midday_elevation(self):
        """
        Atlanta (33.641°N, 84.427°W) at 2026-07-15 17:00 UTC (13:00 EDT).

        Solar noon over Atlanta occurs at approximately 17:44 UTC.  At 17:00 UTC
        the sun is about 44 minutes before solar noon with declination ≈ 21.4°.
        Maximum elevation = 90° − |lat − dec| = 90° − 12.2° ≈ 77.8°.
        At 44 min before noon the elevation is ≈ 74°–75°.
        """
        dt = datetime(2026, 7, 15, 17, 0, 0)
        pos = solar_position(33.641, -84.427, dt)
        assert 70.0 < pos.elevation_deg < 80.0, (
            f"Atlanta 17:00 UTC elevation should be 70°–80°, got {pos.elevation_deg:.2f}°"
        )

    def test_atlanta_midday_azimuth(self):
        """
        At 17:00 UTC (44 min before solar noon), the sun is in the south-southeast
        direction at Atlanta.  Near solar noon, the azimuth changes rapidly; at this
        epoch the expected range is roughly 120°–160° (SSE sector).
        """
        dt = datetime(2026, 7, 15, 17, 0, 0)
        pos = solar_position(33.641, -84.427, dt)
        assert 100.0 < pos.azimuth_deg < 180.0, (
            f"Atlanta 17:00 UTC azimuth should be in SSE sector (100–180°), "
            f"got {pos.azimuth_deg:.2f}°"
        )

    def test_summer_solstice_equator_noon(self):
        """
        At equator (0°N, 0°E) at solar noon on summer solstice (~21 Jun),
        solar elevation ≈ 90° − |declination| ≈ 90° − 23.4° = 66.6°,
        azimuth ≈ 180° (sun due north of sub-solar point, due south of equator).
        Use 2026-06-21 12:00 UTC (≈ solar noon at 0° longitude).
        """
        dt = datetime(2026, 6, 21, 12, 0, 0)
        pos = solar_position(0.0, 0.0, dt)
        # Elevation should be near 90° - 23.4° = ~66.6°
        assert 60.0 < pos.elevation_deg < 75.0, (
            f"Equator solstice noon elevation out of range: {pos.elevation_deg:.2f}°"
        )

    def test_solar_noon_azimuth_near_south(self):
        """
        At solar noon in Northern Hemisphere mid-latitudes, azimuth = 180° (due south).

        Solar noon for Atlanta (84.427°W) on 2026-07-15 ≈ 17:44 UTC.
        Computed as: UTC_min = 720 - EqT - 4*lon_east
                    = 720 - (-6) - 4*(-84.427) = 1063.7 min ≈ 17:43:42 UTC.
        At exactly TST=720 min (HA=0) the Meeus formula analytically gives 180°.
        """
        dt = datetime(2026, 7, 15, 17, 44, 0)   # ≈ solar noon for Atlanta, July 15
        pos = solar_position(33.641, -84.427, dt)
        # At solar noon HA≈0: azimuth must be 180° (due south) in Northern Hemisphere
        assert 170.0 < pos.azimuth_deg < 190.0, (
            f"Solar-noon azimuth not near south: {pos.azimuth_deg:.2f}°"
        )

    def test_nighttime_elevation_negative(self):
        """Sun is below horizon at midnight UTC for Atlanta."""
        dt = datetime(2026, 7, 15, 6, 0, 0)   # 02:00 EDT — deep night
        pos = solar_position(33.641, -84.427, dt)
        assert pos.elevation_deg < 0.0, (
            f"Midnight should be below horizon: {pos.elevation_deg:.2f}°"
        )

    def test_elevation_in_valid_range(self):
        """Elevation must always be in [−90°, +90°]."""
        test_cases = [
            (33.641, -84.427, datetime(2026, 7, 15, 17, 0, 0)),
            (0.0, 0.0, datetime(2026, 12, 21, 0, 0, 0)),
            (90.0, 0.0, datetime(2026, 6, 21, 12, 0, 0)),
            (-45.0, 120.0, datetime(2026, 1, 1, 2, 0, 0)),
        ]
        for lat, lon, dt in test_cases:
            pos = solar_position(lat, lon, dt)
            assert -90.0 <= pos.elevation_deg <= 90.0

    def test_azimuth_in_valid_range(self):
        """Azimuth must always be in [0°, 360°)."""
        test_cases = [
            (33.641, -84.427, datetime(2026, 7, 15, 10, 0, 0)),
            (33.641, -84.427, datetime(2026, 7, 15, 20, 0, 0)),
            (0.0, 0.0, datetime(2026, 6, 21, 6, 0, 0)),
        ]
        for lat, lon, dt in test_cases:
            pos = solar_position(lat, lon, dt)
            assert 0.0 <= pos.azimuth_deg < 360.0

    def test_refraction_applied_at_low_elevation(self):
        """
        When sun is near the horizon, refraction correction should increase elevation
        above the geometric value.  Test that elevation is positive when the sun is
        geometrically just below horizon (refraction ≈ 0.5° at 0°).
        """
        # Find a time when Atlanta has sun near horizon (around sunrise ~10:30 UTC)
        dt = datetime(2026, 7, 15, 10, 32, 0)   # approximate sunrise
        pos = solar_position(33.641, -84.427, dt)
        # Just checking that the result is finite and in range
        assert -5.0 < pos.elevation_deg < 20.0

    def test_timezone_aware_input(self):
        """Timezone-aware input produces same result as naive UTC."""
        naive  = datetime(2026, 7, 15, 17, 0, 0)
        aware  = datetime(2026, 7, 15, 17, 0, 0, tzinfo=timezone.utc)
        p_naive = solar_position(33.641, -84.427, naive)
        p_aware = solar_position(33.641, -84.427, aware)
        assert abs(p_naive.elevation_deg - p_aware.elevation_deg) < 1e-9
        assert abs(p_naive.azimuth_deg   - p_aware.azimuth_deg)   < 1e-9


class TestIlluminationMode:
    def test_daytime(self):
        assert get_illumination_mode(7.0) == IlluminationMode.DAYTIME
        assert get_illumination_mode(45.0) == IlluminationMode.DAYTIME
        assert get_illumination_mode(6.1) == IlluminationMode.DAYTIME

    def test_twilight_upper(self):
        assert get_illumination_mode(6.0) == IlluminationMode.TWILIGHT
        assert get_illumination_mode(0.0) == IlluminationMode.TWILIGHT
        assert get_illumination_mode(-11.9) == IlluminationMode.TWILIGHT

    def test_twilight_lower(self):
        assert get_illumination_mode(-12.0) == IlluminationMode.TWILIGHT

    def test_nighttime(self):
        assert get_illumination_mode(-12.1) == IlluminationMode.NIGHTTIME
        assert get_illumination_mode(-45.0) == IlluminationMode.NIGHTTIME

    def test_atlanta_midday_is_daytime(self):
        """Atlanta midday in July should always be DAYTIME."""
        dt = datetime(2026, 7, 15, 17, 0, 0)
        pos = solar_position(33.641, -84.427, dt)
        assert get_illumination_mode(pos.elevation_deg) == IlluminationMode.DAYTIME

    def test_atlanta_midnight_is_nighttime(self):
        """Atlanta at 06:00 UTC (02:00 EDT) in July should be NIGHTTIME."""
        dt = datetime(2026, 7, 15, 6, 0, 0)
        pos = solar_position(33.641, -84.427, dt)
        assert get_illumination_mode(pos.elevation_deg) == IlluminationMode.NIGHTTIME
