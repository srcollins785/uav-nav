"""
Regression test for the celestial solver (inverse_celnav.py wrapper).

Uses the Polaris + Betelgeuse observations from example_observations.json
(Atlanta, GA area, 2026-01-28T02:15:30Z) and asserts the solver converges
within 5 degrees of Atlanta (33.75°N, 84.39°W).
"""
import pytest
from datetime import datetime, timezone

from uav_nav.celestial.solver import get_celestial_fix, is_available

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="inverse_celnav not on sys.path — skipping celestial solver tests"
)


def _make_observations():
    """
    Compute accurate alt/az observations from the true Atlanta position.

    Using computed (not approximate) values ensures the solver has a
    self-consistent test case and must recover the correct location.
    """
    from uav_nav.celestial.solver import Observation, altaz_from_radec
    dt = datetime(2026, 1, 28, 2, 15, 30, tzinfo=timezone.utc)
    true_lat, true_lon = 33.641, -84.427

    stars = [
        ("Polaris",    37.95456067,  89.26410897),
        ("Betelgeuse", 88.792939,     7.407064),
        ("Sirius",    101.287155,   -16.716116),
    ]
    obs = []
    for name, ra, dec in stars:
        alt, az = altaz_from_radec(true_lat, true_lon, dt, ra, dec)
        if alt > 5.0:
            obs.append(Observation(name, ra, dec, alt, az, 0.3, 1.5))
    return obs


def test_solver_returns_fix():
    dt = datetime(2026, 1, 28, 2, 15, 30, tzinfo=timezone.utc)
    obs = _make_observations()
    fix = get_celestial_fix(obs, dt)
    assert fix is not None


def test_solver_converges():
    dt = datetime(2026, 1, 28, 2, 15, 30, tzinfo=timezone.utc)
    fix = get_celestial_fix(_make_observations(), dt)
    assert fix.optimizer_success, "Optimizer should converge on this well-conditioned example"


def test_solver_atlanta_latitude():
    """Solved latitude should be within 5 degrees of Atlanta (33.75°N)."""
    dt = datetime(2026, 1, 28, 2, 15, 30, tzinfo=timezone.utc)
    fix = get_celestial_fix(_make_observations(), dt)
    assert fix is not None
    assert abs(fix.lat_deg - 33.75) < 5.0, (
        f"Expected lat near 33.75°N, got {fix.lat_deg:.2f}°"
    )


def test_solver_atlanta_longitude():
    """Solved longitude should be within 5 degrees of Atlanta (-84.39°E)."""
    dt = datetime(2026, 1, 28, 2, 15, 30, tzinfo=timezone.utc)
    fix = get_celestial_fix(_make_observations(), dt)
    assert fix is not None
    assert abs(fix.lon_deg_east - (-84.39)) < 5.0, (
        f"Expected lon near -84.39°, got {fix.lon_deg_east:.2f}°"
    )


def test_solver_empty_observations_returns_none():
    dt = datetime(2026, 1, 28, 2, 15, 30, tzinfo=timezone.utc)
    fix = get_celestial_fix([], dt)
    assert fix is None


def test_solver_no_azimuth_mode():
    """Altitude-only mode still converges with 2+ observations."""
    dt = datetime(2026, 1, 28, 2, 15, 30, tzinfo=timezone.utc)
    fix = get_celestial_fix(_make_observations(), dt, use_azimuth=False)
    assert fix is not None
    assert not fix.used_azimuth
