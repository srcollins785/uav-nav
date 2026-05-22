"""Unit tests for the star catalog module."""
import pytest
from pathlib import Path

from uav_nav.celestial.catalog import load_catalog, lookup, StarRecord


def test_catalog_loads_minimum_stars():
    catalog = load_catalog()
    assert len(catalog) >= 50, f"Expected >=50 navigational stars, got {len(catalog)}"


def test_polaris_present():
    record = lookup("Polaris")
    assert record is not None
    assert abs(record.dec_deg - 89.264) < 0.01, "Polaris declination should be ~89.26°"
    assert abs(record.ra_deg - 37.95) < 0.1, "Polaris RA should be ~37.95°"


def test_betelgeuse_present():
    record = lookup("Betelgeuse")
    assert record is not None
    assert abs(record.ra_deg - 88.79) < 0.1
    assert abs(record.dec_deg - 7.41) < 0.1


def test_sirius_is_brightest():
    catalog = load_catalog()
    assert catalog[0].name == "Sirius", "Sirius (magnitude -1.46) should be first"


def test_magnitude_filter():
    catalog = load_catalog(max_magnitude=2.0)
    assert all(s.magnitude <= 2.0 for s in catalog)
    assert len(catalog) > 0


def test_case_insensitive_lookup():
    assert lookup("polaris") is not None
    assert lookup("POLARIS") is not None
    assert lookup("pOlArIs") is not None


def test_unknown_star_returns_none():
    assert lookup("XYZ_NONEXISTENT_STAR_9999") is None


def test_sorted_by_magnitude():
    catalog = load_catalog()
    mags = [s.magnitude for s in catalog]
    assert mags == sorted(mags), "Catalog should be sorted brightest-first"
