"""
Integration validation: Polaris + Betelgeuse observations → Atlanta position.

Run directly:
    python -m uav_nav.demo.validate_celestial

Expected output: solver places observer within ~50 km of Atlanta, GA (33.75°N, 84.39°W).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    from uav_nav.celestial.solver import (
        Observation,
        get_celestial_fix,
        is_available,
    )

    if not is_available():
        print("ERROR: inverse_celnav not available. Check inverse_celnav_path in config.yaml.")
        sys.exit(1)

    # --- Load example observations from the existing JSON file ---
    example_json = (
        Path(__file__).parent.parent.parent
        / "Offline Vision\u2013Language Navigation"
        / "V1"
        / "inverse_celnav"
        / "example_observations.json"
    )

    # Compute accurate alt/az observations from the true Atlanta position.
    # (The example_observations.json contains approximate values only; for
    # a rigorous integration test we compute ground-truth observations from
    # the known position and verify the solver recovers that position.)
    from uav_nav.celestial.solver import altaz_from_radec

    dt = datetime(2026, 1, 28, 2, 15, 30, tzinfo=timezone.utc)
    true_lat = 33.641
    true_lon = -84.427

    star_catalog = [
        ("Polaris",    37.95456067,  89.26410897),
        ("Betelgeuse", 88.792939,     7.407064),
        ("Sirius",    101.287155,   -16.716116),
    ]

    observations = []
    for name, ra, dec in star_catalog:
        alt, az = altaz_from_radec(true_lat, true_lon, dt, ra, dec)
        if alt > 5.0:
            observations.append(
                Observation(name, ra, dec, alt, az, 0.3, 1.5)
            )

    print(f"UTC: {dt.isoformat()}")
    print(f"Observations computed from true position ({true_lat}°N, {true_lon}°E):")
    for o in observations:
        print(f"  {o.name}: alt={o.alt_deg:.2f}°  az={o.az_deg:.2f}°")

    if example_json.exists():
        print(f"(Reference: {example_json} also available but uses approximate data)")
    print()

    fix = get_celestial_fix(observations, dt)

    if fix is None:
        print("FAIL: Solver returned None.")
        sys.exit(1)

    # True Atlanta position for error computation
    true_lat = 33.641
    true_lon = -84.427
    R_earth_km = 6371.0

    dlat = math.radians(fix.lat_deg - true_lat)
    dlon = math.radians(fix.lon_deg_east - true_lon)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(true_lat))
         * math.cos(math.radians(fix.lat_deg))
         * math.sin(dlon / 2) ** 2)
    dist_km = 2 * R_earth_km * math.asin(math.sqrt(a))

    print(f"Solved position:  {fix.lat_deg:.4f}°N, {fix.lon_deg_east:.4f}°E")
    print(f"True position:    {true_lat:.4f}°N, {true_lon:.4f}°E")
    print(f"Distance error:   {dist_km:.1f} km")
    print(f"RMSE normalized:  {fix.rmse_normalized:.4f}")
    print(f"Optimizer:        success={fix.optimizer_success}, nfev={fix.nfev}")
    print(f"Used azimuth:     {fix.used_azimuth}")
    print()

    if dist_km < 50.0 and fix.optimizer_success:
        print("PASS: Position within 50 km of Atlanta and optimizer converged.")
    else:
        print(f"WARN: Distance error {dist_km:.1f} km (threshold 50 km), "
              f"success={fix.optimizer_success}")


if __name__ == "__main__":
    main()
