#!/usr/bin/env python3
"""
Inverse Celestial Navigation (stars only, offline)

Given:
  - UTC timestamp
  - Identified celestial objects (stars) with fixed RA/Dec
  - Measured altitude (elevation above horizon) and optionally azimuth (true)

Solve for observer latitude/longitude by minimizing residual between
measured and predicted (alt, az).

Input format: JSON
{
  "utc": "2026-01-28T02:15:30Z",
  "observations": [
    {
      "name": "Polaris",
      "ra_deg": 37.95456067,
      "dec_deg": 89.26410897,
      "alt_deg": 33.12,
      "az_deg": 0.8,
      "sigma_alt_deg": 0.2,
      "sigma_az_deg": 1.0
    }
  ],
  "options": {
    "use_azimuth": true,
    "grid_lat_step_deg": 10,
    "grid_lon_step_deg": 20
  }
}

Notes:
- Longitude convention: East-positive. Output includes both E-positive and W-positive forms.
- If you only have altitude (no azimuth), you generally need >=2 observations at distinct azimuths/times.
- This implementation ignores refraction, dip, and star proper motion. Add corrections if you need high accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


# ----------------------------
# Time / astronomy utilities
# ----------------------------

def parse_utc(s: str) -> datetime:
    # Accept "Z" or "+00:00"
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # Assume UTC if missing TZ info
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def julian_date(dt_utc: datetime) -> float:
    """Julian Date for UTC datetime (navigation-grade)."""
    dt = dt_utc.astimezone(timezone.utc)
    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond/1e6)/60.0)/60.0)/24.0

    if month <= 2:
        year -= 1
        month += 12

    A = math.floor(year/100)
    B = 2 - A + math.floor(A/4)

    JD = math.floor(365.25*(year + 4716)) + math.floor(30.6001*(month + 1)) + day + B - 1524.5
    return JD

def gmst_deg(dt_utc: datetime) -> float:
    """Greenwich Mean Sidereal Time (degrees)."""
    JD = julian_date(dt_utc)
    T = (JD - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (JD - 2451545.0)
        + 0.000387933 * T*T
        - (T*T*T) / 38710000.0
    )
    return gmst % 360.0


# ----------------------------
# Spherical astronomy
# ----------------------------

def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def wrap360(deg: float) -> float:
    return deg % 360.0

def altaz_from_radec(
    lat_deg: float,
    lon_deg_east: float,
    dt_utc: datetime,
    ra_deg: float,
    dec_deg: float,
) -> Tuple[float, float]:
    """
    Convert RA/Dec to predicted Alt/Az (degrees) for an observer at lat/lon and time.

    Azimuth: degrees from TRUE north, increasing eastward (0=N, 90=E).
    """
    lat = math.radians(lat_deg)
    dec = math.radians(dec_deg)

    lst = math.radians(wrap360(gmst_deg(dt_utc) + lon_deg_east))
    ra = math.radians(wrap360(ra_deg))
    H = lst - ra  # hour angle (radians)

    # Altitude
    sin_alt = math.sin(lat)*math.sin(dec) + math.cos(lat)*math.cos(dec)*math.cos(H)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.asin(sin_alt)

    # Azimuth via atan2 (robust)
    y = -math.sin(H)
    x = math.tan(dec)*math.cos(lat) - math.sin(lat)*math.cos(H)
    az = math.atan2(y, x)

    return math.degrees(alt), wrap360(math.degrees(az))


# ----------------------------
# Optimization model
# ----------------------------

@dataclass
class Observation:
    name: str
    ra_deg: float
    dec_deg: float
    alt_deg: float
    az_deg: Optional[float] = None
    sigma_alt_deg: float = 0.25
    sigma_az_deg: float = 1.0

def residuals(
    x: np.ndarray,
    dt_utc: datetime,
    obs: List[Observation],
    use_azimuth: bool,
) -> np.ndarray:
    lat, lon = float(x[0]), float(x[1])
    res = []
    for o in obs:
        alt_p, az_p = altaz_from_radec(lat, lon, dt_utc, o.ra_deg, o.dec_deg)
        res.append((alt_p - o.alt_deg) / max(o.sigma_alt_deg, 1e-6))
        if use_azimuth and (o.az_deg is not None):
            daz = wrap180(az_p - o.az_deg)
            res.append(daz / max(o.sigma_az_deg, 1e-6))
    return np.array(res, dtype=float)

def coarse_grid_start(
    dt_utc: datetime,
    obs: List[Observation],
    use_azimuth: bool,
    lat_step: float = 10.0,
    lon_step: float = 20.0,
):
    best_sse = None
    best_xy = (0.0, 0.0)
    for lat in np.arange(-80.0, 80.0 + 1e-9, lat_step):
        for lon in np.arange(-180.0, 180.0 + 1e-9, lon_step):
            r = residuals(np.array([lat, lon]), dt_utc, obs, use_azimuth)
            sse = float(r @ r)
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_xy = (float(lat), float(lon))
    return best_xy

def solve_position(
    dt_utc: datetime,
    obs: List[Observation],
    use_azimuth: bool = True,
    grid_lat_step_deg: float = 10.0,
    grid_lon_step_deg: float = 20.0,
) -> dict:
    if not obs:
        raise ValueError("No observations provided.")

    if use_azimuth and all(o.az_deg is None for o in obs):
        use_azimuth = False

    if not use_azimuth and len(obs) < 2:
        raise ValueError("Altitude-only solution requires at least 2 observations.")

    lat0, lon0 = coarse_grid_start(dt_utc, obs, use_azimuth, grid_lat_step_deg, grid_lon_step_deg)

    lb = np.array([-90.0, -180.0], dtype=float)
    ub = np.array([ 90.0,  180.0], dtype=float)

    lsq = least_squares(
        residuals,
        x0=np.array([lat0, lon0], dtype=float),
        bounds=(lb, ub),
        args=(dt_utc, obs, use_azimuth),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=2000,
    )

    lat, lon = float(lsq.x[0]), float(lsq.x[1])
    r = residuals(lsq.x, dt_utc, obs, use_azimuth)
    rmse = float(math.sqrt(float(np.mean(r*r))))

    return {
        "latitude_deg": lat,
        "longitude_deg_east": lon,
        "longitude_deg_west": -lon,
        "used_azimuth": bool(use_azimuth),
        "initial_guess": {"lat_deg": lat0, "lon_deg_east": lon0},
        "rmse_normalized": rmse,
        "optimizer": {
            "success": bool(lsq.success),
            "status": int(lsq.status),
            "message": str(lsq.message),
            "nfev": int(lsq.nfev),
            "cost": float(lsq.cost),
        },
    }


# ----------------------------
# CLI
# ----------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dt = parse_utc(data["utc"])
    options = data.get("options", {}) or {}

    observations = []
    for o in data["observations"]:
        observations.append(
            Observation(
                name=str(o.get("name", "object")),
                ra_deg=float(o["ra_deg"]),
                dec_deg=float(o["dec_deg"]),
                alt_deg=float(o["alt_deg"]),
                az_deg=None if o.get("az_deg", None) is None else float(o["az_deg"]),
                sigma_alt_deg=float(o.get("sigma_alt_deg", 0.25)),
                sigma_az_deg=float(o.get("sigma_az_deg", 1.0)),
            )
        )
    return dt, observations, options

def main():
    ap = argparse.ArgumentParser(description="Inverse Celestial Navigation (stars): solve lat/lon from alt/az observations.")
    ap.add_argument("input_json", help="Path to JSON input file.")
    ap.add_argument("--no-az", action="store_true", help="Ignore azimuth even if provided (alt-only).")
    args = ap.parse_args()

    dt, obs, options = load_json(args.input_json)

    use_az = bool(options.get("use_azimuth", True))
    if args.no_az:
        use_az = False

    sol = solve_position(
        dt_utc=dt,
        obs=obs,
        use_azimuth=use_az,
        grid_lat_step_deg=float(options.get("grid_lat_step_deg", 10.0)),
        grid_lon_step_deg=float(options.get("grid_lon_step_deg", 20.0)),
    )
    print(json.dumps(sol, indent=2))

if __name__ == "__main__":
    main()
