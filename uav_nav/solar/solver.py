"""
Solar navigation position solver.

Mirrors uav_nav/celestial/solver.py but uses a single solar observation
(altitude + azimuth of the sun) instead of a multi-star catalog fix.

The sun is treated as a point source whose predicted position is given by
uav_nav.solar.ephemeris.solar_position(lat, lon, dt_utc).  A scipy
least_squares minimisation over the residuals [Δelev/σ_elev, Δaz/σ_az]
solves for the observer position.

GDOP is computed identically to the night solver (sqrt of the trace of the
inverse normal matrix J^T J) and auto-suppresses the Kalman gain when solar
geometry is poor (sun near zenith → azimuth degenerate → GDOP spikes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from uav_nav.config import get_config
from uav_nav.solar.ephemeris import solar_position

log = logging.getLogger(__name__)

# Physically realistic minimum sigmas (same role as _MIN_SIGMA_* in celestial solver)
_MIN_SIGMA_ELEV: float = 0.15   # degrees — sun disk centroid precision
_MIN_SIGMA_AZ:   float = 0.25   # degrees — better than star azimuth due to large disk


@dataclass
class SolarObservation:
    """
    A single solar altitude/azimuth measurement.

    elevation_deg   : Atmospherically-corrected altitude above horizon (degrees).
    azimuth_deg     : True azimuth, 0=N clockwise=E (degrees).
    sigma_elev_deg  : 1-sigma uncertainty on elevation (degrees).
    sigma_az_deg    : 1-sigma uncertainty on azimuth (degrees).
    dt_utc          : UTC epoch of the measurement (stored for solver closure).
    """
    elevation_deg: float
    azimuth_deg: float
    sigma_elev_deg: float = _MIN_SIGMA_ELEV
    sigma_az_deg: float   = _MIN_SIGMA_AZ
    dt_utc: Optional[datetime] = None


@dataclass
class SolarFix:
    """
    Lat/lon position fix derived from a solar observation.

    Parallel to CelestialFix — same five fields consumed by update_solar_fix().
    """
    lat_deg: float
    lon_deg_east: float
    rmse_normalized: float
    optimizer_success: bool
    nfev: int
    position_gdop: float = 1.0
    """
    Geometric Dilution of Precision (degrees).

    Computed as sqrt(trace((J^T J)^-1)) at the converged solution.
    When the sun is near the zenith, azimuth becomes undefined and GDOP
    spikes, automatically suppressing the Kalman gain.  Capped at 10°.
    """

    @property
    def lon_deg(self) -> float:
        """Alias for compatibility with celestial fix consumers."""
        return self.lon_deg_east


def get_solar_fix_warmstart(
    obs: SolarObservation,
    dt_utc: datetime,
    init_lat_deg: float,
    init_lon_deg: float,
    search_radius_deg: float = 3.0,
) -> Optional[SolarFix]:
    """
    Solve for observer lat/lon from a single solar observation.

    Uses the current navigation estimate as a warm start.  A local grid search
    over ±search_radius_deg finds the basin of attraction, then scipy
    least_squares refines to sub-0.1° accuracy.

    The search radius is wider than the night solver's 2° to compensate for the
    single-object geometry being more ambiguous than a multi-star fix.

    Args:
        obs              : Measured solar elevation and azimuth.
        dt_utc           : UTC epoch of the observation.
        init_lat_deg     : Current estimated latitude (degrees).
        init_lon_deg     : Current estimated longitude (degrees East).
        search_radius_deg: Half-width of the local search window (default 3°).

    Returns:
        SolarFix if the optimizer converges and GDOP is acceptable, else None.
    """
    cfg = get_config()

    def _residuals(x: np.ndarray) -> np.ndarray:
        lat, lon = float(x[0]), float(x[1])
        pred = solar_position(lat, lon, dt_utc)
        s_elev = max(obs.sigma_elev_deg, _MIN_SIGMA_ELEV)
        r_elev = (obs.elevation_deg - pred.elevation_deg) / s_elev
        # Wrap azimuth residual to (−180, +180]
        daz = ((obs.azimuth_deg - pred.azimuth_deg + 180.0) % 360.0) - 180.0
        s_az = max(obs.sigma_az_deg, _MIN_SIGMA_AZ)
        r_az = daz / s_az
        return np.array([r_elev, r_az], dtype=float)

    try:
        # Local grid search to find the best seed for the optimiser
        lat_lo = max(-80.0, init_lat_deg - search_radius_deg)
        lat_hi = min( 80.0, init_lat_deg + search_radius_deg)
        lon_lo = init_lon_deg - 2.0 * search_radius_deg
        lon_hi = init_lon_deg + 2.0 * search_radius_deg

        best_cost = float("inf")
        best_seed = np.array([init_lat_deg, init_lon_deg])

        for lat in np.arange(lat_lo, lat_hi + 0.05, 0.1):
            for lon in np.arange(lon_lo, lon_hi + 0.1, 0.2):
                r = _residuals(np.array([lat, lon]))
                cost = float(np.dot(r, r))
                if cost < best_cost:
                    best_cost = cost
                    best_seed = np.array([lat, lon])

        result = least_squares(
            _residuals, best_seed,
            method="trf", loss="soft_l1",
            bounds=([-80.0, -180.0], [80.0, 180.0]),
        )

        rmse = float(np.sqrt(np.mean(result.fun ** 2)))

        # GDOP from normal matrix (identical to celestial/solver.py lines 238-245)
        try:
            J = np.asarray(result.jac)
            JtJ = J.T @ J
            C = np.linalg.inv(JtJ)
            gdop = float(np.sqrt(abs(C[0, 0]) + abs(C[1, 1])))
            gdop = min(gdop, 10.0)   # cap — degenerate near-zenith case
        except (np.linalg.LinAlgError, ValueError):
            gdop = 10.0              # singular matrix → worst-case GDOP

        # Hard-reject if solar geometry is too poor (sun near zenith)
        solar_gdop_max = getattr(cfg, "solar_gdop_max", 5.0)
        if gdop > solar_gdop_max:
            log.debug(
                "Solar fix rejected: GDOP=%.2f > threshold=%.2f (sun near zenith)",
                gdop, solar_gdop_max,
            )
            return None

        fix = SolarFix(
            lat_deg=float(result.x[0]),
            lon_deg_east=float(result.x[1]),
            rmse_normalized=rmse,
            optimizer_success=bool(result.success),
            nfev=int(result.nfev),
            position_gdop=gdop,
        )
        log.debug(
            "Solar warm-start fix: lat=%.4f lon=%.4f rmse=%.3f gdop=%.3f nfev=%d",
            fix.lat_deg, fix.lon_deg_east, fix.rmse_normalized, fix.position_gdop, fix.nfev,
        )
        return fix

    except Exception as exc:
        log.warning("Solar solver failed: %s", exc)
        return None
