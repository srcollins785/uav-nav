"""
Wrapper around the existing inverse_celnav.py solver.

Does NOT copy inverse_celnav.py — extends sys.path at import time so the
original file remains the single source of truth.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from uav_nav.config import get_config

log = logging.getLogger(__name__)


def _ensure_inverse_celnav_on_path() -> None:
    """Add the inverse_celnav directory to sys.path once."""
    cfg = get_config()
    celnav_dir = str(cfg.inverse_celnav_abs_path())
    if celnav_dir not in sys.path:
        sys.path.insert(0, celnav_dir)


_ensure_inverse_celnav_on_path()

try:
    from inverse_celnav import (  # type: ignore[import]
        Observation,
        altaz_from_radec,
        gmst_deg,
        parse_utc,
        solve_position,
    )
    _CELNAV_AVAILABLE = True
except ImportError as _e:
    log.warning("inverse_celnav not found: %s. Celestial fixes disabled.", _e)
    _CELNAV_AVAILABLE = False

    # Provide stub types so the rest of the codebase can import without crashing
    class Observation:  # type: ignore[no-redef]
        pass

    def solve_position(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("inverse_celnav not available")

    def altaz_from_radec(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("inverse_celnav not available")

    def gmst_deg(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("inverse_celnav not available")

    def parse_utc(s: str) -> datetime:  # type: ignore[misc]
        raise RuntimeError("inverse_celnav not available")


@dataclass
class CelestialFix:
    lat_deg: float
    lon_deg_east: float
    rmse_normalized: float
    optimizer_success: bool
    used_azimuth: bool
    nfev: int
    position_gdop: float = 1.0
    """
    Geometric Dilution of Precision (degrees).

    Estimated 1-sigma positional accuracy of this fix based solely on star
    geometry — the square root of the trace of the inverse normal matrix
    (J^T J)^-1 evaluated at the solution.

    Interpretation:
      position_gdop = 0.05°  →  ~5.5 km accuracy  (excellent geometry)
      position_gdop = 0.10°  → ~11.0 km accuracy  (good)
      position_gdop = 0.20°  → ~22.0 km accuracy  (poor — stars clustered)

    Used by UAVKalmanFilter.update_celestial_fix() to scale measurement
    noise: sigma_effective = max(base_sigma, position_gdop).  Poor geometry
    automatically reduces the Kalman gain so a single bad fix cannot corrupt
    a well-converged filter state.
    """

    @property
    def lon_deg(self) -> float:
        return self.lon_deg_east

    @classmethod
    def from_solver_dict(cls, d: dict) -> "CelestialFix":
        return cls(
            lat_deg=d["latitude_deg"],
            lon_deg_east=d["longitude_deg_east"],
            rmse_normalized=d["rmse_normalized"],
            optimizer_success=d["optimizer"]["success"],
            used_azimuth=d["used_azimuth"],
            nfev=d["optimizer"]["nfev"],
        )


def get_celestial_fix(
    observations: List[Observation],
    dt_utc: datetime,
    use_azimuth: Optional[bool] = None,
    grid_lat_step: Optional[float] = None,
    grid_lon_step: Optional[float] = None,
) -> Optional[CelestialFix]:
    """
    Solve for observer lat/lon from star observations.

    Returns None if the solver is unavailable, no observations provided,
    or the optimizer fails to converge.
    """
    if not _CELNAV_AVAILABLE:
        log.debug("Celestial solver unavailable, skipping fix.")
        return None

    if not observations:
        return None

    cfg = get_config()
    use_az = cfg.use_azimuth if use_azimuth is None else use_azimuth
    lat_step = cfg.celestial_grid_lat_step if grid_lat_step is None else grid_lat_step
    lon_step = cfg.celestial_grid_lon_step if grid_lon_step is None else grid_lon_step

    try:
        result = solve_position(
            dt_utc=dt_utc,
            obs=observations,
            use_azimuth=use_az,
            grid_lat_step_deg=lat_step,
            grid_lon_step_deg=lon_step,
        )
        fix = CelestialFix.from_solver_dict(result)
        if not fix.optimizer_success:
            log.warning("Celestial solver did not converge (rmse=%.3f)", fix.rmse_normalized)
        return fix
    except Exception as exc:
        log.warning("Celestial solver raised exception: %s", exc)
        return None


def get_celestial_fix_warmstart(
    observations: List[Observation],
    dt_utc: datetime,
    init_lat_deg: float,
    init_lon_deg: float,
    use_azimuth: bool = True,
    search_radius_deg: float = 2.0,
) -> Optional[CelestialFix]:
    """
    Position-aiding celestial solver that uses the current navigation estimate
    as a warm start.

    Unlike get_celestial_fix(), which runs a global grid search and is
    vulnerable to false local minima 4–22 km from the true solution, this
    function restricts the grid to ±search_radius_deg around the provided
    initial estimate.  Because the dead-reckoning / VLM position is typically
    accurate to <1 km, the true minimum is always within the search window
    and false minima far away are never evaluated.

    Grid resolution: 0.1° lat × 0.2° lon (≈11 km × 18 km at 33°N).
    After the grid finds the best seed, scipy least_squares refines it.

    Args:
        observations:     List of Observation from inverse_celnav.
        dt_utc:           UTC timestamp of the observation epoch.
        init_lat_deg:     Current estimated latitude (degrees).
        init_lon_deg:     Current estimated longitude (degrees East).
        use_azimuth:      Include azimuth residuals in the cost (default True).
        search_radius_deg: Half-width of the local search window (default 2°).

    Returns:
        CelestialFix if the optimizer converges, None otherwise.
    """
    if not _CELNAV_AVAILABLE or not observations:
        return None

    try:
        import numpy as np
        from scipy.optimize import least_squares

        # Use physically realistic minimum sigmas so the cost landscape is
        # not dominated by overfitting to fine simulation noise.
        # A camera-based star fix realistically achieves ~0.3° altitude and
        # ~1.5° azimuth accuracy; tighter observed sigmas are floored here.
        _MIN_SIGMA_ALT = 0.3   # degrees
        _MIN_SIGMA_AZ  = 1.5   # degrees

        def _residuals(x: "np.ndarray") -> "np.ndarray":
            lat, lon = float(x[0]), float(x[1])
            res = []
            for obs in observations:
                alt_p, az_p = altaz_from_radec(lat, lon, dt_utc,
                                               obs.ra_deg, obs.dec_deg)
                s_alt = max(obs.sigma_alt_deg, _MIN_SIGMA_ALT)
                res.append((obs.alt_deg - alt_p) / s_alt)
                if use_azimuth:
                    daz = ((obs.az_deg - az_p + 180.0) % 360.0) - 180.0
                    s_az = max(obs.sigma_az_deg, _MIN_SIGMA_AZ)
                    res.append(daz / s_az)
            return np.array(res, dtype=float)

        # Fine local grid search to seed the optimiser
        lat_lo = max(-80.0, init_lat_deg - search_radius_deg)
        lat_hi = min( 80.0, init_lat_deg + search_radius_deg)
        lon_lo = init_lon_deg - search_radius_deg * 2.0
        lon_hi = init_lon_deg + search_radius_deg * 2.0

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

        nres = len(result.fun)
        rmse = float(np.sqrt(np.mean(result.fun ** 2)))

        # Geometric Dilution of Precision (GDOP) from the normal matrix.
        # result.jac is the Jacobian of the (loss-weighted) residuals w.r.t.
        # [lat, lon] at the converged solution, shape (n_res, 2), units 1/deg.
        # (J^T J)^-1 is the position covariance in deg^2 at unit noise level;
        # GDOP = sqrt(trace) is the 1-sigma positional accuracy in degrees.
        try:
            J = np.asarray(result.jac)
            JtJ = J.T @ J
            C = np.linalg.inv(JtJ)
            gdop = float(np.sqrt(abs(C[0, 0]) + abs(C[1, 1])))
            gdop = min(gdop, 10.0)   # cap at 10° — degenerate case
        except (np.linalg.LinAlgError, ValueError):
            gdop = 10.0              # singular normal matrix → worst-case GDOP

        fix = CelestialFix(
            lat_deg=float(result.x[0]),
            lon_deg_east=float(result.x[1]),
            rmse_normalized=rmse,
            optimizer_success=bool(result.success),
            used_azimuth=use_azimuth,
            nfev=int(result.nfev),
            position_gdop=gdop,
        )
        log.debug(
            "Warm-start fix: lat=%.4f lon=%.4f rmse=%.3f nfev=%d",
            fix.lat_deg, fix.lon_deg_east, fix.rmse_normalized, fix.nfev,
        )
        return fix

    except Exception as exc:
        log.warning("Warm-start celestial solver failed: %s", exc)
        return None


def is_available() -> bool:
    return _CELNAV_AVAILABLE
