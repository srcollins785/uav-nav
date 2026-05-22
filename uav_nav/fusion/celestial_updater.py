"""
Bridges the star detection → celestial solver → UKF update pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from uav_nav.celestial.observation_builder import build_observations
from uav_nav.celestial.solver import get_celestial_fix, is_available, CelestialFix
from uav_nav.config import get_config
from uav_nav.navigation.state import UAVState
from uav_nav.navigation.ukf import UAVKalmanFilter
from uav_nav.vlm.star_detector import StarCandidate, detect_stars

log = logging.getLogger(__name__)


class CelestialUpdater:
    """
    Orchestrates the full star observation pipeline:
      image → star centroids → catalog match → celestial fix → UKF update
    """

    def __init__(self):
        self._cfg = get_config()

    def process(
        self,
        image_source,
        kf: UAVKalmanFilter,
        dt_utc: datetime,
        image_width_px: int = 1280,
        image_height_px: int = 720,
    ) -> Optional[CelestialFix]:
        """
        Run the full celestial update pipeline on one image frame.

        Returns the CelestialFix used (or None if no fix was applied).
        """
        if not is_available():
            return None

        state = kf.state

        # 1. Detect star centroids
        candidates: List[StarCandidate] = detect_stars(image_source)
        if not candidates:
            log.debug("No star candidates found in frame.")
            return None

        # 2. Match to catalog, build observations
        observations = build_observations(
            candidates=candidates,
            state=state,
            dt_utc=dt_utc,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        )
        if len(observations) < self._cfg.min_stars_for_fix:
            log.debug("Too few matched stars (%d < %d) for celestial fix.",
                      len(observations), self._cfg.min_stars_for_fix)
            return None

        # 3. Solve for position
        fix = get_celestial_fix(observations, dt_utc)
        if fix is None:
            return None

        # 4. Update UKF
        kf.update_celestial_fix(fix)
        return fix
