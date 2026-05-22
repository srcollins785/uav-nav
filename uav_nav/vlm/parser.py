"""
Parses VLM raw text responses into SemanticObservation.

Three-layer fallback strategy:
  1. Strict json.loads() on the full response.
  2. Regex extraction of the outermost {...} block (handles markdown fences).
  3. Return None on complete failure — caller treats missing obs as no-information.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from uav_nav.vlm.schemas import SemanticObservation

log = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_vlm_response(raw: str) -> Optional[SemanticObservation]:
    """
    Parse a raw VLM response string into a SemanticObservation.

    Returns None if the response cannot be parsed into a valid schema.
    """
    if not raw or not raw.strip():
        return None

    # Layer 1: strict parse
    data = _try_json(raw.strip())

    # Layer 2: extract first {...} block
    if data is None:
        match = _JSON_BLOCK_RE.search(raw)
        if match:
            data = _try_json(match.group(0))

    if data is None:
        log.warning("VLM response could not be parsed as JSON: %.200s", raw)
        return None

    # Validate against schema
    try:
        obs = SemanticObservation.model_validate(data)
        return obs
    except Exception as exc:
        log.warning("VLM JSON did not match SemanticObservation schema: %s", exc)
        # Layer 3: partial extraction — build a minimal valid object
        return _partial_extract(data)


def _try_json(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _partial_extract(data: dict) -> Optional[SemanticObservation]:
    """Best-effort construction from a partially valid dict."""
    try:
        # Keep only keys the schema recognizes
        safe = {}
        if "scene_type" in data:
            safe["scene_type"] = str(data["scene_type"])
        if "sky_visible" in data:
            safe["sky_visible"] = bool(data["sky_visible"])
        if "stars_visible" in data and isinstance(data["stars_visible"], list):
            safe["stars_visible"] = [str(s) for s in data["stars_visible"]]
        # Skip landmarks if they don't validate — safer than crashing
        if "landmarks" in data and isinstance(data["landmarks"], list):
            safe["landmarks"] = []
            for lm in data["landmarks"]:
                if isinstance(lm, dict) and "bearing_deg" in lm:
                    try:
                        from uav_nav.vlm.schemas import LandmarkObservation
                        safe["landmarks"].append(
                            LandmarkObservation(**{
                                k: v for k, v in lm.items()
                                if k in LandmarkObservation.model_fields
                            }).model_dump()
                        )
                    except Exception:
                        pass
        return SemanticObservation.model_validate(safe)
    except Exception as exc:
        log.debug("Partial extraction also failed: %s", exc)
        return None
