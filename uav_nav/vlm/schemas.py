"""Pydantic schemas for VLM outputs."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SceneType(str, Enum):
    urban = "urban"
    rural = "rural"
    industrial = "industrial"
    airport = "airport"
    open_water = "open_water"
    unknown = "unknown"


class DistanceCategory(str, Enum):
    near = "near"       # < 500 m
    mid = "mid"         # 500 m – 2 km
    far = "far"         # > 2 km


class LandmarkType(str, Enum):
    transformer_substation = "transformer_substation"
    transmission_line = "transmission_line"
    road_intersection = "road_intersection"
    highway = "highway"
    ridge_line = "ridge_line"
    water_body = "water_body"
    building_cluster = "building_cluster"
    airport = "airport"
    unknown = "unknown"


# Approximate distance priors (meters) per landmark type
LANDMARK_DISTANCE_PRIOR_M: dict[LandmarkType, tuple[float, float]] = {
    LandmarkType.transformer_substation: (100.0, 1000.0),
    LandmarkType.transmission_line:      (50.0, 500.0),
    LandmarkType.road_intersection:      (100.0, 800.0),
    LandmarkType.highway:                (100.0, 2000.0),
    LandmarkType.ridge_line:             (1000.0, 15000.0),
    LandmarkType.water_body:             (200.0, 10000.0),
    LandmarkType.building_cluster:       (200.0, 3000.0),
    LandmarkType.airport:                (500.0, 10000.0),
    LandmarkType.unknown:                (100.0, 5000.0),
}


class LandmarkObservation(BaseModel):
    type: LandmarkType = LandmarkType.unknown
    bearing_deg: float = Field(ge=0.0, lt=360.0)
    distance_category: DistanceCategory = DistanceCategory.mid
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    notes: Optional[str] = None
    apparent_height_px: Optional[float] = None  # landmark apparent pixel height → enables monocular range

    @field_validator("bearing_deg", mode="before")
    @classmethod
    def wrap_bearing(cls, v):
        return float(v) % 360.0

    def distance_prior_m(self) -> float:
        """Return mid-point of the distance range for this landmark type."""
        lo, hi = LANDMARK_DISTANCE_PRIOR_M[self.type]
        return (lo + hi) / 2.0


class SemanticObservation(BaseModel):
    """Full structured output from one VLM query."""
    scene_type: SceneType = SceneType.unknown
    sky_visible: bool = False
    landmarks: List[LandmarkObservation] = Field(default_factory=list)
    stars_visible: List[str] = Field(default_factory=list)
    horizon_features: Optional[str] = None

    def high_confidence_landmarks(self, min_conf: float = 0.70) -> List[LandmarkObservation]:
        return [lm for lm in self.landmarks if lm.confidence >= min_conf]
