"""Unit tests for the SemanticMap and landmark triangulation."""
import math
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from uav_nav.fusion.constraint_converter import ConstraintConverter, _triangulate, _haversine_m
from uav_nav.navigation.state import UAVState
from uav_nav.vlm.schemas import (
    LandmarkObservation, LandmarkType, SemanticObservation,
    SceneType, DistanceCategory,
)


def _make_state(lat=33.641, lon=-84.427, heading=0.0) -> UAVState:
    return UAVState(lat_deg=lat, lon_deg=lon, heading_deg=heading)


def _make_obs(lm_type: LandmarkType, bearing_deg: float, conf: float = 0.85) -> SemanticObservation:
    return SemanticObservation(
        scene_type=SceneType.industrial,
        sky_visible=False,
        landmarks=[
            LandmarkObservation(
                type=lm_type,
                bearing_deg=bearing_deg,
                distance_category=DistanceCategory.mid,
                confidence=conf,
            )
        ],
    )


def _make_mock_kf(state: UAVState) -> MagicMock:
    kf = MagicMock()
    kf.state = state
    return kf


def test_first_sighting_stored_as_pending():
    conv = ConstraintConverter()
    state = _make_state()
    kf = _make_mock_kf(state)
    obs = _make_obs(LandmarkType.transformer_substation, bearing_deg=45.0)
    updates = conv.process(obs, kf)
    assert updates == 0
    assert conv.pending_count == 1
    kf.update_vlm_bearing.assert_not_called()


def test_second_sighting_no_baseline_does_not_triangulate():
    conv = ConstraintConverter()
    state = _make_state()
    kf = _make_mock_kf(state)
    obs = _make_obs(LandmarkType.transformer_substation, bearing_deg=45.0)
    conv.process(obs, kf)
    # Second sighting at same position (baseline = 0)
    conv.process(obs, kf)
    assert len(conv.known_landmarks) == 0
    kf.update_vlm_bearing.assert_not_called()


def test_second_sighting_after_movement_triggers_triangulation():
    conv = ConstraintConverter()

    # First sighting from (33.641, -84.427) with heading=0, bearing=90 → landmark due East
    state1 = _make_state(lat=33.641, lon=-84.427, heading=0.0)
    kf = _make_mock_kf(state1)
    obs1 = _make_obs(LandmarkType.transformer_substation, bearing_deg=90.0)
    conv.process(obs1, kf)

    # Move ~1 km north, observe landmark again at bearing ~130° (SE)
    state2 = _make_state(lat=33.650, lon=-84.427, heading=0.0)
    kf.state = state2
    obs2 = _make_obs(LandmarkType.transformer_substation, bearing_deg=130.0)
    updates = conv.process(obs2, kf)

    assert updates >= 1, "Should apply at least one UKF update after triangulation"
    assert LandmarkType.transformer_substation in conv.known_landmarks
    assert conv.pending_count == 0


def test_known_landmark_triggers_direct_update():
    conv = ConstraintConverter()

    # Plant a known landmark directly into the map.  UAV is at (33.700, -84.500)
    # looking heading=0; landmark is at (33.641, -84.427) — roughly SE.  The
    # observed camera bearing of 135° (SE of forward) must match the predicted
    # bearing within the 30° association gate.
    from uav_nav.fusion.constraint_converter import _SemanticEntry
    conv._known[LandmarkType.airport] = [_SemanticEntry(
        lat_deg=33.641, lon_deg=-84.427, landmark_type=LandmarkType.airport
    )]

    state = _make_state(lat=33.700, lon=-84.500, heading=0.0)
    kf = _make_mock_kf(state)
    obs = _make_obs(LandmarkType.airport, bearing_deg=135.0, conf=0.9)
    updates = conv.process(obs, kf)
    assert updates == 1
    kf.update_vlm_bearing.assert_called_once()


def test_known_landmark_association_picks_correct_bearing():
    """
    Two same-type landmarks simultaneously in view.  The one at the observed
    bearing must be selected, not the one that's merely closer in position.
    """
    from uav_nav.fusion.constraint_converter import _SemanticEntry

    conv = ConstraintConverter()
    # UAV at origin, heading north.  Two substations:
    #   lm_east  — 1 km east of UAV (true bearing 90°)
    #   lm_north — 1.5 km north of UAV (true bearing 0°; closer to UAV in
    #              straight-line for this setup? no — 1 km < 1.5 km, so
    #              nearest-by-position picks lm_east, but if observed bearing
    #              is 0° the correct association is lm_north).
    conv._known[LandmarkType.transformer_substation] = [
        _SemanticEntry(lat_deg=0.0, lon_deg=0.009,        # east: ~1.0 km
                       landmark_type=LandmarkType.transformer_substation),
        _SemanticEntry(lat_deg=0.0135, lon_deg=0.0,       # north: ~1.5 km
                       landmark_type=LandmarkType.transformer_substation),
    ]

    state = _make_state(lat=0.0, lon=0.0, heading=0.0)
    kf = _make_mock_kf(state)
    # Camera bearing 0° (forward, which is north since heading=0).  Old code
    # would pick lm_east (nearest position); new code should pick lm_north.
    obs = _make_obs(LandmarkType.transformer_substation, bearing_deg=0.0, conf=0.9)
    updates = conv.process(obs, kf)

    assert updates == 1
    kf.update_vlm_bearing.assert_called_once()
    # Assert the update used the NORTH landmark's position, not the east one.
    _args, kwargs = kf.update_vlm_bearing.call_args
    assert abs(kwargs["landmark_lat_deg"] - 0.0135) < 1e-6, (
        "should have associated with the north landmark, not the east one"
    )


def test_known_landmark_assoc_gate_rejects_spurious_bearing():
    """Observation bearing with no matching candidate should be dropped."""
    from uav_nav.fusion.constraint_converter import _SemanticEntry

    conv = ConstraintConverter()
    conv._known[LandmarkType.transformer_substation] = [
        _SemanticEntry(lat_deg=0.0, lon_deg=0.009,        # east: bearing 90°
                       landmark_type=LandmarkType.transformer_substation),
    ]
    state = _make_state(lat=0.0, lon=0.0, heading=0.0)
    kf = _make_mock_kf(state)
    # Bearing 270° (due west) — 180° off from only candidate at 90° → reject.
    obs = _make_obs(LandmarkType.transformer_substation, bearing_deg=270.0, conf=0.9)
    updates = conv.process(obs, kf)
    assert updates == 0
    kf.update_vlm_bearing.assert_not_called()


def test_low_confidence_landmark_ignored():
    conv = ConstraintConverter()
    state = _make_state()
    kf = _make_mock_kf(state)
    obs = _make_obs(LandmarkType.highway, bearing_deg=45.0, conf=0.3)  # below 0.7 threshold
    updates = conv.process(obs, kf)
    assert updates == 0
    assert conv.pending_count == 0


def test_triangulate_90_degree_crossing():
    """Two bearing rays crossing at 90 degrees should triangulate accurately."""
    # Observer 1 at (0, 0) bearing due East (90°)
    # Observer 2 at (0.01, 0) bearing due South (180°)
    # Intersection should be near (0, 0.01) — approx 1 km east of obs1
    result = _triangulate(0.0, 0.0, 90.0, 0.01, 0.0, 180.0)
    assert result is not None
    lm_lat, lm_lon, pos_sigma_deg = result
    # Should be near (0.0, 0.01)
    assert abs(lm_lat) < 0.02
    assert abs(lm_lon - 0.01) < 0.02
    # 90° crossing is good geometry — pos_sigma should be small (<0.1°)
    assert pos_sigma_deg < 0.1


def test_triangulate_parallel_rays_returns_none():
    """Parallel rays should return None."""
    result = _triangulate(0.0, 0.0, 90.0, 0.01, 0.0, 90.0)
    assert result is None


def test_haversine_accuracy():
    """Haversine distance between two close points."""
    dist = _haversine_m(33.641, -84.427, 33.650, -84.427)
    # ~1 km north
    assert 900 < dist < 1100, f"Expected ~1000 m, got {dist:.0f} m"
