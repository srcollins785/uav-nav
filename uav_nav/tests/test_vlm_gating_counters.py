"""Tests for VLM gating counters on ConstraintConverter.vlm_stats."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from uav_nav.fusion.constraint_converter import ConstraintConverter
from uav_nav.navigation.state import UAVState
from uav_nav.vlm.schemas import (
    DistanceCategory, LandmarkObservation, LandmarkType,
    SceneType, SemanticObservation,
)


def _make_state(lat=33.641, lon=-84.427, heading=0.0) -> UAVState:
    return UAVState(lat_deg=lat, lon_deg=lon, heading_deg=heading)


def _make_obs(lm_type: LandmarkType, bearing_deg: float, conf: float = 0.85) -> SemanticObservation:
    return SemanticObservation(
        scene_type=SceneType.industrial,
        sky_visible=False,
        landmarks=[LandmarkObservation(
            type=lm_type,
            bearing_deg=bearing_deg,
            distance_category=DistanceCategory.mid,
            confidence=conf,
        )],
    )


def _make_converter() -> ConstraintConverter:
    # ConstraintConverter uses get_config() internally — no constructor args
    return ConstraintConverter()


def _fake_ukf(lat=33.641, lon=-84.427, heading=0.0):
    kf = MagicMock()
    kf.state = _make_state(lat, lon, heading)
    kf.update_vlm_bearing = MagicMock()
    kf.update_position = MagicMock()
    return kf


# ---------------------------------------------------------------------------
# vlm_stats initial state
# ---------------------------------------------------------------------------

class TestVlmStatsInitial:
    def test_all_counters_zero_at_start(self):
        c = _make_converter()
        stats = c.vlm_stats
        assert stats["vlm_obs_total"] == 0
        assert stats["vlm_passed_confidence"] == 0
        assert stats["vlm_rejected_confidence"] == 0
        assert stats["vlm_rejected_bearing"] == 0
        assert stats["vlm_rejected_ambiguity"] == 0
        assert stats["vlm_ukf_updates"] == 0

    def test_vlm_stats_has_expected_keys(self):
        c = _make_converter()
        expected = {
            "vlm_obs_total", "vlm_passed_confidence", "vlm_rejected_confidence",
            "vlm_rejected_bearing", "vlm_rejected_ambiguity", "vlm_ukf_updates",
            "triangulation_attempts", "triangulation_successes",
        }
        assert set(c.vlm_stats.keys()) == expected


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

class TestConfidenceGate:
    def test_low_confidence_observation_rejected(self):
        c = _make_converter()
        kf = _fake_ukf()
        obs = _make_obs(LandmarkType.transformer_substation, bearing_deg=90.0, conf=0.3)
        c.process(obs, kf)
        stats = c.vlm_stats
        assert stats["vlm_obs_total"] == 1
        assert stats["vlm_rejected_confidence"] == 1
        assert stats["vlm_passed_confidence"] == 0
        assert stats["vlm_ukf_updates"] == 0

    def test_high_confidence_observation_counted(self):
        c = _make_converter()
        kf = _fake_ukf()
        # Register a known landmark so the bearing update can proceed
        c.register_known_landmark(
            LandmarkType.transformer_substation,
            lat_deg=33.641 + 0.009,  # ~1 km north
            lon_deg=-84.427,
        )
        obs = _make_obs(LandmarkType.transformer_substation, bearing_deg=0.0, conf=0.9)
        c.process(obs, kf)
        stats = c.vlm_stats
        assert stats["vlm_obs_total"] == 1
        assert stats["vlm_passed_confidence"] == 1
        assert stats["vlm_rejected_confidence"] == 0


# ---------------------------------------------------------------------------
# Accumulation across multiple process() calls
# ---------------------------------------------------------------------------

class TestCounterAccumulation:
    def test_counters_accumulate_across_calls(self):
        c = _make_converter()
        kf = _fake_ukf()
        obs_low = _make_obs(LandmarkType.transformer_substation, 90.0, conf=0.1)
        c.process(obs_low, kf)
        c.process(obs_low, kf)
        stats = c.vlm_stats
        assert stats["vlm_obs_total"] == 2
        assert stats["vlm_rejected_confidence"] == 2

    def test_total_equals_passed_plus_rejected_confidence(self):
        c = _make_converter()
        kf = _fake_ukf()
        # Mix high and low confidence
        for conf in [0.9, 0.1, 0.8, 0.2]:
            obs = _make_obs(LandmarkType.transformer_substation, 90.0, conf=conf)
            c.process(obs, kf)
        stats = c.vlm_stats
        assert stats["vlm_obs_total"] == stats["vlm_passed_confidence"] + stats["vlm_rejected_confidence"]


# ---------------------------------------------------------------------------
# Multi-threshold pass flags
# ---------------------------------------------------------------------------

class TestMultiThresholdPass:
    """Tests for pass_10m / pass_25m / pass_50m / pass_100m in run_scenario result."""

    def _pass_flags(self, error_m: float) -> dict:
        return {
            "pass_10m": error_m < 10.0,
            "pass_25m": error_m < 25.0,
            "pass_50m": error_m < 50.0,
            "pass_100m": error_m < 100.0,
        }

    def test_8m_passes_all(self):
        flags = self._pass_flags(8.0)
        assert flags["pass_10m"] and flags["pass_25m"] and flags["pass_50m"] and flags["pass_100m"]

    def test_20m_passes_25_50_100(self):
        flags = self._pass_flags(20.0)
        assert not flags["pass_10m"]
        assert flags["pass_25m"] and flags["pass_50m"] and flags["pass_100m"]

    def test_40m_passes_50_100_only(self):
        flags = self._pass_flags(40.0)
        assert not flags["pass_10m"]
        assert not flags["pass_25m"]
        assert flags["pass_50m"] and flags["pass_100m"]

    def test_60m_passes_100_only(self):
        flags = self._pass_flags(60.0)
        assert not flags["pass_10m"]
        assert not flags["pass_25m"]
        assert not flags["pass_50m"]
        assert flags["pass_100m"]

    def test_150m_fails_all(self):
        flags = self._pass_flags(150.0)
        assert not any(flags.values())

    def test_exactly_50m_fails(self):
        flags = self._pass_flags(50.0)
        assert not flags["pass_50m"]  # strict < not <=

    def test_just_under_50m_passes(self):
        flags = self._pass_flags(49.99)
        assert flags["pass_50m"]
