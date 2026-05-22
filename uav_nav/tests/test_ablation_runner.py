"""Tests for the sensor ablation runner and CSV row builder."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.run_ablations import (
    ABLATION_CONDITIONS,
    AblationCondition,
    _make_row,
    _CSV_COLUMNS,
)


# ---------------------------------------------------------------------------
# AblationCondition structure
# ---------------------------------------------------------------------------

class TestAblationConditions:
    def test_all_seven_conditions_exist(self):
        assert set(ABLATION_CONDITIONS.keys()) == {"A0", "A1", "A2", "A3", "A4", "A5", "A6"}

    def test_a0_imu_only(self):
        a = ABLATION_CONDITIONS["A0"]
        assert not a.use_barometer
        assert not a.use_magnetometer
        assert not a.use_airspeed
        assert not a.use_vlm
        assert not a.use_celestial_solar

    def test_a6_full_system(self):
        a = ABLATION_CONDITIONS["A6"]
        assert a.use_barometer
        assert a.use_magnetometer
        assert a.use_airspeed
        assert a.use_vlm
        assert a.use_celestial_solar

    def test_a4_vlm_no_celestial(self):
        a = ABLATION_CONDITIONS["A4"]
        assert a.use_vlm
        assert not a.use_celestial_solar

    def test_a5_celestial_no_vlm(self):
        a = ABLATION_CONDITIONS["A5"]
        assert not a.use_vlm
        assert a.use_celestial_solar

    def test_progressive_sensor_addition(self):
        # Each step adds exactly one sensor group over the previous
        prev_score = 0
        for name in ["A0", "A1", "A2", "A3"]:
            a = ABLATION_CONDITIONS[name]
            score = sum([a.use_barometer, a.use_magnetometer,
                         a.use_airspeed, a.use_vlm, a.use_celestial_solar])
            assert score >= prev_score
            prev_score = score

    def test_conditions_are_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            ABLATION_CONDITIONS["A0"].use_vlm = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _make_row: CSV row builder
# ---------------------------------------------------------------------------

_FAKE_RESULT = {
    "scenario_id": "very_short",
    "label": "Very Short",
    "final_error_m": 27.39,
    "max_error_m": 53.17,
    "mean_error_m": 20.97,
    "median_error_m": 23.16,
    "p90_error_m": 33.28,
    "pass_10m": False,
    "pass_25m": False,
    "pass_50m": True,
    "pass_100m": True,
    "passed": True,
    "vlm_obs_total": 134,
    "vlm_passed_confidence": 134,
    "vlm_rejected_confidence": 0,
    "vlm_rejected_bearing": 0,
    "vlm_rejected_ambiguity": 18,
    "vlm_ukf_updates": 116,
    "celestial_updates": 2,
    "solar_updates": 0,
    "nees_alarm_count": 0,
    "n_landmarks": 3,
    "dist_km": 30.22,
    "duration_s": 604.4,
}


class TestMakeRow:
    def test_row_has_all_columns(self):
        row = _make_row(_FAKE_RESULT, "nighttime", 1, ABLATION_CONDITIONS["A6"], "abc12345")
        for col in _CSV_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_nighttime_sets_celestial_not_solar(self):
        row = _make_row(_FAKE_RESULT, "nighttime", 1, ABLATION_CONDITIONS["A6"], "x")
        assert row["celestial_enabled"] is True
        assert row["solar_enabled"] is False

    def test_daytime_sets_solar_not_celestial(self):
        row = _make_row(_FAKE_RESULT, "daytime", 1, ABLATION_CONDITIONS["A6"], "x")
        assert row["solar_enabled"] is True
        assert row["celestial_enabled"] is False

    def test_a0_all_sensor_flags_false(self):
        row = _make_row(_FAKE_RESULT, "nighttime", 1, ABLATION_CONDITIONS["A0"], "x")
        assert row["barometer_enabled"] is False
        assert row["magnetometer_enabled"] is False
        assert row["airspeed_enabled"] is False
        assert row["vlm_enabled"] is False
        assert row["celestial_enabled"] is False
        assert row["solar_enabled"] is False
        assert row["imu_enabled"] is True  # IMU always on

    def test_pass_50m_propagated(self):
        row = _make_row(_FAKE_RESULT, "nighttime", 1, ABLATION_CONDITIONS["A6"], "x")
        assert row["pass_50m"] is True

    def test_pass_50m_from_passed_alias(self):
        result = dict(_FAKE_RESULT)
        del result["pass_50m"]
        result["passed"] = True
        row = _make_row(result, "nighttime", 1, ABLATION_CONDITIONS["A6"], "x")
        assert row["pass_50m"] is True

    def test_vlm_counts_propagated(self):
        row = _make_row(_FAKE_RESULT, "nighttime", 1, ABLATION_CONDITIONS["A6"], "x")
        assert row["vlm_obs_total"] == 134
        assert row["vlm_ukf_updates"] == 116
        assert row["vlm_rejected_ambiguity"] == 18

    def test_nis_alarm_count_from_nees_alias(self):
        row = _make_row(_FAKE_RESULT, "nighttime", 1, ABLATION_CONDITIONS["A6"], "x")
        assert row["nis_alarm_count"] == 0

    def test_notes_field(self):
        row = _make_row(_FAKE_RESULT, "nighttime", 1, ABLATION_CONDITIONS["A0"], "x", notes="CRASH: test")
        assert row["notes"] == "CRASH: test"
