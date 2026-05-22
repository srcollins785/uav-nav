"""Unit tests for VLM response parser."""
import pytest
from uav_nav.vlm.parser import parse_vlm_response
from uav_nav.vlm.schemas import SceneType, LandmarkType


VALID_JSON = """{
  "scene_type": "industrial",
  "sky_visible": true,
  "landmarks": [
    {
      "type": "transformer_substation",
      "bearing_deg": 45.0,
      "distance_category": "mid",
      "confidence": 0.85,
      "notes": "Large substation complex visible to the northeast"
    }
  ],
  "stars_visible": ["Polaris", "Betelgeuse"],
  "horizon_features": "Rolling hills to the south"
}"""

MARKDOWN_WRAPPED = """Sure! Here is the navigation JSON:
```json
{
  "scene_type": "rural",
  "sky_visible": false,
  "landmarks": [],
  "stars_visible": [],
  "horizon_features": null
}
```
Hope this helps!"""

PARTIAL_JSON = '{"scene_type": "urban", "sky_visible": true, "landmarks": []}'

EMPTY_RESPONSE = ""
GARBAGE_RESPONSE = "I see a beautiful sunset with lots of trees and a river."


def test_valid_json_parses():
    obs = parse_vlm_response(VALID_JSON)
    assert obs is not None
    assert obs.scene_type == SceneType.industrial
    assert obs.sky_visible is True
    assert len(obs.landmarks) == 1
    assert obs.landmarks[0].type == LandmarkType.transformer_substation
    assert abs(obs.landmarks[0].bearing_deg - 45.0) < 0.01
    assert "Polaris" in obs.stars_visible


def test_markdown_wrapped_json_parses():
    obs = parse_vlm_response(MARKDOWN_WRAPPED)
    assert obs is not None
    assert obs.scene_type == SceneType.rural
    assert obs.sky_visible is False
    assert obs.landmarks == []


def test_partial_json_parses():
    obs = parse_vlm_response(PARTIAL_JSON)
    assert obs is not None
    assert obs.scene_type == SceneType.urban


def test_empty_response_returns_none():
    assert parse_vlm_response(EMPTY_RESPONSE) is None
    assert parse_vlm_response("   ") is None


def test_garbage_returns_none():
    result = parse_vlm_response(GARBAGE_RESPONSE)
    # May be None (preferred) or a default SemanticObservation — must not crash
    # The important thing is no exception is raised


def test_bearing_wrapped_360():
    json_with_high_bearing = """{
      "scene_type": "unknown",
      "sky_visible": false,
      "landmarks": [{"type": "highway", "bearing_deg": 370.0, "distance_category": "far", "confidence": 0.6}],
      "stars_visible": [],
      "horizon_features": null
    }"""
    obs = parse_vlm_response(json_with_high_bearing)
    assert obs is not None
    assert obs.landmarks[0].bearing_deg == pytest.approx(10.0, abs=0.01)


def test_high_confidence_filter():
    obs = parse_vlm_response(VALID_JSON)
    assert obs is not None
    high = obs.high_confidence_landmarks(min_conf=0.9)
    assert len(high) == 0
    mid = obs.high_confidence_landmarks(min_conf=0.7)
    assert len(mid) == 1
