"""
scenario_loader.py — Load ScenarioConfig from YAML files.

Provides:
    load_scenario(path)           Load a single YAML file → ScenarioConfig
    load_scenario_by_id(sid)      Search scenarios/ directory by scenario_id
    list_scenarios()              Return list of available scenario_ids
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import yaml

# Directory containing the bundled YAML files
_SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Mapping from YAML type strings → LandmarkType enum values
_LANDMARK_TYPE_MAP: dict | None = None


def _get_landmark_type_map() -> dict:
    global _LANDMARK_TYPE_MAP
    if _LANDMARK_TYPE_MAP is None:
        from uav_nav.vlm.schemas import LandmarkType
        _LANDMARK_TYPE_MAP = {e.value: e for e in LandmarkType}
        # also accept the enum name itself (e.g. "transformer_substation")
        _LANDMARK_TYPE_MAP.update({e.name: e for e in LandmarkType})
    return _LANDMARK_TYPE_MAP


def load_scenario(path: str | Path):
    """
    Parse a YAML scenario file and return a ScenarioConfig.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to the YAML file.

    Returns
    -------
    ScenarioConfig
    """
    from uav_nav.demo.scenario_configs import LandmarkConfig, ScenarioConfig

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path, "r") as fh:
        data = yaml.safe_load(fh)

    # ---- route ----
    route = data.get("route", {})
    timing = data.get("timing", {})
    env = data.get("environment", {})

    # ---- timestamp ----
    t_str = timing.get("t_start_utc", "2026-01-28T02:00:00Z")
    # Support both "Z" suffix and "+00:00"
    t_str = t_str.replace("Z", "+00:00")
    t_start = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)

    # ---- landmarks ----
    lm_type_map = _get_landmark_type_map()
    landmarks: List[LandmarkConfig] = []
    for lm in data.get("landmarks", []):
        type_key = str(lm["type"])
        if type_key not in lm_type_map:
            raise ValueError(
                f"Unknown landmark type '{type_key}' in {path}. "
                f"Valid types: {sorted(lm_type_map.keys())}"
            )
        landmarks.append(LandmarkConfig(
            name=lm["name"],
            landmark_type=lm_type_map[type_key],
            lat_deg=float(lm["lat_deg"]),
            lon_deg=float(lm["lon_deg"]),
            visible_range_km=float(lm.get("visible_range_km", 15.0)),
        ))

    return ScenarioConfig(
        scenario_id=str(data["scenario_id"]),
        label=str(data.get("label", data["scenario_id"])),
        description=str(data.get("description", "")),
        origin_lat=float(route["origin_lat"]),
        origin_lon=float(route["origin_lon"]),
        origin_alt_m=float(route.get("origin_alt_m", 500.0)),
        dest_lat=float(route["dest_lat"]),
        dest_lon=float(route["dest_lon"]),
        airspeed_ms=float(route.get("airspeed_ms", 50.0)),
        t_start_utc=t_start,
        landmarks=landmarks,
        mag_declination_deg=float(env.get("mag_declination_deg", -4.9)),
        mag_anomaly_deg=float(env.get("mag_anomaly_deg", 3.0)),
        seed=int(data.get("seed", 42)),
        plot_filename=str(data.get("plot_filename", "")),
    )


def load_scenario_by_id(scenario_id: str):
    """
    Load a scenario by its scenario_id from the bundled scenarios/ directory.

    Parameters
    ----------
    scenario_id : str
        Must match the ``scenario_id`` field in one of the YAML files.

    Returns
    -------
    ScenarioConfig

    Raises
    ------
    KeyError if no matching scenario is found.
    """
    for yaml_path in _SCENARIOS_DIR.glob("*.yaml"):
        if yaml_path.stem == "template":
            continue
        try:
            cfg = load_scenario(yaml_path)
            if cfg.scenario_id == scenario_id:
                return cfg
        except Exception:
            continue
    raise KeyError(
        f"No scenario with id '{scenario_id}' found in {_SCENARIOS_DIR}. "
        f"Available: {list_scenarios()}"
    )


def list_scenarios() -> List[str]:
    """
    Return a sorted list of scenario_ids available in the scenarios/ directory.
    """
    ids = []
    for yaml_path in sorted(_SCENARIOS_DIR.glob("*.yaml")):
        if yaml_path.stem == "template":
            continue
        try:
            with open(yaml_path) as fh:
                data = yaml.safe_load(fh)
            ids.append(str(data["scenario_id"]))
        except Exception:
            pass
    return ids


def scenario_to_yaml(cfg, extra: dict | None = None) -> str:
    """
    Serialise a ScenarioConfig back to a YAML string.
    Used by run_adhoc.py --save-config to persist ad hoc runs.
    """
    doc: dict = {
        "scenario_id": cfg.scenario_id,
        "label": cfg.label,
        "description": cfg.description,
        "route": {
            "origin_lat": cfg.origin_lat,
            "origin_lon": cfg.origin_lon,
            "origin_alt_m": cfg.origin_alt_m,
            "dest_lat": cfg.dest_lat,
            "dest_lon": cfg.dest_lon,
            "airspeed_ms": cfg.airspeed_ms,
        },
        "timing": {
            "t_start_utc": cfg.t_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "environment": {
            "mag_declination_deg": cfg.mag_declination_deg,
            "mag_anomaly_deg": cfg.mag_anomaly_deg,
        },
        "landmarks": [
            {
                "name": lm.name,
                "type": lm.landmark_type.name,
                "lat_deg": lm.lat_deg,
                "lon_deg": lm.lon_deg,
                "visible_range_km": lm.visible_range_km,
            }
            for lm in cfg.landmarks
        ],
        "seed": cfg.seed,
    }
    if extra:
        doc.update(extra)
    return yaml.dump(doc, default_flow_style=False, allow_unicode=True, sort_keys=False)
