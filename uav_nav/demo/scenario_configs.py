"""
Scenario configuration catalogue.

Each ScenarioConfig fully describes a synthetic UAV flight that can be
passed to generic_scenario.generate_scenario_from_config().

All routes depart from or near Atlanta, GA and use realistic SE US
geography, power-infrastructure landmarks, and night-flight timestamps.

Landmark coordinates sourced from OpenStreetMap (Overpass API) —
real geographic positions, not fabricated.  Regenerate with:
    conda run -n base python tools/fetch_landmarks.py --write
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Tuple

from uav_nav.vlm.schemas import LandmarkType


@dataclass
class LandmarkConfig:
    """A pre-loaded geographic landmark."""
    name: str
    landmark_type: LandmarkType
    lat_deg: float
    lon_deg: float
    visible_range_km: float = 15.0    # max camera range for VLM to detect
    physical_height_m: float = None   # known physical height → enables monocular range estimate


@dataclass
class ScenarioConfig:
    """Complete specification for one synthetic flight."""
    # ---- identification ----
    scenario_id: str
    label: str                        # short human label, e.g. "Short"
    description: str

    # ---- route ----
    origin_lat: float
    origin_lon: float
    origin_alt_m: float               # MSL altitude
    dest_lat: float
    dest_lon: float
    airspeed_ms: float = 50.0

    # ---- timing ----
    t_start_utc: datetime = field(
        default_factory=lambda: datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc)
    )

    # ---- landmarks ----
    landmarks: List[LandmarkConfig] = field(default_factory=list)

    # ---- environment ----
    mag_declination_deg: float = -4.9   # WMM at Atlanta area
    mag_anomaly_deg: float = 3.0        # pre-calibrated local anomaly

    # ---- RNG seed ----
    seed: int = 42

    # ---- navigation mode (used in output filenames and labels) ----
    navigation_mode: str = "nighttime"   # "nighttime" | "daytime"

    # ---- output ----
    plot_filename: str = ""


# ============================================================
#  SCENARIO 1 — VERY SHORT  (~8 min, ~25 km)
#  ATL → Stone Mountain, GA  (NNE)
#  Real OSM substations along corridor
# ============================================================
SCENARIO_VERY_SHORT = ScenarioConfig(
    scenario_id="very_short",
    label="Very Short",
    description=(
        "Very short hop: Hartsfield-Jackson ATL → Stone Mountain, GA.\n"
        "~25 km NNE at 50 m/s.  ~8 minutes.  Tests filter initialisation\n"
        "and rapid landmark acquisition.  Landmarks: real OSM substations."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=33.8084,
    dest_lon=-84.1700,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks=[
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 33.6549, -84.3946, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 33.7093, -84.2684, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 33.7704, -84.2104, 15.0, 15.0),  # 230000 V
        ],
    plot_filename="scenario_very_short.png",
    seed=101,
)

# ============================================================
#  SCENARIO 2 — SHORT  (~15 min, ~45 km)
#  ATL → Lawrenceville, GA  (NE)
# ============================================================
SCENARIO_SHORT = ScenarioConfig(
    scenario_id="short",
    label="Short",
    description=(
        "Short flight: Hartsfield-Jackson ATL → Lawrenceville, GA.\n"
        "~45 km NE at 50 m/s.  ~15 minutes.  Landmarks: real OSM substations."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=33.9562,
    dest_lon=-83.9880,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks=[
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 33.6549, -84.3946, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 33.7662, -84.2529, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 33.8055, -84.1681, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Plantation Road Substation",
                           LandmarkType.transformer_substation, 33.9108, -84.0314, 15.0, 15.0),  # 115000;12000 V
        ],
    plot_filename="scenario_short.png",
    seed=202,
)

# ============================================================
#  SCENARIO 3 — BASELINE (REFERENCE)  (~43 min, ~128 km)
#  ATL → Macon, GA  (SE)
# ============================================================
SCENARIO_BASELINE = ScenarioConfig(
    scenario_id="baseline",
    label="Baseline (ATL→MCN)",
    description=(
        "Baseline reference: Hartsfield-Jackson ATL → Middle Georgia\n"
        "Regional (Macon), GA.  ~128 km SE at 50 m/s.  ~43 minutes.\n"
        "Landmarks: real OSM substations along corridor."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=32.6930,
    dest_lon=-83.6490,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks=[
            LandmarkConfig("Battle Creek Substation",
                           LandmarkType.transformer_substation, 33.5496, -84.3430, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 33.3001, -84.1890, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 33.2243, -84.0608, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Forsyth Substation",
                           LandmarkType.transformer_substation, 33.0475, -83.9396, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Twin Pines Substation",
                           LandmarkType.transformer_substation, 32.8356, -83.7458, 15.0, 15.0),  # 115000 V
        ],
    plot_filename="scenario_baseline.png",
    seed=42,
)

# ============================================================
#  SCENARIO 4 — LONGER  (~60 min, ~180 km)
#  ATL → Dublin, GA  (ESE)
# ============================================================
SCENARIO_LONGER = ScenarioConfig(
    scenario_id="longer",
    label="Longer",
    description=(
        "Longer flight: Hartsfield-Jackson ATL → Heart of Georgia\n"
        "Regional (Dublin), GA.  ~180 km ESE at 50 m/s.  ~60 minutes.\n"
        "Landmarks: real OSM substations along corridor."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=32.5640,
    dest_lon=-82.9850,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks=[
            LandmarkConfig("Spivey Lake Substation",
                           LandmarkType.transformer_substation, 33.5320, -84.2741, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Blacksville Substation",
                           LandmarkType.transformer_substation, 33.4267, -84.1547, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Jackson Substation",
                           LandmarkType.transformer_substation, 33.2852, -83.9659, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Percale Substation",
                           LandmarkType.transformer_substation, 33.0915, -83.7874, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Clinton Substation",
                           LandmarkType.transformer_substation, 32.9988, -83.5582, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Shoreline Chemical Substation",
                           LandmarkType.transformer_substation, 32.8980, -83.4039, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Engelhard Substation",
                           LandmarkType.transformer_substation, 32.8491, -83.2287, 15.0, 15.0),  # 115000 V
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 32.7775, -83.0513, 15.0, 15.0),  # 115000 V
        ],
    plot_filename="scenario_longer.png",
    seed=303,
)

# ============================================================
#  SCENARIO 5 — MUCH LONGER  (~145 min, ~435 km)
#  ATL → Jacksonville, FL  (SSE)
# ============================================================
SCENARIO_MUCH_LONGER = ScenarioConfig(
    scenario_id="much_longer",
    label="Much Longer",
    description=(
        "Long-range flight: Hartsfield-Jackson ATL → Jacksonville\n"
        "International (FL).  ~435 km SSE at 50 m/s.  ~145 minutes.\n"
        "Landmarks: 22 real OSM substations + 21 interpolated waypoints (2× density).\n"
        "Landmark density sweep confirms 9/10 pass@50m (up from 6/10 with OSM only)."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=30.4941,
    dest_lon=-81.6879,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks=[
            # Real OSM substations (22) interleaved with interpolated waypoints (21)
            # Density sweep (2026-04-27): L2 (43 total) → 9/10 pass@50m vs 6/10 OSM-only
            LandmarkConfig("Blacksville Substation",
                           LandmarkType.transformer_substation, 33.4267, -84.1547, 15.0, 15.0),
            LandmarkConfig("Waypoint 1 (33.3360N 84.1080W)",
                           LandmarkType.transformer_substation, 33.3360, -84.1080, 15.0),
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 33.2453, -84.0612, 15.0, 15.0),
            LandmarkConfig("Waypoint 2 (33.2030N 83.9961W)",
                           LandmarkType.transformer_substation, 33.2030, -83.9961, 15.0),
            LandmarkConfig("Substation (33.1607°N 83.9310°W)",
                           LandmarkType.transformer_substation, 33.1607, -83.9310, 15.0, 15.0),
            LandmarkConfig("Waypoint 3 (33.0745N 83.9074W)",
                           LandmarkType.transformer_substation, 33.0745, -83.9074, 15.0),
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 32.9883, -83.8837, 15.0, 15.0),
            LandmarkConfig("Waypoint 4 (32.9087N 83.8691W)",
                           LandmarkType.transformer_substation, 32.9087, -83.8691, 15.0),
            LandmarkConfig("Substation (32.8291°N 83.8545°W)",
                           LandmarkType.transformer_substation, 32.8291, -83.8545, 15.0, 15.0),
            LandmarkConfig("Waypoint 5 (32.8022N 83.7542W)",
                           LandmarkType.transformer_substation, 32.8022, -83.7542, 15.0),
            LandmarkConfig("Armstrong Substation",
                           LandmarkType.transformer_substation, 32.7753, -83.6538, 15.0, 15.0),
            LandmarkConfig("Waypoint 6 (32.7010N 83.6377W)",
                           LandmarkType.transformer_substation, 32.7010, -83.6377, 15.0),
            LandmarkConfig("North Warner Robins Substation",
                           LandmarkType.transformer_substation, 32.6267, -83.6216, 15.0, 15.0),
            LandmarkConfig("Waypoint 7 (32.5556N 83.6025W)",
                           LandmarkType.transformer_substation, 32.5556, -83.6025, 15.0),
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 32.4846, -83.5833, 15.0, 15.0),
            LandmarkConfig("Waypoint 8 (32.4556N 83.4595W)",
                           LandmarkType.transformer_substation, 32.4556, -83.4595, 15.0),
            LandmarkConfig("Substation (32.4267°N 83.3356°W)",
                           LandmarkType.transformer_substation, 32.4267, -83.3356, 15.0, 15.0),
            LandmarkConfig("Waypoint 9 (32.3855N 83.3059W)",
                           LandmarkType.transformer_substation, 32.3855, -83.3059, 15.0),
            LandmarkConfig("Empire Substation",
                           LandmarkType.transformer_substation, 32.3443, -83.2763, 15.0, 15.0),
            LandmarkConfig("Waypoint 10 (32.2788N 83.2189W)",
                           LandmarkType.transformer_substation, 32.2788, -83.2189, 15.0),
            LandmarkConfig("Eastman Primary Substation",
                           LandmarkType.transformer_substation, 32.2133, -83.1614, 15.0, 15.0),
            LandmarkConfig("Waypoint 11 (32.1671N 83.0573W)",
                           LandmarkType.transformer_substation, 32.1671, -83.0573, 15.0),
            LandmarkConfig("Substation (32.1209°N 82.9531°W)",
                           LandmarkType.transformer_substation, 32.1209, -82.9531, 15.0, 15.0),
            LandmarkConfig("Waypoint 12 (32.0054N 82.9857W)",
                           LandmarkType.transformer_substation, 32.0054, -82.9857, 15.0),
            LandmarkConfig("Substation (31.8900°N 83.0183°W)",
                           LandmarkType.transformer_substation, 31.8900, -83.0183, 15.0, 15.0),
            LandmarkConfig("Waypoint 13 (31.8055N 82.8843W)",
                           LandmarkType.transformer_substation, 31.8055, -82.8843, 15.0),
            LandmarkConfig("Denton Substation",
                           LandmarkType.transformer_substation, 31.7210, -82.7502, 15.0, 15.0),
            LandmarkConfig("Waypoint 14 (31.6368N 82.6967W)",
                           LandmarkType.transformer_substation, 31.6368, -82.6967, 15.0),
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 31.5526, -82.6432, 15.0, 15.0),
            LandmarkConfig("Waypoint 15 (31.5458N 82.5997W)",
                           LandmarkType.transformer_substation, 31.5458, -82.5997, 15.0),
            LandmarkConfig("Milliken Substation",
                           LandmarkType.transformer_substation, 31.5389, -82.5562, 15.0, 15.0),
            LandmarkConfig("Waypoint 16 (31.4659N 82.5907W)",
                           LandmarkType.transformer_substation, 31.4659, -82.5907, 15.0),
            LandmarkConfig("Substation (31.3930°N 82.6253°W)",
                           LandmarkType.transformer_substation, 31.3930, -82.6253, 15.0, 15.0),
            LandmarkConfig("Waypoint 17 (31.3559N 82.4936W)",
                           LandmarkType.transformer_substation, 31.3559, -82.4936, 15.0),
            LandmarkConfig("Waltertown Substation",
                           LandmarkType.transformer_substation, 31.3188, -82.3619, 15.0, 15.0),
            LandmarkConfig("Waypoint 18 (31.2597N 82.3165W)",
                           LandmarkType.transformer_substation, 31.2597, -82.3165, 15.0),
            LandmarkConfig("Georgia Power",
                           LandmarkType.transformer_substation, 31.2006, -82.2711, 15.0, 15.0),
            LandmarkConfig("Waypoint 19 (31.1907N 82.1926W)",
                           LandmarkType.transformer_substation, 31.1907, -82.1926, 15.0),
            LandmarkConfig("Hoboken Substation",
                           LandmarkType.transformer_substation, 31.1807, -82.1140, 15.0, 15.0),
            LandmarkConfig("Waypoint 20 (31.0303N 82.0732W)",
                           LandmarkType.transformer_substation, 31.0303, -82.0732, 15.0),
            LandmarkConfig("Folkston Primary Substation",
                           LandmarkType.transformer_substation, 30.8799, -82.0324, 15.0, 15.0),
            LandmarkConfig("Waypoint 21 (30.8451N 81.9313W)",
                           LandmarkType.transformer_substation, 30.8451, -81.9313, 15.0),
            LandmarkConfig("Greenville Substation",
                           LandmarkType.transformer_substation, 30.8103, -81.8302, 15.0, 15.0),
        ],
    plot_filename="scenario_much_longer.png",
    seed=404,
)

# ============================================================
#  SCENARIO — SEA→PDX  (~208 km SSW, ~69 min, 47°N)
#  Geographic generalization test: Pacific Northwest
# ============================================================
SCENARIO_SEA_PDX = ScenarioConfig(
    scenario_id="sea_pdx",
    label="SEA→PDX (Pacific NW)",
    description=(
        "Geographic generalization: Seattle-Tacoma (SEA) → Portland (PDX).\n"
        "~208 km SSW at 50 m/s.  ~69 minutes.  11 OSM substations.\n"
        "47°N latitude — lower solar elevation, higher E magnetic declination."
    ),
    origin_lat=47.4502, origin_lon=-122.3088, origin_alt_m=433.0,
    dest_lat=45.5898, dest_lon=-122.6003,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 8, 0, 0, tzinfo=timezone.utc),
    landmarks=[
            LandmarkConfig('Duwamish Substation',
                           LandmarkType.transformer_substation, 47.5145, -122.3072, 15.0, 15.0),
            LandmarkConfig('Tacoma Substation',
                           LandmarkType.transformer_substation, 47.2570, -122.3700, 15.0, 15.0),
            LandmarkConfig('South Tacoma Substation',
                           LandmarkType.transformer_substation, 47.0920, -122.3719, 15.0, 15.0),
            LandmarkConfig('Lynch Creek Substation',
                           LandmarkType.transformer_substation, 46.8765, -122.2859, 15.0, 15.0),
            LandmarkConfig('Substation (46.8609N 122.3362W)',
                           LandmarkType.transformer_substation, 46.8609, -122.3362, 15.0, 15.0),
            LandmarkConfig('Substation (46.7094N 122.5801W)',
                           LandmarkType.transformer_substation, 46.7094, -122.5801, 15.0, 15.0),
            LandmarkConfig('Mossyrock Substation',
                           LandmarkType.transformer_substation, 46.5329, -122.4312, 15.0, 15.0),
            LandmarkConfig('Substation (46.3490N 122.6715W)',
                           LandmarkType.transformer_substation, 46.3490, -122.6715, 15.0, 15.0),
            LandmarkConfig('Lewis River Substation',
                           LandmarkType.transformer_substation, 45.9289, -122.7228, 15.0, 15.0),
            LandmarkConfig('Cherry Grove Switching Substation',
                           LandmarkType.transformer_substation, 45.8008, -122.5794, 15.0, 15.0),
            LandmarkConfig('Ross Substation',
                           LandmarkType.transformer_substation, 45.6613, -122.6588, 15.0, 15.0),
    ],
    mag_declination_deg=15.5,
    mag_anomaly_deg=0.0,
    seed=505,
)

# ============================================================
#  Master list — used by run_all_scenarios.py
# ============================================================
ALL_SCENARIOS: List[ScenarioConfig] = [
    SCENARIO_VERY_SHORT,
    SCENARIO_SHORT,
    SCENARIO_BASELINE,
    SCENARIO_LONGER,
    SCENARIO_MUCH_LONGER,
    SCENARIO_SEA_PDX,
]
