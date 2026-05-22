#!/usr/bin/env python3
"""
tools/fetch_landmarks.py

Fetch real power-substation positions from OpenStreetMap (Overpass API)
for each UAV scenario corridor, then regenerate scenario_configs.py with
real geographic coordinates.

Usage:
    # Preview output without writing any files
    conda run -n base python tools/fetch_landmarks.py --dry-run

    # Fetch and overwrite uav_nav/demo/scenario_configs.py
    conda run -n base python tools/fetch_landmarks.py --write

    # Cache Overpass responses (avoid re-querying on reruns)
    conda run -n base python tools/fetch_landmarks.py --write --cache osm_cache.json

Requirements: Python stdlib only (urllib, json, math).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Overpass query settings
# ---------------------------------------------------------------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
REQUEST_TIMEOUT_S = 30
RETRY_DELAY_S = 5
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Corridor / selection parameters
# ---------------------------------------------------------------------------
CORRIDOR_WIDTH_KM = 20.0   # max cross-track distance to route
MIN_SPACING_KM = 12.0      # minimum spacing between selected landmarks
MAX_LANDMARKS = 12         # hard cap per route
VISIBLE_RANGE_KM = 15.0    # UAV camera detection range
PHYSICAL_HEIGHT_M = 15.0   # nominal transformer height for monocular ranging
ORIGIN_DEST_EXCLUSION = 0.05  # exclude landmarks in first/last 5% of route


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class SubstationNode:
    name: str
    lat: float
    lon: float
    osm_id: int
    voltage: Optional[str] = None
    substation_type: Optional[str] = None
    frac: float = 0.0          # along-track fraction (0=origin, 1=dest)
    cross_track_km: float = 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlon = math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    return math.atan2(
        math.sin(dlon) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon),
    )


def cross_track_and_frac(
    p_lat: float, p_lon: float,
    a_lat: float, a_lon: float,
    b_lat: float, b_lon: float,
) -> Tuple[float, float]:
    """
    Return (cross_track_km, along_track_fraction) for point P relative to path A→B.

    cross_track_km       : perpendicular distance from P to the great-circle (always ≥ 0)
    along_track_fraction : 0.0 = at A, 1.0 = at B.  Negative means P is behind A;
                           > 1 means P is beyond B.  Caller filters to [0.05, 0.95].
    """
    R = 6371.0
    d_ab = haversine_km(a_lat, a_lon, b_lat, b_lon)
    if d_ab < 1e-6:
        return haversine_km(p_lat, p_lon, a_lat, a_lon), 0.0

    d13 = haversine_km(a_lat, a_lon, p_lat, p_lon) / R   # angular distance A→P (rad)
    theta13 = _bearing_rad(a_lat, a_lon, p_lat, p_lon)
    theta12 = _bearing_rad(a_lat, a_lon, b_lat, b_lon)

    dtheta = theta13 - theta12

    xt_rad = math.asin(math.sin(d13) * math.sin(dtheta))
    xt_km = abs(xt_rad * R)

    # acos always returns ≥ 0, so we recover the sign from cos(dtheta):
    # if P is behind A the bearing difference is > 90° → cos(dtheta) < 0.
    cos_xt = math.cos(xt_rad)
    if abs(cos_xt) < 1e-12:
        return xt_km, 0.0
    at_km = math.acos(max(-1.0, min(1.0, math.cos(d13) / cos_xt))) * R
    if math.cos(dtheta) < 0:
        at_km = -at_km                  # P is behind the origin

    return xt_km, at_km / d_ab


def corridor_bbox(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    buffer_km: float = 25.0,
) -> Tuple[float, float, float, float]:
    buf_deg = buffer_km / 111.0
    min_lat = min(origin[0], dest[0]) - buf_deg
    max_lat = max(origin[0], dest[0]) + buf_deg
    min_lon = min(origin[1], dest[1]) - buf_deg
    max_lon = max(origin[1], dest[1]) + buf_deg
    return min_lat, min_lon, max_lat, max_lon


# ---------------------------------------------------------------------------
# Overpass fetch
# ---------------------------------------------------------------------------
def fetch_overpass(query: str, cache: Optional[Dict], cache_key: str) -> dict:
    if cache is not None and cache_key in cache:
        print(f"    [cache] {cache_key}", file=sys.stderr)
        return cache[cache_key]

    # Use GET with URL-encoded query — more compatible with Overpass than POST
    url = OVERPASS_URL + "?" + urllib.parse.urlencode({"data": query})
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "uav_nav-landmark-fetcher/1.0 (research)")
    req.add_header("Accept", "application/json")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                result = json.loads(resp.read().decode())
                if cache is not None:
                    cache[cache_key] = result
                return result
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code} on attempt {attempt}/{MAX_RETRIES}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)
        except Exception as e:
            print(f"    Error on attempt {attempt}/{MAX_RETRIES}: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)
    raise RuntimeError(f"Overpass query failed after {MAX_RETRIES} attempts")


def query_substations(
    bbox: Tuple[float, float, float, float],
    cache: Optional[Dict],
    cache_key: str,
) -> List[SubstationNode]:
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""[out:json][timeout:25];
(
  node["power"="substation"]({min_lat:.4f},{min_lon:.4f},{max_lat:.4f},{max_lon:.4f});
  way["power"="substation"]({min_lat:.4f},{min_lon:.4f},{max_lat:.4f},{max_lon:.4f});
  relation["power"="substation"]({min_lat:.4f},{min_lon:.4f},{max_lat:.4f},{max_lon:.4f});
);
out center tags;
"""
    raw = fetch_overpass(query, cache, cache_key)
    nodes: List[SubstationNode] = []
    for el in raw.get("elements", []):
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        tags = el.get("tags", {})
        # Name: prefer "name", fallback to operator or a position-based label.
        # Include unnamed substations — rural corridors have sparse named coverage.
        name = (tags.get("name")
                or tags.get("operator")
                or tags.get("ref")
                or f"Substation ({lat:.4f}°N {abs(lon):.4f}°W)")
        nodes.append(SubstationNode(
            name=name,
            lat=lat,
            lon=lon,
            osm_id=el["id"],
            voltage=tags.get("voltage"),
            substation_type=tags.get("substation"),
        ))
    return nodes


# ---------------------------------------------------------------------------
# Corridor filtering and spacing selection
# ---------------------------------------------------------------------------
def filter_to_corridor(
    nodes: List[SubstationNode],
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    width_km: float = CORRIDOR_WIDTH_KM,
) -> List[SubstationNode]:
    result = []
    for n in nodes:
        xt, frac = cross_track_and_frac(n.lat, n.lon, origin[0], origin[1], dest[0], dest[1])
        if xt <= width_km and ORIGIN_DEST_EXCLUSION <= frac <= (1.0 - ORIGIN_DEST_EXCLUSION):
            n.cross_track_km = xt
            n.frac = frac
            result.append(n)
    result.sort(key=lambda n: n.frac)
    return result


def select_segments(
    nodes: List[SubstationNode],
    n_segments: int,
    window_frac: float = 0.07,
    max_gap_frac: Optional[float] = None,
) -> List[SubstationNode]:
    """
    Divide the route into n_segments equal portions and pick the substation
    closest to each segment's midpoint (by cross-track distance).

    This guarantees landmark coverage throughout the route, not just near the
    origin where dense urban OSM coverage would otherwise dominate a greedy pick.

    After the segment pass, if max_gap_frac is set and any two consecutive picks
    are further apart than max_gap_frac along-track, fill the gap by picking
    the on-corridor substation closest to the midpoint of the gap.  This
    prevents rural OSM-sparse areas from leaving the UAV in dead reckoning.
    """
    selected: List[SubstationNode] = []
    seen_ids: set = set()

    for i in range(n_segments):
        target_frac = (i + 0.5) / n_segments
        window = [n for n in nodes if abs(n.frac - target_frac) <= window_frac]
        if not window:
            continue
        named = [n for n in window if not n.name.startswith("Substation (")]
        pool = named if named else window
        best = min(pool, key=lambda n: n.cross_track_km)
        if best.osm_id not in seen_ids:
            seen_ids.add(best.osm_id)
            selected.append(best)

    if max_gap_frac is None or len(selected) < 2:
        return selected

    # Gap-fill pass: keep inserting until no gap exceeds max_gap_frac.
    filled = True
    while filled:
        filled = False
        selected.sort(key=lambda n: n.frac)
        for i in range(len(selected) - 1):
            gap = selected[i + 1].frac - selected[i].frac
            if gap <= max_gap_frac:
                continue
            midpoint = (selected[i].frac + selected[i + 1].frac) / 2.0
            # Pick best on-corridor substation near the midpoint, not yet chosen
            # Only fill with landmarks the UAV can actually see (xt < visible
            # range — otherwise the landmark never enters the VLM detection
            # window and contributes nothing).
            candidates = [
                n for n in nodes
                if n.osm_id not in seen_ids
                and selected[i].frac < n.frac < selected[i + 1].frac
                and n.cross_track_km < VISIBLE_RANGE_KM
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda n: abs(n.frac - midpoint))
            seen_ids.add(best.osm_id)
            selected.append(best)
            filled = True
            break  # restart loop with updated `selected`

    selected.sort(key=lambda n: n.frac)
    return selected


def prefer_transmission(nodes: List[SubstationNode]) -> List[SubstationNode]:
    """Sort so transmission substations (high voltage) come before distribution ones."""
    def voltage_kv(n: SubstationNode) -> float:
        if n.voltage is None:
            return 0.0
        # voltage tag may be "115000", "115000;230000", "115 kV", etc.
        try:
            raw = n.voltage.split(";")[0].strip().replace(" kV", "000").replace("kV", "000")
            return float("".join(c for c in raw if c.isdigit() or c == "."))
        except (ValueError, AttributeError):
            return 0.0

    return sorted(nodes, key=lambda n: (-voltage_kv(n), n.frac))


# ---------------------------------------------------------------------------
# Route definitions (origin, dest pairs)
# ---------------------------------------------------------------------------
ROUTES = {
    "very_short": {
        "origin":     (33.6413, -84.4277),
        "dest":       (33.8084, -84.1700),
        "label":      "ATL → Stone Mountain",
        "n_segments": 3,    # ~25 km / 3 ≈ one landmark per 8 km
    },
    "short": {
        "origin":     (33.6413, -84.4277),
        "dest":       (33.9562, -83.9880),
        "label":      "ATL → Lawrenceville",
        "n_segments": 4,    # ~45 km / 4 ≈ one per 11 km
    },
    "baseline": {
        "origin":     (33.6413, -84.4277),
        "dest":       (32.6930, -83.6490),
        "label":      "ATL → Macon (MCN)",
        "n_segments": 8,    # ~128 km / 8 ≈ one per 16 km
    },
    "longer": {
        "origin":     (33.6413, -84.4277),
        "dest":       (32.5640, -82.9850),
        "label":      "ATL → Dublin",
        "n_segments": 10,   # ~180 km / 10 ≈ one per 18 km
    },
    "much_longer": {
        "origin":     (33.6413, -84.4277),
        "dest":       (30.4941, -81.6879),
        "label":      "ATL → Jacksonville",
        "n_segments": 25,       # ~435 km / 25 ≈ one per 17 km target
        "max_gap_frac": 0.05,   # ≈ 22 km; gap-fill if rural OSM leaves bigger holes
    },
}


# ---------------------------------------------------------------------------
# Fetch landmarks for all routes
# ---------------------------------------------------------------------------
def fetch_all_routes(cache: Optional[Dict]) -> Dict[str, List[SubstationNode]]:
    results: Dict[str, List[SubstationNode]] = {}
    for route_id, cfg in ROUTES.items():
        origin = cfg["origin"]
        dest = cfg["dest"]
        print(f"\n[{route_id}] {cfg['label']}", file=sys.stderr)

        bbox = corridor_bbox(origin, dest, buffer_km=25.0)
        print(f"  Querying Overpass bbox "
              f"({bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f})",
              file=sys.stderr)

        raw_nodes = query_substations(bbox, cache, f"substations_{route_id}")
        print(f"  Raw substations found: {len(raw_nodes)}", file=sys.stderr)

        in_corridor = filter_to_corridor(raw_nodes, origin, dest)
        print(f"  In corridor (±{CORRIDOR_WIDTH_KM} km): {len(in_corridor)}", file=sys.stderr)

        # Gap-fill for long routes where rural OSM coverage is sparse.
        # max_gap_frac = 0.05 ≈ 22 km for a 435 km route — well under the
        # 30 km (2× visible range) that would leave a blind patch.
        max_gap = cfg.get("max_gap_frac")
        selected = select_segments(in_corridor, cfg["n_segments"],
                                    max_gap_frac=max_gap)
        # Remove any two consecutive picks that ended up < 8 km apart
        deduped = [selected[0]] if selected else []
        for n in selected[1:]:
            if haversine_km(deduped[-1].lat, deduped[-1].lon, n.lat, n.lon) >= 8.0:
                deduped.append(n)
        selected = deduped
        print(f"  Selected ({cfg['n_segments']} segments, {len(selected)} placed):",
              file=sys.stderr)
        for s in selected:
            dist_km = haversine_km(origin[0], origin[1], dest[0], dest[1])
            print(f"    {s.frac*100:4.0f}%  {s.lat:.4f}°N {s.lon:.4f}°W  "
                  f"xt={s.cross_track_km:.1f}km  {s.name[:50]}", file=sys.stderr)

        results[route_id] = selected

    return results


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------
def _clean_name(name: str) -> str:
    """Trim long OSM names to something reasonable for a Python string."""
    # Remove common OSM boilerplate
    for suffix in [" Substation", " substation", " Power Station", " Switching Station"]:
        if name.endswith(suffix):
            break  # keep as-is; it reads naturally
    # Truncate if very long
    if len(name) > 60:
        name = name[:57] + "..."
    return name


def generate_landmark_list(nodes: List[SubstationNode], indent: str = "        ") -> str:
    if not nodes:
        return f"{indent}[],"
    lines = ["["]
    for n in nodes:
        name = _clean_name(n.name)
        voltage_comment = f"  # {n.voltage} V" if n.voltage else ""
        lines.append(
            f'{indent}    LandmarkConfig("{name}",'
            f"\n{indent}                   LandmarkType.transformer_substation,"
            f" {n.lat:.4f}, {n.lon:.4f},"
            f" {VISIBLE_RANGE_KM:.1f}, {PHYSICAL_HEIGHT_M:.1f}),{voltage_comment}"
        )
    lines.append(f"{indent}],")
    return "\n".join(lines)


def generate_scenario_configs(route_landmarks: Dict[str, List[SubstationNode]]) -> str:
    """Return the complete scenario_configs.py content with real OSM landmarks."""

    def lm_block(route_id: str, indent: str = "        ") -> str:
        nodes = route_landmarks.get(route_id, [])
        return generate_landmark_list(nodes, indent)

    # Daytime scenario uses the same ATL→MCN corridor as baseline
    daytime_nodes = route_landmarks.get("baseline", [])
    # For daytime, take a subset — first 2 that fire closest to destination
    # (mirroring the original intent; we keep all and let the scenario pick)

    return f'''\
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
        "Very short hop: Hartsfield-Jackson ATL → Stone Mountain, GA.\\n"
        "~25 km NNE at 50 m/s.  ~8 minutes.  Tests filter initialisation\\n"
        "and rapid landmark acquisition.  Landmarks: real OSM substations."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=33.8084,
    dest_lon=-84.1700,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks={lm_block("very_short")}
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
        "Short flight: Hartsfield-Jackson ATL → Lawrenceville, GA.\\n"
        "~45 km NE at 50 m/s.  ~15 minutes.  Landmarks: real OSM substations."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=33.9562,
    dest_lon=-83.9880,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks={lm_block("short")}
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
        "Baseline reference: Hartsfield-Jackson ATL → Middle Georgia\\n"
        "Regional (Macon), GA.  ~128 km SE at 50 m/s.  ~43 minutes.\\n"
        "Landmarks: real OSM substations along corridor."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=32.6930,
    dest_lon=-83.6490,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks={lm_block("baseline")}
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
        "Longer flight: Hartsfield-Jackson ATL → Heart of Georgia\\n"
        "Regional (Dublin), GA.  ~180 km ESE at 50 m/s.  ~60 minutes.\\n"
        "Landmarks: real OSM substations along corridor."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=32.5640,
    dest_lon=-82.9850,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks={lm_block("longer")}
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
        "Long-range flight: Hartsfield-Jackson ATL → Jacksonville\\n"
        "International (FL).  ~435 km SSE at 50 m/s.  ~145 minutes.\\n"
        "Landmarks: real OSM substations along corridor."
    ),
    origin_lat=33.6413,
    origin_lon=-84.4277,
    origin_alt_m=463.0,
    dest_lat=30.4941,
    dest_lon=-81.6879,
    airspeed_ms=50.0,
    t_start_utc=datetime(2026, 1, 28, 2, 0, 0, tzinfo=timezone.utc),
    landmarks={lm_block("much_longer")}
    plot_filename="scenario_much_longer.png",
    seed=404,
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
]
'''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated scenario_configs.py; do not write.")
    parser.add_argument("--write", action="store_true",
                        help="Overwrite uav_nav/demo/scenario_configs.py.")
    parser.add_argument("--cache", metavar="FILE",
                        help="JSON file to cache Overpass responses.")
    args = parser.parse_args()

    if not args.dry_run and not args.write:
        parser.print_help()
        print("\nAdd --dry-run to preview or --write to update scenario_configs.py.",
              file=sys.stderr)
        sys.exit(0)

    # Load cache if specified
    cache: Optional[Dict] = None
    if args.cache:
        try:
            with open(args.cache) as f:
                cache = json.load(f)
            print(f"Loaded cache from {args.cache}", file=sys.stderr)
        except FileNotFoundError:
            cache = {}
            print(f"Cache not found — will create {args.cache}", file=sys.stderr)

    # Fetch
    try:
        route_landmarks = fetch_all_routes(cache)
    except RuntimeError as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        print("Check your internet connection or try again later.", file=sys.stderr)
        sys.exit(1)

    # Save cache
    if args.cache and cache is not None:
        with open(args.cache, "w") as f:
            json.dump(cache, f)
        print(f"\nCache saved → {args.cache}", file=sys.stderr)

    # Print summary table
    print("\n" + "=" * 65, file=sys.stderr)
    print("LANDMARK SUMMARY", file=sys.stderr)
    print("=" * 65, file=sys.stderr)
    for route_id, nodes in route_landmarks.items():
        route_label = ROUTES[route_id]["label"]
        print(f"\n{route_id:12s}  {route_label}", file=sys.stderr)
        if not nodes:
            print("  *** NO LANDMARKS FOUND — check corridor width or OSM coverage", file=sys.stderr)
        for n in nodes:
            print(f"  {n.lat:.4f}°N  {n.lon:.4f}°W  "
                  f"xt={n.cross_track_km:.1f}km  {n.name[:50]}", file=sys.stderr)

    # Generate code
    code = generate_scenario_configs(route_landmarks)

    if args.dry_run:
        print("\n" + "=" * 65)
        print("# Generated scenario_configs.py (--dry-run, not written)")
        print("=" * 65)
        print(code)
        return

    if args.write:
        import os
        # Path relative to this script's location (tools/ → uav_nav/demo/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, "..", "uav_nav", "demo", "scenario_configs.py")
        out_path = os.path.normpath(out_path)

        # Backup original
        backup_path = out_path + ".bak"
        if os.path.exists(out_path):
            with open(out_path) as f:
                original = f.read()
            with open(backup_path, "w") as f:
                f.write(original)
            print(f"\nBackup → {backup_path}", file=sys.stderr)

        with open(out_path, "w") as f:
            f.write(code)
        print(f"Written → {out_path}", file=sys.stderr)
        print("Run tests to verify: conda run -n base python -m pytest uav_nav/tests/ -v",
              file=sys.stderr)


if __name__ == "__main__":
    main()
