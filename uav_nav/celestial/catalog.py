"""
Offline star catalog — 57 standard navigational stars (J2000 epoch).

Loads from a local CSV so the module works without any network access.
The CSV at almanac_data/stars.csv contains columns:
    name, hr, ra_deg, dec_deg, magnitude
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from uav_nav.config import get_config

_DEFAULT_CSV = Path(__file__).parent / "almanac_data" / "stars.csv"


@dataclass
class StarRecord:
    name: str
    hr: int          # Harvard Revised catalog number
    ra_deg: float    # Right Ascension, degrees, J2000
    dec_deg: float   # Declination, degrees, J2000
    magnitude: float


def load_catalog(
    csv_path: Optional[Path] = None,
    max_magnitude: Optional[float] = None,
) -> List[StarRecord]:
    """
    Load the star catalog from CSV.

    Args:
        csv_path: Override the default catalog path.
        max_magnitude: Discard stars dimmer than this value.
                       Defaults to config star_max_magnitude.

    Returns:
        List of StarRecord, sorted by magnitude (brightest first).
    """
    if csv_path is None:
        cfg = get_config()
        catalog_file = cfg.star_catalog_abs_path()
    else:
        catalog_file = csv_path

    if max_magnitude is None:
        max_magnitude = get_config().star_max_magnitude

    records: List[StarRecord] = []
    with open(catalog_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mag = float(row["magnitude"])
            if mag <= max_magnitude:
                records.append(
                    StarRecord(
                        name=row["name"].strip(),
                        hr=int(row["hr"]),
                        ra_deg=float(row["ra_deg"]),
                        dec_deg=float(row["dec_deg"]),
                        magnitude=mag,
                    )
                )

    records.sort(key=lambda s: s.magnitude)
    return records


def build_name_index(catalog: List[StarRecord]) -> Dict[str, StarRecord]:
    """Return a dict mapping lowercase star name → StarRecord."""
    return {s.name.lower(): s for s in catalog}


# Module-level singleton
_catalog: Optional[List[StarRecord]] = None
_name_index: Optional[Dict[str, StarRecord]] = None


def get_catalog() -> List[StarRecord]:
    global _catalog
    if _catalog is None:
        _catalog = load_catalog()
    return _catalog


def get_name_index() -> Dict[str, StarRecord]:
    global _name_index
    if _name_index is None:
        _name_index = build_name_index(get_catalog())
    return _name_index


def lookup(name: str) -> Optional[StarRecord]:
    """Case-insensitive lookup by star name."""
    return get_name_index().get(name.lower().strip())
