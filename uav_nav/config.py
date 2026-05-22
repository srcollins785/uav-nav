"""
Runtime configuration for the UAV offline navigation framework.
Loaded once at startup from config.yaml alongside this package.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class ProcessNoise(BaseModel):
    pos_m: float = 0.1
    vel_ms: float = 0.05
    heading_deg: float = 0.1
    baro_bias_m: float = 0.01
    mag_bias_deg: float = 0.01
    gyro_bias_z_dps: float = 0.002   # gyro z bias random walk (deg/s per sqrt(s))


class MeasurementNoise(BaseModel):
    baro_m: float = 2.0
    mag_deg: float = 3.0
    vlm_bearing_deg: float = 5.0      # calibrated camera VLM (DJI Zenmuse with VLM calibration)
    celestial_lat_deg: float = 0.02   # ~2.2 km — 57-star catalog at mid-latitudes, good GDOP
    celestial_lon_deg: float = 0.02   # ~1.8 km
    solar_lat_deg: float = 0.05       # ~5.5 km — tighter with full refraction correction
    solar_lon_deg: float = 0.05       # ~4.6 km


class Settings(BaseModel):
    # VLM
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    vlm_rate_hz: float = 0.1
    vlm_timeout_s: float = 30.0
    vlm_camera_fov_deg: float = 70.0

    # Sensor rates
    imu_rate_hz: float = 100.0
    baro_rate_hz: float = 10.0
    mag_rate_hz: float = 10.0
    celestial_rate_hz: float = 0.017

    # UKF sigma point parameters
    ukf_alpha: float = 0.001
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0

    # Celestial (star-based, night)
    use_azimuth: bool = True
    sigma_alt_deg: float = 0.3
    sigma_az_deg: float = 1.5
    celestial_grid_lat_step: float = 10.0
    celestial_grid_lon_step: float = 20.0
    min_stars_for_fix: int = 1

    # Solar (sun-based, daytime)
    sigma_solar_elev_deg: float = 0.15   # sun disk centroid precision (degrees)
    sigma_solar_az_deg: float = 0.25     # azimuth precision — large disk, good contrast
    solar_gdop_max: float = 5.0          # hard-reject threshold for near-zenith fixes

    # Initial GPS fix
    initial_lat_deg: float = 33.641
    initial_lon_deg: float = -84.427
    initial_alt_m: float = 313.0
    initial_heading_deg: float = 0.0

    # Landmark triangulation
    landmark_min_confidence: float = 0.70
    landmark_min_baseline_m: float = 200.0

    # VLM → known-landmark data association: reject an observation when the
    # best-matching known landmark's predicted bearing differs from the
    # observed bearing by more than this gate.  Prevents misattribution when
    # two same-type landmarks are simultaneously in view.
    vlm_assoc_max_bearing_mismatch_deg: float = 15.0

    # Ambiguity gate: when ≥2 known landmarks of the same type are in view,
    # require the best mismatch to beat the second-best by this margin.  If
    # the top two candidates are within this bearing, the association is
    # ambiguous and the observation is dropped rather than guessed — a
    # wrong pick at σ≈5° pulls the filter hard in the wrong direction and
    # trips NEES divergence.
    vlm_assoc_min_margin_deg: float = 8.0

    # Monocular range estimation from known landmark height
    focal_length_px: float = 2000.0       # DJI Zenmuse X7 equivalent (24 mm, 4/3" sensor)
    vlm_range_sigma_frac: float = 0.05    # 5% fractional range sigma from pixel noise

    # Target pass criterion
    pass_threshold_m: float = 50.0        # Final position error must be < this to PASS

    # Star catalog
    star_catalog_path: str = "celestial/almanac_data/stars.csv"
    star_max_magnitude: float = 4.0

    # Path to existing inverse_celnav module (relative to uav_nav parent dir)
    inverse_celnav_path: str = "Offline Vision\u2013Language Navigation/V1/inverse_celnav"

    # Noise models
    process_noise: ProcessNoise = Field(default_factory=ProcessNoise)
    measurement_noise: MeasurementNoise = Field(default_factory=MeasurementNoise)

    def inverse_celnav_abs_path(self) -> Path:
        """Resolve absolute path to the inverse_celnav directory."""
        base = Path(__file__).parent.parent
        return (base / self.inverse_celnav_path).resolve()

    def star_catalog_abs_path(self) -> Path:
        """Resolve absolute path to the star catalog CSV."""
        base = Path(__file__).parent
        return (base / self.star_catalog_path).resolve()


def load_config(path: Optional[str] = None) -> Settings:
    """Load Settings from a YAML file. Falls back to defaults if file is absent."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Flatten nested dicts for pydantic
        if "process_noise" in data and isinstance(data["process_noise"], dict):
            data["process_noise"] = ProcessNoise(**data["process_noise"])
        if "measurement_noise" in data and isinstance(data["measurement_noise"], dict):
            data["measurement_noise"] = MeasurementNoise(**data["measurement_noise"])
        return Settings(**{k: v for k, v in data.items() if v is not None})
    return Settings()


# Module-level singleton loaded on first import
_config: Optional[Settings] = None


def get_config() -> Settings:
    global _config
    if _config is None:
        _config = load_config()
    return _config
