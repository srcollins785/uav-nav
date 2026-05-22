"""
nav_viz — Standalone visualization package for UAV navigation scenarios.

This package is intentionally decoupled from uav_nav. It consumes only
the result dict produced by uav_nav.demo.run_scenario.run_scenario().

Public API:
    generate_flight_png(result, output_path)   → str  (path written)
    generate_flight_video(result, output_path) → str  (path written)
"""
from nav_viz.flight_plot import generate_flight_png
from nav_viz.flight_video import generate_flight_video
from nav_viz.solar_comparison_video import generate_solar_comparison_video

__all__ = ["generate_flight_png", "generate_flight_video",
           "generate_solar_comparison_video"]
