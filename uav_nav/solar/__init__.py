"""
Solar navigation subpackage — daytime position fixing via sun position.

Provides a parallel to uav_nav.celestial for daytime operation:
  - ephemeris   : NOAA solar position algorithm (no external dependencies)
  - solver      : Warm-start lat/lon solver from a single solar observation
  - observation_builder : Build SolarObservation from measurement or simulation
"""
