Inverse Celestial Navigation (stars, offline)

Files
- inverse_celnav.py: CLI program that solves latitude/longitude from star observations.
- example_observations.json: Example input.

Run
  python3 inverse_celnav.py example_observations.json

What to put in the input JSON
- utc: observation timestamp in UTC (ISO-8601).
- observations[].ra_deg / dec_deg: star coordinates (J2000) from your offline catalog.
- observations[].alt_deg: measured elevation above the horizon (degrees).
- observations[].az_deg: measured true azimuth (degrees from true north, east-positive). Optional.
- sigma_*: measurement uncertainty (degrees). Helps the solver weight observations.
