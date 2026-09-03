# Landmark Density Sensitivity

## Scenario: Much Longer

| Level | Description | N landmarks | Pass@50m | Mean err | Median | Max |
|---|---|---|---|---|---|---|
| L0 | 0 landmarks (dead reckoning) | 0 | 2/10 | 211.7 m | 161.1 m | 651.1 m |
| L1 | 22 landmarks (real OSM, baseline) | 22 | 6/10 | 47.5 m | 38.8 m | 127.5 m |
| L2 | 43 landmarks (2× density) | 43 | 9/10 | 29.3 m | 23.6 m | 60.0 m |
| L3 | 85 landmarks (4× density) | 85 | 7/10 | 40.2 m | 39.9 m | 76.5 m |
| L4 | 42 landmarks (1 per ~10 km, upper bound) | 42 | 5/10 | 61.1 m | 51.0 m | 172.2 m |
