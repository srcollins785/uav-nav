# uav-nav — Offline GPS-Denied UAV Navigation

Offline, GPS-denied UAV navigation research framework built at Clark Atlanta University (Ph.D. in Cyber-Physical Systems). Preliminary work supporting an NSF CPS Medium proposal on resilient cyber-physical systems.

The UAV receives a single GPS fix at launch, then navigates entirely using onboard sensors fused in a **10-dimensional Unscented Kalman Filter**.

---

## Sensor Suite

| Sensor | Rate | Role |
|---|---|---|
| IMU (accel + gyro) | 100 Hz | Dead reckoning — propagates position, velocity, heading |
| Barometer | 10 Hz | Altitude correction |
| Magnetometer | 10 Hz | Heading correction |
| Camera → gemma3:4b VLM | 0.1 Hz | Landmark bearing observations |
| Camera → Celestial solver | 0.017 Hz | Absolute lat/lon (nighttime) |
| Camera → Solar tracker | 0.1 Hz | Absolute lat/lon (daytime) |

**UKF state vector (10-D):** `[lat_deg, lon_deg, alt_m, v_north, v_east, v_down, heading_deg, baro_bias_m, mag_bias_deg, gyro_bias_z_dps]`

> VLM observations are geometry-derived (simulated bearings), not live gemma3:4b inference. Results characterize estimator and data-association behaviour, not end-to-end perception performance.

---

## Scenarios

| ID | Name | Distance | Landmarks | Route |
|---|---|---|---|---|
| `very_short` | Very Short | 30 km | 3 | ATL → near Macon |
| `short` | Short | 60 km | 6 | ATL → Macon |
| `baseline` | Baseline (ATL→MCN) | 128 km | 11 | ATL → Macon extended |
| `longer` | Longer | 240 km | 16 | ATL → Savannah |
| `much_longer` | Much Longer | 435 km | 22 | ATL → Jacksonville, FL |

---

## Pass Criterion

**Sponsor target: final position error < 50 m.**

---

## Key Results (10-seed sweeps, current codebase)

### Nighttime (celestial aiding)
| Scenario | Pass@50m | Mean final err |
|---|---|---|
| Very Short | 9/10 | 31 m |
| Short | 9/10 | 26 m |
| Baseline | 9/10 | 31 m |
| Longer | 8/10 | 35 m |
| Much Longer | **9/10** | **29 m** |

### Daytime (solar aiding, all-5 sweep)
| Scenario | Pass@50m | Mean final err |
|---|---|---|
| Very Short | 10/10 | 20 m |
| Short | 9/10 | 22 m |
| Baseline | 10/10 | 30 m |
| Longer | 8/10 | 36 m |
| Much Longer | 8/10 | 40 m |

The Much Longer (435 km) scenario improved from 6/10 to 9/10 by adding 21 interpolated waypoints between OSM substation pairs (43 total landmarks). Density sweep analysis at `results/landmark_density/much_longer/density_summary.md`.

---

## Reproducing the Paper's Results

Everything needed is in this repository. See **[REPRODUCE.md](REPRODUCE.md)** for the
full guide; the short version:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-repro.txt
python tools/verify_reproduction.py
```

That checks your environment against the authors' per-seed data in
[`results/paper/`](results/paper/), which backs every table in the paper.

> **The celestial solver must be present.** It ships at `third_party/inverse_celnav`.
> If it is missing, the code prints one easily-missed warning and runs with celestial
> fixes disabled — which moves `very_short` seed 1 from **27.32 m** to 17.51 m. All
> published numbers use the solver. `verify_reproduction.py` checks this first.

---

## Environment Setup

```bash
conda activate base
```

All experiments use `conda run -n base python <script>` — no special installation beyond the project's existing conda environment.

---

## Running Experiments

### Quick smoke test (2 seeds, very short scenario)
```bash
conda run -n base python scripts/run_smoke_test.py
conda run -n base python scripts/generate_report.py results/smoke_test/smoke_test_results.csv
```

### Full scenario sweep (all 5 scenarios, N seeds)
```bash
conda run -n base python scripts/run_experiments.py --seeds 10
conda run -n base python scripts/generate_report.py results/latest/run_results.csv
```

### Sensor ablation (A0–A6)
```bash
conda run -n base python scripts/run_ablations.py --seeds 10
conda run -n base python scripts/generate_report.py results/ablations/ablation_results.csv
```

### Landmark density sweep (L0–L4, Baseline scenario)
```bash
conda run -n base python tools/landmark_density_sweep.py --seeds 10
```

### ATL→JAX diagnostic
```bash
conda run -n base python scripts/run_experiments.py much_longer --seeds 10 \
    --output results/jax_diagnostic
```

### Generate report from any CSV
```bash
conda run -n base python scripts/generate_report.py <path/to/results.csv>
```

---

## Sensor Ablation Conditions

| Condition | Description | Sensors |
|---|---|---|
| A0 | IMU only | IMU |
| A1 | + barometer | IMU, Baro |
| A2 | + magnetometer | IMU, Baro, Mag |
| A3 | + airspeed | IMU, Baro, Mag, Airspeed |
| A4 | + VLM landmarks | IMU, Baro, Mag, Airspeed, VLM |
| A5 | + celestial/solar (no VLM) | IMU, Baro, Mag, Airspeed, Cel/Solar |
| A6 | Full system | IMU, Baro, Mag, Airspeed, VLM, Cel/Solar |

---

## Landmark Density Sweep (L0–L4)

Tests how pass rate and final error respond to progressive landmark enrichment.

| Level | Description |
|---|---|
| L0 | 0 landmarks — pure dead reckoning + celestial |
| L1 | Real OSM landmarks only (baseline default) |
| L2 | 2× density — real + interpolated midpoints |
| L3 | 4× density — real + 3 interpolated points per gap |
| L4 | Dense upper bound — one landmark every ~10 km |

### Baseline (ATL→MCN, 128 km) — 10 seeds

| Level | N lm | Pass@50m | Mean err | Median | Max |
|---|---|---|---|---|---|
| L0 | 0 | 4/10 | 64 m | 52 m | 147 m |
| L1 | 5 | 9/10 | 32 m | 29 m | 86 m |
| L2 | 9 | 9/10 | 26 m | 30 m | 52 m |
| L3 | 17 | **10/10** | 30 m | 27 m | 50 m |
| L4 | 11 | **10/10** | **18 m** | 14 m | 45 m |

**Key finding (Baseline):** Evenly-spaced L4 achieves 100% pass at 18 m mean with only 11 landmarks,
outperforming 17-landmark L3 — uniform spacing beats denser-but-clustered.

### Much Longer (ATL→JAX, 435 km) — 10 seeds

| Level | N lm | Pass@50m | Mean err | Median | Max |
|---|---|---|---|---|---|
| L0 | 0 | 2/10 | 212 m | 161 m | 651 m |
| L1 | 22 | 6/10 | 48 m | 39 m | 128 m |
| L2 | 43 | **9/10** | **29 m** | 24 m | 60 m |
| L3 | 85 | 7/10 | 40 m | 40 m | 77 m |
| L4 | 42 | 5/10 | 61 m | 51 m | 172 m |

**Key finding (Much Longer):** L2 (interpolated midpoints, 43 total landmarks) achieves 9/10.
L4 (uniform synthetic) is counterintuitively worse than L1 on long routes — synthetic landmarks
placed on the exact great-circle path appear to interfere with data-association more than the
OSM-anchored interpolations used by L2. `SCENARIO_MUCH_LONGER` has been updated to the L2 configuration.

Run: `conda run -n base python tools/landmark_density_sweep.py --scenario <id> --seeds 10`  
Results: `results/landmark_density/<scenario_id>/density_summary.md`

---

## Tests

```bash
conda run -n base python -m pytest uav_nav/tests/ -v
```

Current suite: **106 tests** across unit, integration, and VLM gating counter tests.

Key test files:
- `uav_nav/tests/test_vlm_gating_counters.py` — VLM observation funnel counters
- `uav_nav/tests/test_ablation_runner.py` — ablation condition structure and CSV builder
- `integration_tests/test_full_pipeline.py` — end-to-end pipeline integration

---

## Project Structure

```
uav_nav/                    Python package (navigation + fusion + VLM)
  vlm/                      VLM pipeline schemas and mock (gemma3:4b via Ollama)
  celestial/                Star catalog + solver
  navigation/               UKF, process model, observation model
  fusion/                   ConstraintConverter — sensor orchestration
  demo/                     Scenario configs and runner
  tests/                    Unit + integration tests

scripts/
  run_experiments.py        Multi-scenario sweep runner
  run_ablations.py          Sensor ablation runner (A0–A6)
  run_smoke_test.py         Quick 2-seed validation
  generate_report.py        Markdown report from any results CSV

tools/
  landmark_density_sweep.py  L0–L4 density sweep
  diagnose_jax.py            ATL→JAX per-seed diagnostic
  vlm_probe.py               VLM model evaluation harness

configs/
  config.yaml               UKF noise parameters, sensor flags

results/                    Experiment outputs (CSV, plots, reports)
architecture/               Software Architecture Document (LaTeX)
research paper/             MILCOM 2026 submission (LaTeX)
```

---

## Authors

Samuel Collins, Clark Atlanta University  
Advisors: Dr. Kishor Datta Gupta, Dr. Roy George
