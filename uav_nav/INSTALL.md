# Installation Guide — UAV Offline Navigation Framework

## Prerequisites

### 1. Ollama (one-time system install)

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama starts automatically as a background daemon.

### 2. Pull the VLM model (~4 GB download)
```bash
ollama pull bakllava
```

Verify it works:
```bash
ollama run bakllava "Describe this image in one sentence."
```

---

## Python Dependencies

From the `uav_nav/` directory (or its parent):

```bash
pip install filterpy opencv-python-headless ollama matplotlib pytest
```

Full install from requirements.txt:
```bash
pip install -r uav_nav/requirements.txt
```

Install the package in editable mode:
```bash
pip install -e uav_nav/
```

### Optional: High-precision astronomy
```bash
pip install skyfield astropy
```

---

## Verify the Existing Celestial Solver

The `inverse_celnav.py` module must be reachable. Its path is configured in
`uav_nav/config.yaml` under `inverse_celnav_path`. The default is:

```
../Offline Vision–Language Navigation/V1/inverse_celnav
```

(relative to the `uav_nav/` parent directory)

Test it:
```bash
python -m uav_nav.demo.validate_celestial
```

Expected: `PASS: Position within 50 km of Atlanta and optimizer converged.`

---

## Run Unit Tests

No Ollama required:
```bash
cd "Dr Gupta's Project"
python -m pytest uav_nav/tests/ -v
```

Expected: 5 test modules, all pass.

---

## Run the Full Demo

```bash
python -m uav_nav.demo.run_demo
```

This runs a 30-minute synthetic Atlanta → Macon night flight through the
complete pipeline (IMU + barometer + magnetometer + VLM semantic landmarks +
celestial navigation + UKF sensor fusion). Takes ~30–60 seconds CPU time.

Expected output:
- JSON-lines position log to stdout
- Final position error < 500 m from Macon
- `uav_nav_demo_trajectory.png` trajectory plot

Quick 5-minute test:
```bash
python -m uav_nav.demo.run_demo --duration 300
```

---

## Run VLM Smoke Test (requires Ollama)

```bash
OLLAMA_AVAILABLE=1 python -m uav_nav.demo.validate_vlm path/to/any_image.jpg
```

Expected: `PASS: Got SemanticObservation`

---

## Directory Layout

```
Dr Gupta's Project/
├── Offline Vision–Language Navigation/
│   └── V1/
│       └── inverse_celnav/
│           └── inverse_celnav.py   ← existing celestial solver (do not modify)
└── uav_nav/                        ← this package
    ├── config.yaml                 ← runtime configuration
    ├── vlm/                        ← BakLLaVA offline VLM pipeline
    ├── celestial/                  ← celestial navigation wrapper
    ├── navigation/                 ← UKF state estimator
    ├── fusion/                     ← orchestration layer
    ├── demo/                       ← demo and validation scripts
    └── tests/                      ← pytest unit tests
```
