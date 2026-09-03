# Reproducing the MILCOM 2026 results

Everything needed to regenerate the paper's numbers is in this repository:
the code, the celestial solver, the per-seed data to compare against, and a
script that tells you whether your environment agrees.

## Quick start

```bash
git clone https://github.com/srcollins785/uav-nav && cd uav-nav
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-repro.txt

python tools/verify_reproduction.py
```

Expected output:

```
  celestial solver: OK  (.../third_party/inverse_celnav)
very_short
  seed   published       yours     delta
     1     27.325m     27.319m    0.006m
     2     17.281m     17.078m    0.203m
     3     66.045m     66.270m    0.225m
PASS: all 3 run(s) within 1.0 m (worst delta 0.225 m)
```

If that passes, your environment reproduces the paper. Then:

```bash
python tools/verify_reproduction.py --all      # all 5 scenarios, 10 seeds
python tools/nighttime_10seed_sweep.py         # regenerate the nighttime table
python tools/daytime_all5_10seed_sweep.py      # regenerate the daytime table
python tools/ablation_sensor_sweep.py          # regenerate the ablation table
python tools/p1_3_statistical_test.py          # regenerate the 30-seed paired test
```

A 10-seed sweep of all five scenarios takes roughly 45–60 minutes
single-threaded; the 435 km scenario alone is ~870 VLM epochs per seed.

## The one thing that silently breaks reproduction

The celestial solver lives at `third_party/inverse_celnav`, and
`uav_nav/config.yaml` points at it via `inverse_celnav_path`. **If it is
missing, the code prints one warning line and continues with celestial fixes
disabled.** That single line is easy to miss, and the consequences are not
subtle:

| very_short seed 1 | result |
|---|---|
| solver present (published configuration) | **27.32 m** |
| solver absent | 17.51 m |

Every number in the paper was produced with the solver present.
`tools/verify_reproduction.py` checks for it first and exits non-zero with an
explanation rather than letting you compare against the wrong configuration.

Note also that the scenario generator draws all sensor noise from one shared
random stream. A consequence: the solver's presence changes not just the
celestial fixes but the IMU, barometer, magnetometer and VLM noise
realizations too, because the celestial code path consumes draws when it is
active and returns early when it is not. This is why the difference above is
9.8 m rather than the ~0.06 m the celestial fixes themselves contribute.

This behaviour is preserved deliberately here so that the artifact matches
the paper. It is fixed in the successor repository, which is therefore not
numerically comparable to this one.

## Library versions matter at the sub-metre level

`requirements-repro.txt` pins a configuration verified to reproduce the
published per-seed values to sub-metre agreement. `uav_nav/requirements.txt`
keeps lower bounds, which is the right thing for *using* the package but not
for reproducing exact figures.

How close depends on the route, because the filter integrates for longer
before the final fix:

| scenario | route | seeds checked | worst delta |
|---|---|---|---|
| `very_short` | 30 km | 3 | 0.23 m |
| `much_longer` | 435 km | 10 | 0.82 m |

The exact versions used for the original 2026-05 runs were not recorded, so
sub-metre agreement is the best available standard. `verify_reproduction.py`
defaults to a 1.0 m tolerance for that reason. Note that this leaves only
about 0.2 m of headroom on the 435 km route: on a different BLAS you may
exceed it and see a FAIL that reflects nothing but numerical drift. Pass
`--tolerance 2.0` when checking `longer` or `much_longer`, and read the
reported worst delta rather than only the PASS/FAIL line.

## Where the numbers live

`results/paper/` holds the authors' original per-seed outputs, with an index
in `results/paper/README.md` mapping each file to the table or claim it backs.

## Known discrepancy

The paper reports Much Longer at 29 m mean; the saved 10-seed data gives
30.8 m (pass rate 9/10 agrees). The other four scenarios match to rounding.
See `results/paper/README.md`.
