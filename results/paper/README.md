# results/paper — the per-seed data behind the MILCOM 2026 paper

The paper's Availability statement says per-seed details are available in this
repository. They are here. Every table and statistical claim in the paper maps
to a file below.

These are the authors' original run outputs, not regenerated. To check that
your environment reproduces them, run:

```bash
python tools/verify_reproduction.py          # fast: very_short, 3 seeds
python tools/verify_reproduction.py --all    # every scenario, 10 seeds
```

## What backs what

| File | Backs |
|---|---|
| `nighttime_10seed_sweep.json` | Nighttime 10-seed table: all five scenarios plus SEA→PDX. Per-seed `final_error_m`, `p95_error_m`, VLM/celestial update counts, NEES alarms. |
| `daytime_all5_10seed_sweep.json` | Daytime (solar-aided) 10-seed table, all five scenarios. |
| `ablation_sensor_sweep.json` | Sensor ablation A0–A6, including the bearing-only configuration reported at 10/10 and 29.7 m on the 128 km baseline. |
| `p1_3/paired_errors.csv`, `p1_3/statistical_test.md` | The 30-seed paired solar-vs-multi-star test (p = 0.94 on 128 km, p = 0.10 on 435 km). One row per (scenario, seed) with both modes' errors and their difference. |
| `sea_pdx_statistical_test.json` | The 30-seed SEA→PDX corridor test at 47.5°N, with means, 95% CI, t-statistic and Wilcoxon. |
| `landmark_density/` | L0–L4 landmark-density sweep. Supports the claim that the 435 km route improves from 6/10 to 9/10 with 21 interpolated waypoints (43 landmarks total). |
| `sigma0_sensitivity_sweep.json` | Initial-covariance sensitivity. |
| `p2_4_threshold_sensitivity*.json` | VLM data-association threshold sensitivity. |
| `vlm_benchmark/benchmark_results.json` | VLM detection benchmark. |
| `paper_fig_trajectories.json` | Trajectory data behind the paper's figures. |

## Reading the per-seed files

`nighttime_10seed_sweep.json` and `daytime_all5_10seed_sweep.json` share a
shape: `summaries` (per-scenario aggregates) and `per_scenario` (a list of
per-seed rows). Aggregates are recomputable from the rows — no seed is
excluded, and no seed was selected after the fact.

## One discrepancy, stated rather than hidden

The paper reports Much Longer at **29 m** mean. `nighttime_10seed_sweep.json`
gives **30.8 m** over its ten seeds (9/10 pass@50 m, which matches). The other
four scenarios agree with the paper to rounding. The 1.8 m gap is unresolved;
treat the per-seed data here as authoritative, since it is the run that was
saved.
