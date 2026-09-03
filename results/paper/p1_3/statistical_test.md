# P1.3 Statistical Test: Solar vs Multi-Star Celestial Navigation

**Date:** 2026-05-02
**Seeds:** 30 paired (same seed used for both modes)
**VLM:** Synthetic baseline (P_detect=100%, σ_bearing=2°) — paper headline condition
**Pass criterion:** 50 m final position error

---

## Method

For each seed, BOTH daytime (solar) and nighttime (multi-star) modes are run
with identical IMU noise, VLM observations, and scenario parameters.  Only the
celestial sensor model changes: daytime uses astropy sun position; nighttime uses
astropy star catalogue.  Daytime `t_start_utc` = 2026-01-28 15:00 UTC (10 AM EST),
giving a solar elevation of ~35° at ATL (33.6°N).  Nighttime `t_start_utc` =
2026-01-28 02:00 UTC (scenario default).

**Statistical tests:**
- **Paired t-test** (parametric): tests whether mean(day) ≠ mean(night)
- **Wilcoxon signed-rank** (non-parametric): robust to non-Gaussianity

A 95% CI that **includes 0** means the data are consistent with no difference
at α=0.05; the claim "solar outperforms multi-star" (or vice versa) is not
supported at this sample size.  A CI that **excludes 0** supports a directional claim.

---

## Results

### Baseline (ATL→MCN 128 km)

| Metric | Daytime (solar) | Nighttime (multi-star) |
|---|---|---|
| Mean final error | 31.0 m | 30.7 m |
| Median final error | 29.5 m | 28.0 m |
| Std dev | 23.1 m | 15.2 m |
| Pass@50m (30 seeds) | 26/30 | 27/30 |

**Paired difference (day − night):**
- Mean difference: +0.3 m (positive = solar worse, negative = solar better)
- 95% CI: [-8.4 m, +9.1 m] — contains 0
- Paired t-test: t = 0.080, p = 0.9369 → not significant at α=0.05
- Wilcoxon signed-rank: W = 222, p = 0.8394 → not significant at α=0.05

### Much Longer (ATL→JAX 435 km)

| Metric | Daytime (solar) | Nighttime (multi-star) |
|---|---|---|
| Mean final error | 38.2 m | 31.5 m |
| Median final error | 30.6 m | 30.7 m |
| Std dev | 27.1 m | 19.7 m |
| Pass@50m (30 seeds) | 25/30 | 26/30 |

**Paired difference (day − night):**
- Mean difference: +6.8 m (positive = solar worse, negative = solar better)
- 95% CI: [-1.4 m, +15.0 m] — contains 0
- Paired t-test: t = 1.689, p = 0.1019 → not significant at α=0.05
- Wilcoxon signed-rank: W = 170, p = 0.2054 → not significant at α=0.05


---

## Interpretation

**Both scenarios show no statistically significant difference** (all 95% CIs contain 0).

This supports the paper's **operational framing** over a performance claim:
> "Solar navigation achieves comparable accuracy to multi-star navigation
> (no significant difference at α=0.05), while enabling 24-hour operations
> without requiring a star-tracking camera."

The directional claim "solar outperforms multi-star on the longest routes"
from the prior draft is not supported by the paired data.  The recommended
revision is to reframe as operational equivalence with the deployment advantage
of requiring only one celestial sensor for 24-hour capability.

---

## Raw Data

- `results/p1_3/paired_errors.csv` — per-seed errors for both modes and scenarios
