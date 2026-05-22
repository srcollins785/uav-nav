"""
Regression tests for uav_nav.navigation.uncertainty.build_Q.

Process noise covariance for a continuous-time white-noise model must
accumulate *linearly* in dt — Q(dt) = σ²·dt — not quadratically.  A
prior implementation used (σ·dt)² which underweighted Q by a factor of
dt per predict step, making the filter overconfident and producing a
non-monotonic "more measurements → more error" failure mode under dense
landmark coverage.  This test guards against regression.
"""
import numpy as np

from uav_nav.navigation.uncertainty import build_Q


def test_Q_scales_linearly_with_dt():
    """Q entries must scale proportionally to dt, not dt²."""
    Q_01 = build_Q(0.01)
    Q_10 = build_Q(0.10)
    # 10× larger dt should give 10× larger Q (not 100× as the old bug did)
    ratio = np.divide(Q_10.diagonal(), Q_01.diagonal(),
                       out=np.zeros_like(Q_01.diagonal()),
                       where=Q_01.diagonal() != 0)
    for i, r in enumerate(ratio):
        if Q_01.diagonal()[i] == 0:
            continue
        assert 9.5 < r < 10.5, (
            f"Q[{i},{i}] scaled by {r:.2f} for 10× dt; expected ~10× (linear)"
        )


def test_Q_has_correct_units():
    """
    Sanity check: integrating σ²·dt for 1 s should yield σ² variance.
    Verifies the altitude diagonal matches the configured pos_m²
    regardless of whether pos_m comes from Python defaults or yaml.
    """
    from uav_nav.config import get_config
    pos_m = get_config().process_noise.pos_m
    Q = build_Q(1.0)
    expected = pos_m ** 2
    assert abs(Q[2, 2] - expected) < 1e-9, (
        f"Q[2,2] = {Q[2,2]}, expected {expected} (= pos_m² for σ²·dt at dt=1)"
    )


def test_Q_zero_dt_gives_zero():
    """No time step ⇒ no accumulated process noise."""
    Q = build_Q(0.0)
    assert np.allclose(Q, 0.0)


def test_Q_is_symmetric_and_psd():
    """Covariance matrices must be symmetric positive semi-definite."""
    Q = build_Q(0.01)
    assert np.allclose(Q, Q.T), "Q must be symmetric"
    eigs = np.linalg.eigvalsh(Q)
    assert eigs.min() >= -1e-12, f"Q must be PSD; min eig = {eigs.min()}"
