"""
Integration Test Runner
=======================
Runs all integration tests and prints a summary table.

Usage:
    python3 integration_tests/run_integration_tests.py

    # Verbose (shows logging output from uav_nav):
    python3 integration_tests/run_integration_tests.py --verbose

Exit code: 0 if all tests pass or skip, 1 if any test fails.

Tests
-----
1. Celestial Image Pipeline — exercises detect_stars() + build_observations()
   + get_celestial_fix_warmstart() against a rendered star-field image.
   Runs entirely offline; no Ollama required.

2. VLM Pipeline — exercises VLMClient.query() against a real satellite tile
   of a transformer substation.  SKIPS if Ollama is not running.

3. Full End-to-End Pipeline — composite scene (sky + ground) through both
   celestial and VLM branches into the same UKF instance.  Proves both
   pipeline branches are wired to UKF update methods.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _status(result) -> str:
    if result.skipped:
        return "SKIP"
    return "PASS" if result.passed else "FAIL"


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    import logging
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s  %(name)s  %(message)s")

    print("\n" + "=" * 60)
    print("  UAV Nav Integration Test Suite")
    print("=" * 60)

    results = []

    # ------------------------------------------------------------------
    # Test 1: Celestial image pipeline
    # ------------------------------------------------------------------
    print("\n[1/4] Celestial Image Pipeline")
    print("      render star field → detect_stars → build_observations")
    print("      → get_celestial_fix_warmstart")

    t0 = time.time()
    from integration_tests.test_celestial_image_pipeline import run as run_celestial
    cel_result = run_celestial()
    cel_elapsed = time.time() - t0

    cel_result.print_summary()
    results.append(("Celestial Image Pipeline", cel_result, cel_elapsed))

    # ------------------------------------------------------------------
    # Test 2: VLM pipeline (BakLLaVA / Ollama — skips if not installed)
    # ------------------------------------------------------------------
    print("\n[2/4] VLM Pipeline (BakLLaVA)")
    print("      ESRI satellite tile → VLMClient (BakLLaVA) → SemanticObservation")

    t0 = time.time()
    from integration_tests.test_vlm_pipeline import run as run_vlm
    vlm_result = run_vlm()
    vlm_elapsed = time.time() - t0

    vlm_result.print_summary()
    results.append(("VLM Pipeline (BakLLaVA)", vlm_result, vlm_elapsed))

    # ------------------------------------------------------------------
    # Test 3: Full end-to-end pipeline (composite scene → UKF updates)
    # ------------------------------------------------------------------
    print("\n[3/3] Full End-to-End Pipeline (Composite Scene)")
    print("      sky (stars) + ground (landmark) → celestial fix + VLM bearing → UKF")

    t0 = time.time()
    from integration_tests.test_full_pipeline import run as run_full_pipeline
    full_result = run_full_pipeline()
    full_elapsed = time.time() - t0

    full_result.print_summary()
    results.append(("Full Pipeline (composite)", full_result, full_elapsed))

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Test':<30}  {'Status':<6}  {'Time':>6}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}")

    any_fail = False
    for name, r, elapsed in results:
        s = _status(r)
        if s == "FAIL":
            any_fail = True
        print(f"  {name:<30}  {s:<6}  {elapsed:5.1f}s")

    print("=" * 60)

    if any_fail:
        print("\n  Result: FAIL — see details above\n")
        return 1
    else:
        print("\n  Result: ALL PASS (or SKIP)\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
