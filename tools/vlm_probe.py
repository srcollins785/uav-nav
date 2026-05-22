#!/usr/bin/env python3
"""
tools/vlm_probe.py — quick diagnostic: what does Gemma 3 4B say
about a known-substation aerial image?  Prints the raw model output for
three different prompt styles so we can verify output format before
committing to a calibration design.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import ollama

MODEL = "gemma3:4b"

IMG = Path(__file__).parent / "vlm_calibration" / "very_short_01_Georgia_Power.png"

PROMPTS = {
    "open_description": "Describe what you see in this aerial image in one sentence.",
    "infrastructure_yes_no": (
        "Is there an electrical substation or transformer yard visible in this "
        "aerial image? Answer only yes or no, then explain in one short sentence."
    ),
    "free_landmarks": (
        "This is a top-down aerial view from a UAV camera. List every "
        "man-made infrastructure feature you can identify (roads, substations, "
        "buildings, etc.), one per line."
    ),
}


def main():
    if not IMG.exists():
        print(f"missing: {IMG}")
        sys.exit(1)

    b64 = base64.b64encode(IMG.read_bytes()).decode()
    client = ollama.Client(host="http://localhost:11434")

    for label, prompt in PROMPTS.items():
        print("=" * 70)
        print(f"PROMPT: {label}")
        print(f"   {prompt!r}")
        print("-" * 70)
        resp = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt, "images": [b64]}],
            options={"temperature": 0.1},
        )
        print(resp["message"]["content"])
        print()


if __name__ == "__main__":
    main()
