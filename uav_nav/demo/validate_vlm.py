"""
VLM smoke test: sends a sample image to Ollama (BakLLaVA) and validates
the response conforms to the SemanticObservation schema.

Requires Ollama to be running with bakllava pulled:
    ollama pull bakllava
    ollama serve   (or it auto-starts)

Run with:
    OLLAMA_AVAILABLE=1 python -m uav_nav.demo.validate_vlm [image_path]

Or with pytest:
    OLLAMA_AVAILABLE=1 pytest uav_nav/demo/validate_vlm.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main(image_path: str | None = None):
    if not os.environ.get("OLLAMA_AVAILABLE"):
        print("Set OLLAMA_AVAILABLE=1 to run this test (requires Ollama + bakllava).")
        sys.exit(0)

    from uav_nav.vlm.client import VLMClient, VLMUnavailableError
    from uav_nav.vlm.schemas import SemanticObservation

    client = VLMClient()

    if not client.is_available():
        print("ERROR: Ollama server not reachable at", client.host)
        print("Start Ollama with: ollama serve")
        sys.exit(1)

    # Use provided image or fall back to a simple white test image
    if image_path and Path(image_path).exists():
        src = image_path
        print(f"Using image: {src}")
    else:
        # Create a minimal 100×100 dark image as a stand-in
        try:
            from PIL import Image
            import io
            img = Image.new("RGB", (100, 100), color=(10, 10, 10))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            src = buf.getvalue()
            print("Using synthetic dark test image (100×100).")
        except ImportError:
            print("ERROR: Pillow not installed. Provide an image path argument.")
            sys.exit(1)

    print(f"Querying {client.model} at {client.host} ...")
    try:
        obs = client.query(src)
    except VLMUnavailableError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if obs is None:
        print("FAIL: VLM returned None or unparseable response.")
        sys.exit(1)

    print(f"PASS: Got SemanticObservation")
    print(f"  scene_type:    {obs.scene_type.value}")
    print(f"  sky_visible:   {obs.sky_visible}")
    print(f"  landmarks:     {len(obs.landmarks)}")
    print(f"  stars_visible: {obs.stars_visible}")
    if obs.landmarks:
        for lm in obs.landmarks:
            print(f"    - {lm.type.value} @ {lm.bearing_deg:.1f}° conf={lm.confidence:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLM smoke test")
    parser.add_argument("image", nargs="?", help="Path to a test image file")
    args = parser.parse_args()
    main(args.image)
