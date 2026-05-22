"""
OpenCV-based star centroid detector for night-sky images.

Pipeline:
  1. Convert to grayscale.
  2. Adaptive Gaussian threshold to isolate bright spots on dark sky.
  3. SimpleBlobDetector with circularity and area filters.
  4. Return list of StarCandidate(pixel_x, pixel_y, estimated_magnitude).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class StarCandidate:
    pixel_x: float   # Pixel column (from left)
    pixel_y: float   # Pixel row (from top)
    area_px: float   # Blob area in pixels (proxy for brightness)
    estimated_magnitude: float  # Estimated visual magnitude (smaller = brighter)


def detect_stars(
    image_source: Union[str, Path, bytes, np.ndarray],
    max_stars: int = 20,
) -> List[StarCandidate]:
    """
    Detect star centroids in a night-sky image.

    Args:
        image_source: File path, raw bytes, or numpy BGR/grayscale array.
        max_stars: Maximum number of candidates to return (brightest first).

    Returns:
        List of StarCandidate, sorted by estimated brightness (brightest first).
        Returns empty list if opencv-python-headless is not installed or no stars found.
    """
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python-headless not installed. Star detection disabled.")
        return []

    gray = _load_gray(image_source, cv2)
    if gray is None:
        return []

    # --- Step 1: Normalize to improve dynamic range ---
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- Step 2: Adaptive threshold to isolate bright points ---
    # Invert so stars (bright) become white blobs
    _, thresh = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 3: Blob detection ---
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 2.0
    params.maxArea = 200.0
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 255  # bright blobs

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)

    if not keypoints:
        log.debug("No star candidates detected.")
        return []

    # --- Step 4: Convert to StarCandidate ---
    max_area = max(kp.size ** 2 for kp in keypoints) or 1.0
    candidates: List[StarCandidate] = []

    for kp in keypoints:
        area = kp.size ** 2
        # Rough magnitude: brightest star (area=max) → 1.0, dimmest → 5.0
        relative_brightness = area / max_area
        est_mag = 1.0 + (1.0 - relative_brightness) * 4.0

        candidates.append(StarCandidate(
            pixel_x=float(kp.pt[0]),
            pixel_y=float(kp.pt[1]),
            area_px=float(area),
            estimated_magnitude=est_mag,
        ))

    candidates.sort(key=lambda c: c.estimated_magnitude)
    return candidates[:max_stars]


def _load_gray(source, cv2) -> Optional[np.ndarray]:
    """Load image from any supported source and convert to grayscale."""
    try:
        if isinstance(source, np.ndarray):
            img = source
        elif isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                log.warning("Could not read image: %s", source)
                return None
        elif isinstance(source, bytes):
            buf = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        else:
            log.warning("Unsupported image source type: %s", type(source))
            return None

        if img is None:
            return None

        if len(img.shape) == 2:
            return img  # already grayscale
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    except Exception as exc:
        log.warning("Image loading failed: %s", exc)
        return None
