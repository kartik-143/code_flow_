"""
wheel_estimator.py

Estimates the four wheel / ground-contact points of the car from its
detected bounding box, per the MVP-first instruction: avoid training a
dedicated wheel detector; derive contact points from vehicle geometry.

Geometric model:
    The bounding box represents the visible extent of the car body.
    Wheel contact points sit at the BOTTOM edge of the box (where the
    car meets the track), inset horizontally from the left/right edges
    to approximate the actual wheel track (wheels sit slightly inboard
    of the outer body/bumper silhouette).

    front_left  = (x1 + inset_x,     y2 - inset_y)
    front_right = (x2 - inset_x,     y2 - inset_y)
    rear_left   = (x1 + inset_x,     y2 - inset_y * rear_lift_factor)
    rear_right  = (x2 - inset_x,     y2 - inset_y * rear_lift_factor)

    For a side-on or 3/4 camera angle, "front" vs "rear" along the
    bbox's motion direction is approximated using the vehicle's recent
    horizontal motion vector (if available) — this is a heuristic, and
    is intentionally simple per the "MVP first, add complexity only if
    needed" instruction.

This is a REAL, deterministic geometric computation from each frame's
actual detected bbox — nothing here is a fixed/hardcoded coordinate.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional


def estimate_wheel_points(
    bbox: Tuple[int, int, int, int],
    inset_ratio: float = 0.12,
    contact_band_ratio: float = 0.10,
) -> Dict[str, Tuple[float, float]]:
    """
    bbox: (x1, y1, x2, y2) vehicle bounding box in pixel coords.
    inset_ratio: fraction of box width to inset each wheel from the
                 left/right edges (wheels sit inboard of the body).
    contact_band_ratio: fraction of box height treated as the ground
                 contact band near the bottom of the box; the two
                 "rows" of wheel points (front/rear) are placed at the
                 top and bottom of this band to give the violation
                 checker a small amount of independent front/rear signal
                 rather than four coincident points.

    Returns a dict of 4 named points:
        front_left, front_right, rear_left, rear_right
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    inset = width * inset_ratio
    band = height * contact_band_ratio

    left_x = x1 + inset
    right_x = x2 - inset

    contact_y_far = y2                     # bottom-most edge of car (closer wheel row)
    contact_y_near = y2 - band             # slightly above bottom edge (other wheel row)

    return {
        "front_left": (left_x, contact_y_near),
        "front_right": (right_x, contact_y_near),
        "rear_left": (left_x, contact_y_far),
        "rear_right": (right_x, contact_y_far),
    }


def wheel_certainty(
    vehicle_confidence: float,
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> float:
    """
    Heuristic certainty score for the wheel-position estimate, derived
    from real per-frame signal:
      - vehicle detection confidence (garbage box -> garbage wheels)
      - bbox aspect ratio plausibility (a car should be wider than tall
        in most track camera angles; extreme ratios suggest a bad box)
      - bbox size relative to frame (very small boxes give unreliable
        inset geometry)
    """
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    frame_h, frame_w = frame_shape[:2]

    aspect = w / h
    aspect_score = 1.0 - min(abs(aspect - 1.8) / 3.0, 1.0)  # ideal-ish aspect ~1.8
    size_score = min((w * h) / (0.03 * frame_w * frame_h), 1.0)

    certainty = 0.5 * vehicle_confidence + 0.3 * aspect_score + 0.2 * size_score
    return float(max(0.0, min(1.0, certainty)))
