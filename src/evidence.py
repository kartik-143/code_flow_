"""
evidence.py

Draws a clean, professional annotation overlay on the confirmed
violation frame and saves it to disk.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import os

import cv2
import numpy as np

GREEN = (60, 200, 60)
RED = (40, 40, 230)
WHITE = (255, 255, 255)
YELLOW = (30, 220, 240)


def annotate_frame(
    frame: np.ndarray,
    vehicle: Dict[str, Any],
    boundary: Dict[str, Any],
    wheel_points: Dict[str, Tuple[float, float]],
    wheels_beyond: Dict[str, bool],
    status: str,
    confidence: float,
    frame_number: int,
    timestamp_sec: float,
    turn_label: str = "",
    car_number: str | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    accent = RED if status == "RED" else GREEN

    # Vehicle bbox
    x1, y1, x2, y2 = vehicle["bbox"]
    cv2.rectangle(out, (x1, y1), (x2, y2), accent, 2)
    label = f"CAR{' #' + car_number if car_number else ''} {vehicle['confidence']*100:.0f}%"
    cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 2)

    # Track boundary line
    lx1, ly1, lx2, ly2 = boundary["line"]
    cv2.line(out, (int(lx1), int(ly1)), (int(lx2), int(ly2)), YELLOW, 2)
    cv2.putText(
        out, "TRACK LIMIT", (int(lx1), int(ly1) - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2,
    )

    # Wheel / contact points
    for name, (px, py) in wheel_points.items():
        beyond = wheels_beyond.get(name, False)
        color = RED if beyond else GREEN
        cv2.circle(out, (int(px), int(py)), 6, color, -1)
        cv2.circle(out, (int(px), int(py)), 8, WHITE, 1)

    # Status banner
    banner_h = 70
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    status_text = "VIOLATION" if status == "RED" else "CLEAN"
    dot_color = RED if status == "RED" else GREEN
    cv2.circle(out, (28, banner_h // 2), 12, dot_color, -1)
    cv2.putText(
        out, status_text, (50, banner_h // 2 + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2,
    )
    cv2.putText(
        out, f"Confidence: {confidence*100:.0f}%",
        (250, banner_h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2,
    )
    right_text = f"Frame {frame_number}  |  t={timestamp_sec:.2f}s"
    if turn_label:
        right_text += f"  |  {turn_label}"
    (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(
        out, right_text, (w - tw - 15, banner_h // 2 + 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1,
    )

    return out


def save_evidence(
    annotated_frame: np.ndarray, output_dir: str, incident_id: int
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"incident_{incident_id:03d}.jpg")
    cv2.imwrite(path, annotated_frame)
    return path
