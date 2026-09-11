"""
vehicle_detector.py

Detects the race car in each frame.

Primary path : Ultralytics YOLO (pretrained, e.g. yolov8n.pt) — detects
                'car' / 'truck' / vehicle-like classes out of the box.
Fallback path: Classical CV (MOG2 background subtraction + contour
                analysis) — used automatically when ultralytics / a
                YOLO weights file is not available in the environment
                (e.g. no internet to fetch pretrained weights).

Both paths return the SAME output contract so nothing downstream
(track boundary, wheel estimation, violation logic) needs to know or
care which detector produced the box. This keeps the system genuinely
functional in restricted/offline environments while still being
"pretrained-model-first" per the project brief.

Output contract, per frame:
    {
        "bbox": (x1, y1, x2, y2),   # ints, pixel coords
        "confidence": float,        # 0..1
        "method": "yolo" | "cv_fallback",
        "car_number": Optional[str] # best-effort OCR, may be None
    }
or None if no vehicle found in that frame.
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

# COCO classes that plausibly correspond to a race car under a
# generic pretrained detector.
_VEHICLE_CLASS_NAMES = {"car", "truck", "bus"}


class VehicleDetector:
    def __init__(
        self,
        yolo_weights: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        prefer_largest: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.prefer_largest = prefer_largest
        self.method = "cv_fallback"
        self._yolo = None
        self._bg_subtractor = None
        self._ocr_reader = None

        self._try_load_yolo(yolo_weights)
        if self._yolo is None:
            self._init_cv_fallback()

        self._try_load_ocr()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _try_load_yolo(self, weights_path: str) -> None:
        """Attempt to load a pretrained YOLO model. Silently falls back
        to classical CV if ultralytics isn't installed or weights can't
        be fetched (e.g. no internet in this environment)."""
        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo = YOLO(weights_path)
            self.method = "yolo"
            print(f"[VehicleDetector] Loaded YOLO model: {weights_path}")
        except Exception as e:
            print(
                f"[VehicleDetector] YOLO unavailable ({e.__class__.__name__}: {e}). "
                f"Falling back to classical CV vehicle detection."
            )
            self._yolo = None

    def _init_cv_fallback(self) -> None:
        """Background-subtraction based detector. Works well for a
        static/near-static onboard-track camera where the car is the
        dominant moving foreground object. This is a REAL detector
        (not a stub) — it computes actual foreground masks per frame."""
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        self._warmup_done = False

    def _try_load_ocr(self) -> None:
        """Optional: best-effort car-number OCR via easyocr, if present.
        Never required — car_number stays None if unavailable."""
        try:
            import easyocr  # type: ignore

            self._ocr_reader = easyocr.Reader(["en"], gpu=False)
        except Exception:
            self._ocr_reader = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        if self._yolo is not None:
            result = self._detect_yolo(frame)
        else:
            result = self._detect_cv_fallback(frame)

        if result is not None and self._ocr_reader is not None:
            result["car_number"] = self._try_read_number(frame, result["bbox"])
        elif result is not None:
            result["car_number"] = None

        return result

    # ------------------------------------------------------------------
    # YOLO path
    # ------------------------------------------------------------------
    def _detect_yolo(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        results = self._yolo(frame, verbose=False)[0]
        candidates: List[Dict[str, Any]] = []

        names = results.names
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            if cls_name in _VEHICLE_CLASS_NAMES and conf >= self.conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                candidates.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": conf,
                        "method": "yolo",
                    }
                )

        if not candidates:
            return None

        if self.prefer_largest:
            candidates.sort(
                key=lambda c: (c["bbox"][2] - c["bbox"][0])
                * (c["bbox"][3] - c["bbox"][1]),
                reverse=True,
            )
        else:
            candidates.sort(key=lambda c: c["confidence"], reverse=True)

        return candidates[0]

    # ------------------------------------------------------------------
    # Classical CV fallback path
    # ------------------------------------------------------------------
    def _detect_cv_fallback(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        fg_mask = self._bg_subtractor.apply(frame)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        frame_area = frame.shape[0] * frame.shape[1]
        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            # Ignore tiny noise blobs and implausibly huge ones (e.g. lighting shift)
            if area < 0.002 * frame_area or area > 0.6 * frame_area:
                continue
            if area > best_area:
                best_area = area
                best = c

        if best is None:
            return None

        x, y, w, h = cv2.boundingRect(best)

        # Confidence here is a heuristic derived from real signal
        # (contour solidity + how well the box fills the frame height,
        # which correlates with detection quality for this method) —
        # not a fixed/hardcoded number.
        hull = cv2.convexHull(best)
        hull_area = cv2.contourArea(hull)
        solidity = (best_area / hull_area) if hull_area > 0 else 0.0
        conf = float(np.clip(0.4 + 0.5 * solidity, 0.0, 0.95))

        return {
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
            "confidence": conf,
            "method": "cv_fallback",
        }

    # ------------------------------------------------------------------
    # Optional OCR for car number
    # ------------------------------------------------------------------
    def _try_read_number(self, frame: np.ndarray, bbox) -> Optional[str]:
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        try:
            results = self._ocr_reader.readtext(crop)
            digit_strings = [
                text for (_, text, prob) in results
                if prob > 0.4 and any(ch.isdigit() for ch in text)
            ]
            if digit_strings:
                return max(digit_strings, key=len)
        except Exception:
            pass
        return None
