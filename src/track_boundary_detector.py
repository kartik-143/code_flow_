"""
track_boundary_detector.py

Detects the white track-limit boundary line in a frame using classical
computer vision: HSV white-pixel segmentation -> edge detection ->
probabilistic Hough transform -> robust line fitting.

Output contract, per frame:
    {
        "line": (x1, y1, x2, y2),     # representative line segment, pixel coords
        "slope": float,                 # dy/dx (None if vertical)
        "intercept": float | None,      # y = slope*x + intercept  (in image coords)
        "confidence": float,            # 0..1, based on inlier support
        "is_vertical": bool,
    }
or None if no reliable line found in that frame.

Notes:
- The line is fit fresh from real pixel data every frame (no hardcoded
  coordinates). A short temporal smoothing buffer is used only to damp
  jitter, and only ever *combines* real per-frame detections.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from collections import deque

import cv2
import numpy as np


class TrackBoundaryDetector:
    def __init__(
        self,
        roi_fraction: Tuple[float, float] = (0.35, 1.0),
        smoothing_window: int = 5,
        white_s_max: int = 60,
        white_v_min: int = 170,
    ):
        """
        roi_fraction: (top, bottom) fraction of frame height to search in,
                      since the track boundary is usually in the lower/mid
                      portion of an onboard or trackside camera view.
        white_s_max / white_v_min: HSV thresholds for "white paint".
        """
        self.roi_top_frac, self.roi_bottom_frac = roi_fraction
        self.white_s_max = white_s_max
        self.white_v_min = white_v_min
        self._history: deque = deque(maxlen=smoothing_window)

    def detect(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        h, w = frame.shape[:2]
        roi_top = int(h * self.roi_top_frac)
        roi_bottom = int(h * self.roi_bottom_frac)
        roi = frame[roi_top:roi_bottom, :]

        mask = self._white_mask(roi)
        edges = cv2.Canny(mask, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=int(w * 0.15),
            maxLineGap=25,
        )

        if lines is None or len(lines) == 0:
            return self._use_history_or_none()

        fitted = self._fit_dominant_line(lines, roi_top)
        if fitted is None:
            return self._use_history_or_none()

        self._history.append(fitted)
        return self._smoothed()

    # ------------------------------------------------------------------
    def _white_mask(self, roi: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, self.white_v_min])
        upper = np.array([180, self.white_s_max, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _fit_dominant_line(
        self, lines: np.ndarray, y_offset: int
    ) -> Optional[Dict[str, Any]]:
        """Group Hough segments by orientation, keep the largest cluster
        (by total length), fit a single robust line to its endpoints."""
        segs = lines.reshape(-1, 4)

        angles = np.degrees(
            np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])
        )
        # Normalize angle to 0-180 so opposite-direction segments cluster together
        angles = angles % 180

        # Bucket into 10-degree bins, pick bin with greatest total segment length
        lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        bins = (angles // 10).astype(int)
        bin_scores = {}
        for b, length in zip(bins, lengths):
            bin_scores[b] = bin_scores.get(b, 0.0) + length
        if not bin_scores:
            return None
        best_bin = max(bin_scores, key=bin_scores.get)

        chosen = segs[bins == best_bin]
        if len(chosen) == 0:
            return None

        xs = np.concatenate([chosen[:, 0], chosen[:, 2]])
        ys = np.concatenate([chosen[:, 1], chosen[:, 3]]) + y_offset

        # Robust line fit (least squares on the inlier cluster)
        is_vertical = np.std(xs) < 1e-3 or (np.max(xs) - np.min(xs)) < 3
        if is_vertical:
            x_mean = float(np.mean(xs))
            y_min, y_max = float(np.min(ys)), float(np.max(ys))
            line = (x_mean, y_min, x_mean, y_max)
            slope, intercept = None, None
        else:
            coeffs = np.polyfit(xs, ys, 1)
            slope, intercept = float(coeffs[0]), float(coeffs[1])
            x_min, x_max = float(np.min(xs)), float(np.max(xs))
            y_min = slope * x_min + intercept
            y_max = slope * x_max + intercept
            line = (x_min, y_min, x_max, y_max)

        total_length = float(np.sum(lengths))
        cluster_length = float(np.sum(lengths[bins == best_bin]))
        confidence = float(np.clip(cluster_length / max(total_length, 1e-6), 0.3, 0.98))

        return {
            "line": tuple(map(float, line)),
            "slope": slope,
            "intercept": intercept,
            "confidence": confidence,
            "is_vertical": bool(is_vertical),
        }

    def _smoothed(self) -> Dict[str, Any]:
        if len(self._history) == 1:
            return self._history[-1]

        slopes = [h["slope"] for h in self._history if h["slope"] is not None]
        intercepts = [h["intercept"] for h in self._history if h["intercept"] is not None]
        confs = [h["confidence"] for h in self._history]
        latest = self._history[-1]

        if slopes and intercepts:
            avg_slope = float(np.mean(slopes))
            avg_intercept = float(np.mean(intercepts))
            x1, _, x2, _ = latest["line"]
            y1 = avg_slope * x1 + avg_intercept
            y2 = avg_slope * x2 + avg_intercept
            return {
                "line": (x1, y1, x2, y2),
                "slope": avg_slope,
                "intercept": avg_intercept,
                "confidence": float(np.mean(confs)),
                "is_vertical": False,
            }
        return latest

    def _use_history_or_none(self) -> Optional[Dict[str, Any]]:
        """If detection fails this frame but we have recent history,
        reuse the smoothed recent line with a reduced confidence penalty
        rather than dropping the boundary entirely (real footage will
        have occasional occlusion/motion-blur frames)."""
        if not self._history:
            return None
        result = dict(self._smoothed())
        result["confidence"] = max(0.15, result["confidence"] * 0.6)
        return result

    @staticmethod
    def side_of_line(point: Tuple[float, float], line_info: Dict[str, Any]) -> float:
        """Returns a signed distance-like value: >0 means the point is on
        one side of the line, <0 the other, 0 means exactly on it.
        Sign convention is consistent across calls (same line orientation),
        so it can be used to test whether ALL points share a side."""
        x, y = point
        if line_info["is_vertical"]:
            line_x = line_info["line"][0]
            return x - line_x
        slope = line_info["slope"]
        intercept = line_info["intercept"]
        expected_y = slope * x + intercept
        return y - expected_y
