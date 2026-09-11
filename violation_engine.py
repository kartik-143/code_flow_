"""
violation_engine.py

Implements the core decision pipeline described in the brief:

    car detected -> boundary detected -> 4 wheel points estimated
    -> each wheel checked vs boundary -> all four beyond line?
    -> persist N consecutive frames -> CONFIRMED VIOLATION

Also computes an explainable confidence score from real per-frame
signals (never a fixed number).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

from .track_boundary_detector import TrackBoundaryDetector


@dataclass
class FrameAnalysis:
    frame_number: int
    timestamp_sec: float
    vehicle: Optional[Dict[str, Any]]
    boundary: Optional[Dict[str, Any]]
    wheel_points: Optional[Dict[str, Tuple[float, float]]]
    wheels_beyond: Optional[Dict[str, bool]]
    all_four_beyond: bool
    status: str                     # "GREEN" or "RED" (candidate/confirmed)
    confirmed: bool
    confidence: float
    consecutive_count: int


class ViolationEngine:
    def __init__(self, required_consecutive_frames: int = 4):
        """
        required_consecutive_frames: how many consecutive frames must
        show "all four wheels beyond the boundary" before a violation
        is CONFIRMED (reduces false positives from detection jitter).
        Configurable per the brief.
        """
        self.required_consecutive_frames = required_consecutive_frames
        self._consecutive_count = 0
        self._reference_side: Optional[float] = None  # which side of the line is "in track"

    def reset(self) -> None:
        self._consecutive_count = 0
        self._reference_side = None

    def analyze_frame(
        self,
        frame_number: int,
        timestamp_sec: float,
        vehicle: Optional[Dict[str, Any]],
        boundary: Optional[Dict[str, Any]],
        wheel_points: Optional[Dict[str, Tuple[float, float]]],
        wheel_certainty: float = 0.5,
    ) -> FrameAnalysis:

        if vehicle is None or boundary is None or wheel_points is None:
            # Missing signal -> cannot evaluate; do not count towards a
            # violation streak, but do not falsely reset a healthy
            # in-progress streak either if it's just a single dropped frame.
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp_sec=timestamp_sec,
                vehicle=vehicle,
                boundary=boundary,
                wheel_points=wheel_points,
                wheels_beyond=None,
                all_four_beyond=False,
                status="GREEN",
                confirmed=False,
                confidence=0.0,
                consecutive_count=self._consecutive_count,
            )

        sides = {
            name: TrackBoundaryDetector.side_of_line(pt, boundary)
            for name, pt in wheel_points.items()
        }

        # Establish which sign corresponds to "on track" the first time
        # we get a confident, clearly-on-one-side reading (majority of
        # wheels on the same side, before any violation has occurred).
        if self._reference_side is None:
            signs = [1 if v >= 0 else -1 for v in sides.values()]
            majority_sign = 1 if signs.count(1) >= signs.count(-1) else -1
            self._reference_side = majority_sign

        # A wheel is "beyond" the boundary if its side is OPPOSITE the
        # established in-track reference side, with a small margin so a
        # wheel sitting exactly on the paint isn't ambiguously flipped
        # by pixel noise.
        margin = 2.0  # pixels, small deliberate tolerance for line thickness
        wheels_beyond = {}
        for name, signed_dist in sides.items():
            beyond = (signed_dist * self._reference_side) < -margin
            wheels_beyond[name] = bool(beyond)

        all_four_beyond = all(wheels_beyond.values())

        if all_four_beyond:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        confirmed = self._consecutive_count >= self.required_consecutive_frames
        status = "RED" if (all_four_beyond or confirmed) else "GREEN"

        confidence = self._compute_confidence(
            vehicle_conf=vehicle["confidence"],
            boundary_conf=boundary["confidence"],
            wheel_certainty=wheel_certainty,
            consecutive_count=self._consecutive_count,
            required=self.required_consecutive_frames,
            sides=sides,
        )

        return FrameAnalysis(
            frame_number=frame_number,
            timestamp_sec=timestamp_sec,
            vehicle=vehicle,
            boundary=boundary,
            wheel_points=wheel_points,
            wheels_beyond=wheels_beyond,
            all_four_beyond=all_four_beyond,
            status=status,
            confirmed=confirmed,
            confidence=confidence,
            consecutive_count=self._consecutive_count,
        )

    @staticmethod
    def _compute_confidence(
        vehicle_conf: float,
        boundary_conf: float,
        wheel_certainty: float,
        consecutive_count: int,
        required: int,
        sides: Dict[str, float],
    ) -> float:
        """
        Explainable confidence = weighted blend of:
          - vehicle detection confidence      (30%)
          - track-line detection confidence   (25%)
          - wheel-position certainty          (20%)
          - temporal persistence ratio        (15%)
          - geometric consistency             (10%)  <- how unanimous /
                                                          unambiguous the
                                                          four wheel-vs-line
                                                          distances are
        """
        persistence_ratio = min(consecutive_count / max(required, 1), 1.0)

        # Geometric consistency: if all four signed distances have the
        # same sign and comparable magnitude, that's a clean, unambiguous
        # geometric read (all four clearly on one side). High variance in
        # sign/magnitude implies a marginal / ambiguous case.
        values = list(sides.values())
        signs = [1 if v >= 0 else -1 for v in values]
        sign_agreement = max(signs.count(1), signs.count(-1)) / len(signs)

        mags = [abs(v) for v in values]
        mean_mag = sum(mags) / len(mags)
        if mean_mag > 0:
            spread = (max(mags) - min(mags)) / (mean_mag + 1e-6)
            magnitude_consistency = max(0.0, 1.0 - min(spread, 1.0))
        else:
            magnitude_consistency = 0.5

        geometric_consistency = 0.5 * sign_agreement + 0.5 * magnitude_consistency

        confidence = (
            0.30 * vehicle_conf
            + 0.25 * boundary_conf
            + 0.20 * wheel_certainty
            + 0.15 * persistence_ratio
            + 0.10 * geometric_consistency
        )
        return float(max(0.0, min(1.0, confidence)))
