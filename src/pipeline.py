"""
pipeline.py

Orchestrates the end-to-end flow:

    read video -> per frame: [detect vehicle -> detect boundary ->
    estimate wheels -> evaluate violation] -> on confirmed violation:
    [annotate + save evidence, generate replay, log incident]

Designed to be driven either from a CLI script or the Streamlit app,
with a progress callback so a UI can show live status.
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, List

import cv2
import numpy as np

from .vehicle_detector import VehicleDetector
from .track_boundary_detector import TrackBoundaryDetector
from .wheel_estimator import estimate_wheel_points, wheel_certainty
from .violation_engine import ViolationEngine, FrameAnalysis
from .evidence import annotate_frame, save_evidence
from .replay import generate_replay_clip
from .incident_log import IncidentLog


@dataclass
class PipelineConfig:
    required_consecutive_frames: int = 4
    yolo_weights: str = "yolov8n.pt"
    vehicle_conf_threshold: float = 0.35
    turn_label: str = "Turn 1"
    output_dir: str = "output"
    replay_seconds_before: float = 1.5
    replay_seconds_after: float = 1.5
    frame_skip: int = 1  # process every Nth frame (1 = every frame)
    max_frames: Optional[int] = None  # cap for quick demo runs; None = full video


@dataclass
class PipelineResult:
    total_frames_processed: int
    fps: float
    duration_sec: float
    incidents: List[dict]
    overall_status: str  # "GREEN" or "RED" — RED if ANY confirmed violation occurred
    vehicle_detection_method: str
    annotated_video_path: Optional[str]
    log_path: str


ProgressCallback = Callable[[int, int, FrameAnalysis], None]


def run_pipeline(
    video_path: str,
    config: PipelineConfig = PipelineConfig(),
    progress_callback: Optional[ProgressCallback] = None,
    write_annotated_video: bool = True,
) -> PipelineResult:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(config.output_dir, exist_ok=True)
    evidence_dir = os.path.join(config.output_dir, "evidence")
    replay_dir = os.path.join(config.output_dir, "replays")
    log_path = os.path.join(config.output_dir, "logs", "incident_log.json")

    vehicle_detector = VehicleDetector(
        yolo_weights=config.yolo_weights,
        conf_threshold=config.vehicle_conf_threshold,
    )
    boundary_detector = TrackBoundaryDetector()
    violation_engine = ViolationEngine(
        required_consecutive_frames=config.required_consecutive_frames
    )
    incident_log = IncidentLog(log_path=log_path)

    annotated_writer = None
    annotated_video_path = None
    if write_annotated_video:
        annotated_video_path = os.path.join(config.output_dir, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_writer = cv2.VideoWriter(
            annotated_video_path, fourcc, fps / max(config.frame_skip, 1), (width, height)
        )

    overall_status = "GREEN"
    already_logged_streak = False  # avoid duplicate incidents for one continuous violation
    frame_idx = 0
    processed = 0

    limit = total_frames if config.max_frames is None else min(total_frames, config.max_frames)

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= limit:
            break

        if frame_idx % config.frame_skip != 0:
            frame_idx += 1
            continue

        timestamp_sec = frame_idx / fps

        vehicle = vehicle_detector.detect(frame)
        boundary = boundary_detector.detect(frame)

        wheel_points = None
        certainty = 0.0
        if vehicle is not None:
            wheel_points = estimate_wheel_points(vehicle["bbox"])
            certainty = wheel_certainty(vehicle["confidence"], vehicle["bbox"], frame.shape)

        analysis = violation_engine.analyze_frame(
            frame_number=frame_idx,
            timestamp_sec=timestamp_sec,
            vehicle=vehicle,
            boundary=boundary,
            wheel_points=wheel_points,
            wheel_certainty=certainty,
        )

        if analysis.status == "RED":
            overall_status = "RED"

        # Log exactly once per continuous violation streak (on the frame
        # where it first becomes CONFIRMED), then wait for a GREEN frame
        # before allowing a new incident to be logged.
        if analysis.confirmed and not already_logged_streak:
            already_logged_streak = True
            annotated = annotate_frame(
                frame, analysis.vehicle, analysis.boundary,
                analysis.wheel_points, analysis.wheels_beyond,
                analysis.status, analysis.confidence,
                analysis.frame_number, analysis.timestamp_sec,
                turn_label=config.turn_label,
                car_number=vehicle.get("car_number") if vehicle else None,
            )
            incident_num = len(incident_log) + 1
            evidence_path = save_evidence(annotated, evidence_dir, incident_num)

            replay_path = generate_replay_clip(
                source_video_path=video_path,
                incident_frame_number=analysis.frame_number,
                fps=fps,
                output_dir=replay_dir,
                incident_id=incident_num,
                seconds_before=config.replay_seconds_before,
                seconds_after=config.replay_seconds_after,
            )

            incident_log.add_incident(
                car_number=vehicle.get("car_number") if vehicle else None,
                timestamp_sec=analysis.timestamp_sec,
                frame_number=analysis.frame_number,
                turn=config.turn_label,
                status="VIOLATION",
                confidence=analysis.confidence,
                evidence_path=evidence_path,
                replay_path=replay_path,
            )
        elif not analysis.all_four_beyond:
            already_logged_streak = False

        if annotated_writer is not None:
            display_frame = frame
            if vehicle is not None and boundary is not None and wheel_points is not None:
                display_frame = annotate_frame(
                    frame, vehicle, boundary, wheel_points,
                    analysis.wheels_beyond or {}, analysis.status,
                    analysis.confidence, frame_idx, timestamp_sec,
                    turn_label=config.turn_label,
                    car_number=vehicle.get("car_number") if vehicle else None,
                )
            annotated_writer.write(display_frame)

        if progress_callback is not None:
            progress_callback(frame_idx, limit, analysis)

        processed += 1
        frame_idx += 1

    cap.release()
    if annotated_writer is not None:
        annotated_writer.release()

    duration_sec = total_frames / fps if fps else 0.0

    return PipelineResult(
        total_frames_processed=processed,
        fps=fps,
        duration_sec=duration_sec,
        incidents=[i.__dict__ for i in incident_log.incidents],
        overall_status=overall_status,
        vehicle_detection_method=vehicle_detector.method,
        annotated_video_path=annotated_video_path,
        log_path=log_path,
    )
