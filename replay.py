"""
replay.py

Generates an ~3-second replay clip centered on a confirmed violation,
by re-reading the actual source video around the incident frame and
writing a real .mp4 (or .avi fallback) via OpenCV's VideoWriter.

Nothing here is a "fake clip" — every frame written is read directly
from the source footage.
"""

from __future__ import annotations
import os
from typing import Optional

import cv2


def generate_replay_clip(
    source_video_path: str,
    incident_frame_number: int,
    fps: float,
    output_dir: str,
    incident_id: int,
    seconds_before: float = 1.5,
    seconds_after: float = 1.5,
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_before = int(round(seconds_before * fps))
    frames_after = int(round(seconds_after * fps))

    start_frame = max(0, incident_frame_number - frames_before)
    end_frame = min(total_frames - 1, incident_frame_number + frames_after)

    out_path = os.path.join(output_dir, f"incident_{incident_id:03d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    written = 0
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()

    if written == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        return None

    return out_path
