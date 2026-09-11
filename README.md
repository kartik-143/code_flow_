# 🏁 AI Race Steward Assistant — Track Limits Detection

Built for **TrackShift 2026 — Theme 2: Track Limits Detection**.

An automated race-steward system that analyzes race footage frame-by-frame
and determines whether a car violated track limits — defined as **all four
wheels completely crossing the white track-boundary line** — with temporal
confirmation, an explainable confidence score, evidence capture, replay
generation, and a Streamlit dashboard.

## How it works

```
Race Video
   │
   ▼
Vehicle Detection (pretrained YOLO, auto-fallback to classical CV)
   │
   ▼
Track Boundary Detection (HSV white-mask → Canny → Hough lines → robust fit)
   │
   ▼
Wheel / Contact-Point Estimation (geometric, from vehicle bbox)
   │
   ▼
Per-wheel vs. Boundary Check  →  All 4 beyond line?
   │
   ▼
Temporal Confirmation (N consecutive frames, configurable)
   │
   ▼
Confirmed Violation
   │
   ├─► Explainable Confidence Score
   ├─► Annotated Evidence Frame (.jpg)
   ├─► ~3s Replay Clip (.mp4, cut from the real source video)
   └─► Incident Log entry (JSON + DataFrame)
   │
   ▼
Streamlit Steward Dashboard
```

## Project structure

```
ai-race-steward/
├── app.py                     # Streamlit dashboard
├── run_analysis.py            # CLI runner
├── requirements.txt
├── src/
│   ├── vehicle_detector.py        # YOLO (primary) + classical-CV fallback
│   ├── track_boundary_detector.py # White-line detection (HSV/Canny/Hough)
│   ├── wheel_estimator.py         # 4-point geometric wheel estimation
│   ├── violation_engine.py        # Decision logic + confidence scoring
│   ├── evidence.py                # Annotated evidence frame rendering
│   ├── replay.py                  # ~3s replay clip generation
│   ├── incident_log.py            # JSON + Pandas incident log
│   └── pipeline.py                # Orchestrates the full pipeline
├── tools/
│   └── make_test_video.py     # Generates a SYNTHETIC self-test clip
├── sample_videos/             # Synthetic test clip lives here
└── output/
    ├── evidence/               # incident_NNN.jpg
    ├── replays/                 # incident_NNN.mp4
    ├── logs/incident_log.json
    └── annotated_output.mp4    # Full video with live overlay
```

## Setup

```bash
pip install -r requirements.txt
```

> **Note on YOLO weights:** the first run with `ultralytics` installed will
> auto-download `yolov8n.pt` (needs internet). If that's not available in
> your environment, `VehicleDetector` automatically falls back to a
> classical-CV (background subtraction) vehicle detector — the pipeline
> still runs end-to-end, just with a different detection method (visible
> in the dashboard and in `result.vehicle_detection_method`).

## Run the dashboard

```bash
streamlit run app.py
```

Upload a race video (or pick the bundled synthetic sample), tune the
"consecutive frames to confirm" and frame-skip settings in the sidebar,
and click **Start Analysis**.

## Run from the command line

```bash
python run_analysis.py path/to/race_video.mp4 --turn "Turn 3" --frames-required 4
```

## About the bundled sample video

`sample_videos/synthetic_test_track.mp4` is a **synthetically generated**
clip (see `tools/make_test_video.py`) — a rendered car drifting across a
white boundary line — used only to self-test the pipeline in environments
without real race footage or internet access for YOLO weights. It proves
every stage of the pipeline (detection → boundary → wheels → decision →
evidence → replay → log) runs on genuine per-frame processing, with
**zero hardcoded outcomes**. Swap in real race footage for actual use —
the pipeline logic is identical either way.

## Design notes / engineering principles followed

- **No hardcoded results.** Every value in the incident log (timestamp,
  frame, confidence, car number, status) is computed from actual frame
  processing on the specific input video.
- **MVP-first.** Wheel positions are estimated geometrically from the
  vehicle bounding box rather than training a dedicated wheel detector,
  per the brief. This can be swapped for pose/segmentation-based
  estimation later without touching the decision or logging layers.
- **Temporal confirmation.** A configurable number of consecutive
  "all-four-wheels-beyond" frames is required before a violation is
  confirmed, filtering out single-frame detection jitter.
- **Explainable confidence.** Computed as a weighted blend of vehicle
  detection confidence, boundary-line detection confidence, wheel
  position certainty, temporal persistence, and geometric consistency —
  never a fixed number.
- **Graceful degradation.** If `ultralytics`/YOLO weights aren't
  available, the system automatically falls back to a real classical-CV
  detector rather than failing or faking results.
