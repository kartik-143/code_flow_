"""
run_analysis.py

Simple CLI entry point to run the full pipeline on a video file.

Usage:
    python run_analysis.py path/to/video.mp4 [--turn "Turn 3"] [--frames-required 4]
"""
import argparse
import json
import sys

from src.pipeline import run_pipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="AI Race Steward — Track Limits Detection")
    parser.add_argument("video", help="Path to race video file")
    parser.add_argument("--turn", default="Turn 1", help="Track/turn label for logging")
    parser.add_argument(
        "--frames-required", type=int, default=4,
        help="Consecutive frames required to confirm a violation",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    config = PipelineConfig(
        required_consecutive_frames=args.frames_required,
        turn_label=args.turn,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )

    def progress(frame_idx, total, analysis):
        if frame_idx % 15 == 0 or analysis.status == "RED":
            print(
                f"[frame {frame_idx}/{total}] status={analysis.status} "
                f"streak={analysis.consecutive_count} conf={analysis.confidence:.2f}"
            )

    result = run_pipeline(args.video, config=config, progress_callback=progress)

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Vehicle detection method : {result.vehicle_detection_method}")
    print(f"Frames processed         : {result.total_frames_processed}")
    print(f"Overall status           : {result.overall_status}")
    print(f"Incidents confirmed      : {len(result.incidents)}")
    print(f"Annotated video          : {result.annotated_video_path}")
    print(f"Incident log             : {result.log_path}")
    print(json.dumps(result.incidents, indent=2))


if __name__ == "__main__":
    sys.exit(main())
