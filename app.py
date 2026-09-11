"""
app.py

Streamlit dashboard for the AI Race Steward Assistant.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import time

import streamlit as st
import pandas as pd

from src.pipeline import run_pipeline, PipelineConfig

st.set_page_config(
    page_title="AI Race Steward — Track Limits Detection",
    page_icon="🏁",
    layout="wide",
)

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .status-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .status-green { background-color: #1e8a3c; color: white; }
    .status-red { background-color: #c62828; color: white; }
    .metric-card {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("TRACKGUARD-AI
")
st.caption("Automated Track Limits Detection — TrackShift 2026, Theme 2")

# ----------------------------------------------------------------------
# Sidebar: configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    turn_label = st.text_input("Track / Turn label", value="Turn 1")
    frames_required = st.slider(
        "Consecutive frames to confirm violation", min_value=1, max_value=15, value=4,
        help="Higher = fewer false positives, slower to confirm.",
    )
    frame_skip = st.slider(
        "Process every Nth frame", min_value=1, max_value=5, value=1,
        help="Increase to speed up analysis on long videos (trade-off: coarser timing).",
    )
    max_frames = st.number_input(
        "Max frames to process (0 = full video)", min_value=0, value=0, step=50
    )
    st.divider()
    st.caption(
        "Vehicle detection uses pretrained YOLO when available, "
        "and automatically falls back to classical CV background "
        "subtraction if YOLO / internet weights aren't available."
    )

# ----------------------------------------------------------------------
# Video input
# ----------------------------------------------------------------------
col_up1, col_up2 = st.columns([2, 1])
with col_up1:
    uploaded = st.file_uploader(
        "Upload race footage", type=["mp4", "mov", "avi", "mkv"]
    )
with col_up2:
    sample_dir = "sample_videos"
    sample_files = (
        [f for f in os.listdir(sample_dir) if f.lower().endswith((".mp4", ".avi", ".mov"))]
        if os.path.isdir(sample_dir) else []
    )
    chosen_sample = st.selectbox(
        "...or pick a sample video", options=["(none)"] + sample_files
    )

video_path = None
if uploaded is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tmp.write(uploaded.read())
    tmp.close()
    video_path = tmp.name
elif chosen_sample != "(none)":
    video_path = os.path.join(sample_dir, chosen_sample)

if video_path:
    st.video(video_path)

run_clicked = st.button("▶️ Start Analysis", type="primary", disabled=video_path is None)

# ----------------------------------------------------------------------
# Session state for results
# ----------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if run_clicked and video_path:
    progress_bar = st.progress(0, text="Starting analysis...")
    status_placeholder = st.empty()

    def progress_cb(frame_idx, total, analysis):
        pct = min(1.0, frame_idx / max(total, 1))
        status_text = (
            f"Frame {frame_idx}/{total} — status: {analysis.status} "
            f"— streak: {analysis.consecutive_count} — confidence: {analysis.confidence:.0%}"
        )
        progress_bar.progress(pct, text=status_text)

    config = PipelineConfig(
        required_consecutive_frames=frames_required,
        turn_label=turn_label,
        frame_skip=frame_skip,
        max_frames=(max_frames if max_frames > 0 else None),
    )

    with st.spinner("Running AI Race Steward analysis..."):
        result = run_pipeline(video_path, config=config, progress_callback=progress_cb)

    st.session_state.result = result
    progress_bar.progress(1.0, text="Analysis complete.")

# ----------------------------------------------------------------------
# Results display
# ----------------------------------------------------------------------
result = st.session_state.result

if result is not None:
    st.divider()

    status_class = "status-red" if result.overall_status == "RED" else "status-green"
    status_word = "VIOLATION DETECTED" if result.overall_status == "RED" else "CLEAN"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="status-badge {status_class}">{status_word}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Incidents", len(result.incidents))
    with c3:
        st.metric("Frames processed", result.total_frames_processed)
    with c4:
        st.metric("Detector", result.vehicle_detection_method.upper())

    tab_overview, tab_incidents, tab_log = st.tabs(
        ["📺 Analyzed Video", "🚨 Incidents & Evidence", "📋 Incident Log"]
    )

    with tab_overview:
        if result.annotated_video_path and os.path.exists(result.annotated_video_path):
            st.video(result.annotated_video_path)
        else:
            st.info("Annotated video not available.")

    with tab_incidents:
        if not result.incidents:
            st.success("No track-limit violations were confirmed in this footage.")
        for inc in result.incidents:
            with st.container(border=True):
                st.subheader(f"Incident #{inc['incident_id']} — {inc['status']}")
                ic1, ic2 = st.columns([1, 1])
                with ic1:
                    if inc["evidence"] and os.path.exists(inc["evidence"]):
                        st.image(inc["evidence"], caption="Evidence frame")
                with ic2:
                    st.write(f"**Timestamp:** {inc['timestamp']}")
                    st.write(f"**Frame:** {inc['frame']}")
                    st.write(f"**Turn:** {inc['turn']}")
                    st.write(f"**Confidence:** {inc['confidence']*100:.1f}%")
                    st.write(f"**Car number:** {inc['car_number'] or 'Not detected'}")
                    if inc["replay"] and os.path.exists(inc["replay"]):
                        st.write("**Replay:**")
                        st.video(inc["replay"])

    with tab_log:
        df = pd.DataFrame(result.incidents)
        st.dataframe(df, use_container_width=True)
        if not df.empty:
            st.download_button(
                "⬇️ Download incident log (JSON)",
                data=open(result.log_path, "rb").read(),
                file_name="incident_log.json",
                mime="application/json",
            )
else:
    st.info("Upload or select a video, configure settings in the sidebar, then click **Start Analysis**.")
