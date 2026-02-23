import os
import cv2
import random
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# 1. SYSTEM CONFIGURATION
# Using relative paths since everything is in the same local folder
DATASET_DIR = "GUB-STFN-Fall-Dataset"
MODEL_PATH = "model_enhanced.keras"  # Make sure this file is right next to app.py
YOLO_MODEL = "yolov8n-pose.pt"

SEQ_LEN = 50
STRIDE = 10
CLASSES = ["NoFall", "Faint", "Slip", "Trip"]
EDGES = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

@st.cache_resource
def load_system_models():
    pose = YOLO(YOLO_MODEL) if os.path.exists(YOLO_MODEL) else YOLO("yolov8n-pose.pt")
    clf = load_model(MODEL_PATH)
    return pose, clf

pose_model, clf_model = load_system_models()

# 2. CORE LOGIC
def normalize_skeleton(skel):
    xy = skel[..., :2]
    conf = skel[..., 2:]
    hip_center = (xy[:, 11:12, :] + xy[:, 12:13, :]) / 2
    return np.concatenate([xy - hip_center, conf], axis=2)

def draw_overlay(frame, kp, conf):
    for i in range(len(kp)):
        if conf[i] > 0.5:
            cv2.circle(frame, (int(kp[i,0]), int(kp[i,1])), 4, (0, 255, 0), -1)
    for edge in EDGES:
        i, j = edge
        if conf[i] > 0.5 and conf[j] > 0.5:
            cv2.line(frame, (int(kp[i,0]), int(kp[i,1])), (int(kp[j,0]), int(kp[j,1])), (0, 255, 0), 2)
    return frame

def process_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_data = {'skeletons': [], 'heights': [], 'confs': []}
    temp_avi = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    writer = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = pose_model(frame, verbose=False)
        skel, h_val, avg_conf = np.zeros((17,3)), np.nan, 0.0
        if results and results[0].keypoints is not None:
            kp, conf = results[0].keypoints.xy.cpu().numpy(), results[0].keypoints.conf.cpu().numpy()
            if len(kp) > 0:
                idx = np.argmax(np.mean(conf, axis=1))
                p_kp, p_conf = kp[idx], conf[idx]
                skel = np.concatenate([p_kp, p_conf[:,None]], axis=1)
                h_val = ((skel[11,1] + skel[12,1]) / 2) - skel[0,1]
                avg_conf = np.mean(p_conf)
                frame = draw_overlay(frame, p_kp, p_conf)
        frames_data['skeletons'].append(skel)
        frames_data['heights'].append(h_val)
        frames_data['confs'].append(avg_conf)
        writer.write(frame)

    cap.release(); writer.release()
    final_mp4 = temp_avi.replace(".avi", ".mp4")

    # Force Video to 16:9 Landscape
    os.system(f"ffmpeg -y -i {temp_avi} -vf \"scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2\" -vcodec libx264 {final_mp4} > /dev/null 2>&1")

    return frames_data, final_mp4, fps

def check_alarm_conditions(heights, fps):
    standing_height = np.nanpercentile(heights, 95)
    lay_threshold = 0.40 * standing_height
    down_counter, recovery_buffer, alarm_triggered = 0, 0, False
    ALARM_LIMIT, RECOVERY_LIMIT = int(5 * fps), 15

    for h in heights:
        if not np.isnan(h):
            if h < lay_threshold:
                down_counter += 1
                recovery_buffer = 0
                if down_counter >= ALARM_LIMIT:
                    alarm_triggered = True
            else:
                if down_counter > 0:
                    recovery_buffer += 1
                    if recovery_buffer >= RECOVERY_LIMIT:
                        down_counter, recovery_buffer = 0, 0
    return ("ALARM" if alarm_triggered else "SAFE"), lay_threshold

# 3. UI LAYOUT
st.set_page_config(page_title="Fall Recognition System", layout="wide", page_icon="🦺")

# [UI FIX] AGGRESSIVE CSS OVERRIDE
st.markdown('''
    <style>
        /* Force title to be smaller and ignore Streamlit defaults */
        .stMarkdown h1 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            white-space: nowrap !important; /* Forces text to stay on one line */
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
        }
        /* Style for the subtitle H3 */
        .stMarkdown h3 {
            font-size: 1.8rem !important;
            margin-top: 5px !important;
            font-weight: 400 !important;
        }
        /* Style for Author text */
        .author-text {
            font-style: italic;
            font-size: 0.9rem;
            color: #4F8BF9;
            margin-top: -5px;
        }
        /* Reduce top padding */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
        }
        /* Sidebar Button Fix */
        .stButton>button {
            width: 100%;
            margin-top: 10px;
            border-radius: 5px;
        }
    </style>
''', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/falling-person.png", width=60)
    st.title("Control Panel")

    mode = st.radio("Input Source", ["📂 Sample Video", "📤 Upload Video"])
    if mode == "📂 Sample Video":
        cls = st.selectbox("Select Class", CLASSES)
        if st.button("🎲 Pick Random Video"):
            folder = os.path.join(DATASET_DIR, cls)
            files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
            if files: st.session_state.video = os.path.join(folder, random.choice(files))
    else:
        up = st.file_uploader("Upload MP4", type=["mp4"])
        if up:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t.write(up.read())
            st.session_state.video = t.name

    st.markdown('<hr style="margin-top: 15px; margin-bottom: 15px; border: 1px solid #f0f2f6;">', unsafe_allow_html=True)
    run_btn = st.button("🚀 START ANALYSIS", type="primary")

# HEADERS
st.markdown("# Human Fall Recognition & Classification System")
st.markdown("### Thesis Demonstration: Skeleton-Based Bi-LSTM Approach")
st.markdown('<p class="author-text">Developed by: Kazi Nur Ali</p>', unsafe_allow_html=True)
st.markdown('<hr style="margin-top: 15px; margin-bottom: 15px; border: 1px solid #f0f2f6;">', unsafe_allow_html=True)

if run_btn and "video" in st.session_state:
    with st.spinner("Analyzing Video Logic... (Please wait 3-5min as it loads the model and analyzes your video)"):
        data, overlay_video, video_fps = process_video_file(st.session_state.video)
        skeletons, heights = np.array(data['skeletons']), np.array(data['heights'])
        X_input = []
        for i in range(0, len(skeletons)-SEQ_LEN, STRIDE):
            X_input.append(normalize_skeleton(skeletons[i:i+SEQ_LEN]).reshape(SEQ_LEN, -1))
        preds = clf_model.predict(np.array(X_input), verbose=0)
        avg_probs = np.mean(preds, axis=0)
        final_class = CLASSES[np.argmax(avg_probs)]
        alarm_status, thresh_val = "SAFE", 0
        if final_class != "NoFall":
            alarm_status, thresh_val = check_alarm_conditions(heights, video_fps)

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("🎥 Skeleton Analysis Overlay")
        st.video(overlay_video)
    with c2:
        st.subheader("📊 System Status")
        if alarm_status == "ALARM":
            st.error("## 🚨 ALARM TRIGGERED\n**Subject Failed to Recover > 5s**")
            st.audio("https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg")
        elif final_class != "NoFall":
            st.warning("## ⚠️ FALL DETECTED\n**Subject Recovered within 5s**")
        else:
            st.success("## ✅ NORMAL ACTIVITY\n**No Threat Detected**")
        st.metric("Detected Class", final_class)
        st.metric("Confidence", f"{np.max(avg_probs)*100:.1f}%")
        st.bar_chart(pd.DataFrame({"Confidence": avg_probs}, index=CLASSES))

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### 📉 Post-Fall Height Analysis")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot(heights, label="Height Trace", color="#1f77b4")
        if final_class != "NoFall":
            ax.axhline(thresh_val, color="red", linestyle="--", label="Threshold")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    with g2:
        st.markdown("##### 🔍 Skeleton Tracking Quality")
        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        ax2.plot(data['confs'], color="green", label="Confidence")
        ax2.set_ylim(0, 1); ax2.axhline(0.5, color="orange", linestyle=":", label="Low Visibility")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
else:
    st.info("👈 Select a sample video from the sidebar to begin.")
