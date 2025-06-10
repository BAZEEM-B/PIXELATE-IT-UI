import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import re
import gc
import yt_dlp
import os

from ultralytics import YOLO

# --- Constants and Configuration ---
# ... (This part remains the same) ...
TEMP_DIR_UPLOADS = tempfile.mkdtemp(prefix="st_uploads_")
TEMP_DIR_DOWNLOADS = tempfile.mkdtemp(prefix="st_downloads_")
PROCESSED_VIDEOS_DIR = "processed_videos"
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

MODEL_OPTIONS = {
    "YOLOv8n (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolov8n.pt"),
    "YOLOv8n (OIV7 - 600 classes)": os.path.join("YOLO-MODELS", "yolov8n-oiv7.pt"),
    "YOLOv8m (OIV7 - Medium)": os.path.join("YOLO-MODELS", "yolov8m-oiv7.pt"),
    "YOLOv8l (OIV7 - Large)": os.path.join("YOLO-MODELS", "yolov8l-oiv7.pt"),
    "YOLOv8x (OIV7 - Extra Large)": os.path.join("YOLO-MODELS", "yolov8x-oiv7.pt"),
    "YOLOv11n (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11n.pt"),
    "YOLOv11m (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11m.pt"),
    "YOLOv11s (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11s.pt"),
    "YOLOv11x (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11x.pt"),
    "YOLOv11l (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11l.pt"),
    "YOLOv11n-seg (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11n-seg.pt"),
    "YOLOv11m-seg (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11m-seg.pt"),
    "YOLOv11s-seg (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11s.pt"),
    "YOLOv11x-seg (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11x-seg.pt"),
    "YOLOv11l-seg (COCO - 80 classes)": os.path.join("YOLO-MODELS", "yolo11l-seg.pt"),
}

# --- Core & UI Functions ---
# ... (All functions from the previous version remain the same up to `main_app`) ...
@st.cache_resource
def load_yolo_model(model_path):
    """Loads and caches the YOLO model from the specified path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model '{model_path}': {e}")
        return None

def get_target_class_ids(model, class_names_list):
    """Finds all class IDs for a given list of class names."""
    if not all([model, model.names, class_names_list]):
        return set(), class_names_list

    found_ids, not_found_names = set(), []
    name_to_id_map = {name.lower(): class_id for class_id, name in model.names.items()}

    for name in class_names_list:
        name_lower = name.strip().lower()
        if name_lower in name_to_id_map:
            found_ids.add(name_to_id_map[name_lower])
        else:
            not_found_names.append(name)
            
    return found_ids, not_found_names

def download_video_from_url(url, save_dir):
    """Downloads video from a URL using yt-dlp."""
    st.info(f"Initializing downloader for {url}...")
    try:
        info_dict = yt_dlp.YoutubeDL({'noplaylist': True, 'quiet': True}).extract_info(url, download=False)
        base_filename = sanitize_filename(f"{info_dict.get('title', 'video')}.mp4")
        video_save_path = os.path.join(save_dir, base_filename)

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': video_save_path,
            'noplaylist': True,
            'merge_output_format': 'mp4',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            st.info(f"Starting download of '{info_dict.get('title')}'...")
            ydl.download([url])

        return video_save_path if os.path.exists(video_save_path) else None
    
    except yt_dlp.utils.DownloadError as e:
        st.error(f"Download Error: {e}", icon="🚨")
        st.warning("This may be a private video or require login. Publicly available content works best.", icon="⚠️")
        return None
    except Exception as e:
        st.error(f"An unexpected downloader error occurred: {e}", icon="🚨")
        return None

def process_single_frame(frame, model, target_class_ids, conf_thresh, block_size):
    """Processes a single video frame to detect and anonymize specified classes."""
    if not target_class_ids: return frame
        
    results = model(frame, verbose=False) # Process one frame
    for res in results:
        apply_pixelation_to_frame(frame, res, target_class_ids, conf_thresh, block_size)
    return frame

def apply_pixelation_to_frame(frame, result, target_class_ids, conf_thresh, block_size):
    """Helper function to apply pixelation based on model results."""
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id in target_class_ids and float(box.conf[0]) >= conf_thresh:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 >= x2 or y1 >= y2: continue
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue
            
            roi_h, roi_w = roi.shape[:2]
            eff_pixel_w = max(1, roi_w // block_size)
            eff_pixel_h = max(1, roi_h // block_size)
            
            small_roi = cv2.resize(roi, (eff_pixel_w, eff_pixel_h), interpolation=cv2.INTER_LINEAR)
            pixelated_roi = cv2.resize(small_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            
            frame[y1:y2, x1:x2] = pixelated_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


def sanitize_filename(filename):
    """Cleans a string to be a valid filename."""
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    name = name.replace(" ", "_")
    name = re.sub(r'[^\w\-_.]', '', name)
    return f"{name}{ext}" if name else f"processed_video{ext}"


# --- UI Rendering Functions ---

def render_sidebar():
    """Renders the sidebar UI and returns all configuration settings."""
    st.sidebar.header("⚙️ Configuration")
    
    # 1. Model Selection
    selected_model_key = st.sidebar.selectbox("1. Select YOLO Model", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[selected_model_key]
    model = load_yolo_model(model_path)
    
    if any(tag in selected_model_key for tag in ["Large", "Extra Large"]):
        st.sidebar.warning("Large models are slower and use more memory. A lower processing resolution and batch processing are recommended.", icon="⚠️")

    # 2. Class Selection
    target_class_ids = set()
    if model:
        st.sidebar.success(f"Model '{os.path.basename(model_path)}' loaded.")
        default_classes = "Woman, Girl" if "OIV7" in selected_model_key else "person"
        class_names_input = st.sidebar.text_input(f"2. Classes to detect (comma-separated):", default_classes)
        
        if class_names_input:
            class_list = [name.strip() for name in class_names_input.split(',') if name.strip()]
            if class_list:
                target_class_ids, not_found = get_target_class_ids(model, class_list)
                if target_class_ids:
                    found_names = [model.names[id] for id in target_class_ids]
                    st.sidebar.info(f"✅ Targeting: **{', '.join(found_names)}**")
                if not_found:
                    st.sidebar.error(f"⚠️ Not Found: **{', '.join(not_found)}**")
    else:
        st.sidebar.error("Model failed to load. Cannot proceed.")

    # 3. Processing Parameters
    conf_thresh = st.sidebar.slider("3. Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    block_size = st.sidebar.slider("4. Pixelation Block Size", 5, 50, 15, 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Performance")
    resolution = st.sidebar.selectbox("Processing Resolution", ["Original", "720p", "540p", "480p"], index=3)
    
    # NEW: Option to choose processing mode
    processing_mode = st.sidebar.selectbox(
        "Processing Mode", 
        ["Serial (Stable)", "Batch (Fast)"], 
        index=1,
        help="'Batch' mode is faster but uses more RAM. 'Serial' is slower but more stable with large videos or models."
    )
    batch_size = 32 if processing_mode == "Batch (Fast)" else 1

    return model, target_class_ids, conf_thresh, block_size, resolution, batch_size

def render_main_content():
    """Renders the main page UI for video input and returns the path to the video."""
    st.header("🎞️ Video Input")
    input_method = st.radio("Choose input method:", ("Upload File", "Enter URL"))
    
    video_path = None
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_file:
            video_path = os.path.join(TEMP_DIR_UPLOADS, uploaded_file.name)
            with open(video_path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.video(video_path)
            
    elif input_method == "Enter URL":
        url = st.text_input("Enter video URL (YouTube, Crunchyroll, etc.)")
        if url and st.button("Download and Preview"):
            with st.spinner("Downloading video..."):
                video_path = download_video_from_url(url, TEMP_DIR_DOWNLOADS)
            if video_path: st.video(video_path)
                
    return video_path


# --- Main Application Logic ---

def main_app():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="Video Anonymizer AI")
    st.title("🚀 High-Speed AI Video Anonymizer")
    st.markdown("Pixelate objects in videos using YOLOv8. Choose 'Batch' mode in the sidebar for faster processing.")
    st.markdown("---")

    model, target_ids, conf, block, res, batch_size = render_sidebar()
    input_path = render_main_content()

    st.markdown("---")
    st.header("⚙️ Processing Engine")

    can_process = input_path and os.path.exists(input_path) and model and bool(target_ids)
    if st.button("Start Video Processing", disabled=not can_process):
        run_video_processing_batched(input_path, model, target_ids, conf, block, res, batch_size)
        if input_path and (TEMP_DIR_UPLOADS in input_path or TEMP_DIR_DOWNLOADS in input_path):
            try: os.remove(input_path)
            except OSError as e: st.warning(f"Could not remove temporary file: {e}")
        gc.collect()

# NEW: Main processing function now handles batching
def run_video_processing_batched(video_path, model, target_ids, conf, block, resolution, batch_size):
    """Handles the video processing pipeline with optional batching."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return

    # 1. Setup video properties
    w_orig, h_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    h_target = int(resolution.replace('p', '')) if 'p' in resolution else h_orig
    w_out, h_out = (int(h_target * (w_orig/h_orig)), h_target) if h_target < h_orig else (w_orig, h_orig)
    w_out += w_out % 2 # Ensure even width

    # 2. Setup video writer
    sanitized_base, _ = os.path.splitext(sanitize_filename(f"processed_{os.path.basename(video_path)}"))
    out_path = os.path.join(PROCESSED_VIDEOS_DIR, f"{sanitized_base}_{h_out}p.mp4")
    # Use 'mp4v' as a more compatible FourCC for .mp4 files
    # Use 'mp4v' as a more compatible FourCC for .mp4 files
    out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_out, h_out))
    if not out_writer.isOpened():
        st.error("FATAL: Could not open video writer. Ensure ffmpeg is installed.")
        cap.release()
        return

    # 3. Main processing loop
    progress_bar, status_text = st.progress(0.0), st.empty()
    frames_batch, frames_processed = [], 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if (w_out, h_out) != (w_orig, h_orig):
            frame = cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_AREA)
        frames_batch.append(frame)

        if len(frames_batch) == batch_size:
            # Process the current batch
            results = model(frames_batch, verbose=False) # Single call for the whole batch
            for i, res in enumerate(results):
                processed_frame = apply_pixelation_to_frame(frames_batch[i], res, target_ids, conf, block)
                out_writer.write(processed_frame)
            
            frames_processed += len(frames_batch)
            frames_batch = [] # Clear the batch

            # Update progress
            progress = frames_processed / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing: {frames_processed}/{total_frames} frames ({progress*100:.1f}%)")

    # Process any remaining frames in the last batch
    if frames_batch:
        results = model(frames_batch, verbose=False)
        for i, res in enumerate(results):
            processed_frame = apply_pixelation_to_frame(frames_batch[i], res, target_ids, conf, block)
            out_writer.write(processed_frame)
        frames_processed += len(frames_batch)

    # 4. Finalize
    cap.release()
    out_writer.release()
    progress_bar.empty()
    status_text.success(f"✅ Video processing complete! Frames processed: {frames_processed}")
    
    with open(out_path, "rb") as f:
        st.download_button("⬇️ Download Processed Video", data=f, file_name=os.path.basename(out_path), mime="video/mp4")
    st.video(out_path)

if __name__ == "__main__":
    main_app()