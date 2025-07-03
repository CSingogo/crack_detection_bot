import streamlit as st
import pandas as pd
import cv2
import easyocr
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
import re
import tempfile
import time
import torch

# --- Page Configuration ---
st.set_page_config(
    page_title="Road Defect Detector",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model and OCR Loading (using cache) ---
@st.cache_resource
def load_models():
    """Loads the YOLO model and OCR reader."""
    yolo_model = YOLO('best.pt')
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return yolo_model, ocr_reader

# --- Helper Functions ---
def parse_gps_from_text(text):
    """Extracts GPS coordinates from a string."""
    text = text.replace(" ", "")
    lat, lon = None, None
    lat_match = re.search(r'([NS])(\d{1,2}\.\d+)', text, re.IGNORECASE)
    lon_match = re.search(r'([EW])(\d{1,3}\.\d+)', text, re.IGNORECASE)
    if lat_match:
        direction, value = lat_match.groups()
        lat = float(value) if direction.upper() == 'N' else -float(value)
    if lon_match:
        direction, value = lon_match.groups()
        lon = float(value) if direction.upper() == 'E' else -float(value)
    return lat, lon

# --- Session State Management ---
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

def reset_analysis():
    """Resets the session state when a new file is uploaded."""
    st.session_state.analysis_complete = False
    st.session_state.results_df = pd.DataFrame()

# --- Main Application UI ---
st.title("🚧 Road Defect Detection & Mapping")
st.markdown("""
    **Drag and drop a dashcam video below.** The app will analyze it frame-by-frame to detect and map road defects.
""")

with st.spinner('Loading AI models, this may take a moment...'):
    yolo_model, ocr_reader = load_models()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.info(f"Using **{device.upper()}** for processing.")

uploaded_file = st.file_uploader(
    "Choose a video file to analyze",
    type=['mp4', 'mov', 'avi', 'mkv'],
    on_change=reset_analysis
)

# --- Main Logic: Video Processing ---
if uploaded_file is not None and not st.session_state.analysis_complete:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    st.success(f"Video '{uploaded_file.name}' uploaded. Starting analysis...")

    # Set up layout for live preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Video Analysis Preview")
        stframe = st.empty()
    with col2:
        st.subheader("Detected Defect Map")
        # Display a placeholder map directly.
        st_folium(folium.Map(location=[20, 0], zoom_start=2), use_container_width=True)

    # <<< FIX #2: MOVE PROGRESS BAR CREATION HERE >>>
    # Create the progress bar placeholder right under the columns.
    progress_bar = st.progress(0, text="Initializing analysis...")

    # Initialize video capture and parameters
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    FRAME_STRIDE = int(fps / 2) if fps > 2 else 1
    OCR_INTERVAL = int(fps * 2) if fps > 0.5 else 1

    st.sidebar.info(f"Video FPS: {fps:.2f}")
    st.sidebar.info(f"Analyzing 1 frame every {FRAME_STRIDE} frames.")
    st.sidebar.info(f"Updating GPS every {OCR_INTERVAL} frames.")

    defect_locations = []
    last_known_gps = None
    frame_idx = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_idx % OCR_INTERVAL == 0:
            height, width, _ = frame.shape
            crop = frame[int(height * 0.85):height, 0:int(width * 0.50)]
            ocr_results = ocr_reader.readtext(crop, detail=0, paragraph=True)
            if ocr_results:
                lat, lon = parse_gps_from_text(" ".join(ocr_results))
                if lat and lon:
                    last_known_gps = (lat, lon)

        if frame_idx % FRAME_STRIDE == 0:
            results = yolo_model.predict(source=frame, device=device, verbose=False)
            result = results[0]

            progress_text = f"Analyzing... Frame {frame_idx}/{total_frames} | Last GPS: {last_known_gps}"
            progress_bar.progress(frame_idx / total_frames, text=progress_text)

            if result.boxes and last_known_gps:
                annotated_frame = result.plot()
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)
                defect_locations.append({'latitude': last_known_gps[0], 'longitude': last_known_gps[1]})

        frame_idx += 1

    video.release()
    progress_bar.progress(1.0, text="Analysis Complete!")
    time.sleep(2)
    progress_bar.empty()

    if defect_locations:
        st.session_state.results_df = pd.DataFrame(defect_locations).drop_duplicates().reset_index(drop=True)

    st.session_state.analysis_complete = True
    st.rerun()

# --- Display Final Results (This block runs AFTER analysis is complete) ---
if st.session_state.analysis_complete:
    st.header("Analysis Results")
    df_locations = st.session_state.results_df

    if not df_locations.empty:
        st.sidebar.header("Filter Results")
        max_defects = len(df_locations)

        # <<< FIX #1: CONDITIONAL SLIDER >>>
        # Only show the slider if there is more than one defect to filter.
        if max_defects > 1:
            selected_range = st.sidebar.slider(
                "Select range of defects to display:",
                min_value=1,
                max_value=max_defects,
                value=(1, max_defects)
            )
            start_idx, end_idx = selected_range[0] - 1, selected_range[1]
            filtered_df = df_locations.iloc[start_idx:end_idx]
            st.success(f"Displaying {len(filtered_df)} of {max_defects} total mapped defects.")
        else:
            # If there's only 1 defect, no slider is needed. Just use the full dataframe.
            filtered_df = df_locations
            st.success(f"Found {max_defects} total mapped defect.")


        if not filtered_df.empty:
            map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
            m = folium.Map(location=map_center, zoom_start=16, tiles="OpenStreetMap")

            for index, loc in filtered_df.iterrows():
                lat, lon = loc['latitude'], loc['longitude']
                # Use the original index from df_locations for consistent numbering
                original_index = loc.name 
                google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"
                popup_html = f"<b>Defect #{original_index + 1}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br><br><a href='{google_maps_url}' target='_blank'>Open in Google Maps</a>"
                iframe = folium.IFrame(popup_html, width=220, height=120)
                popup = folium.Popup(iframe, max_width=220)
                folium.Marker(
                    location=[lat, lon], popup=popup, tooltip=f"Defect #{original_index+1}",
                    icon=folium.Icon(color='red', icon='exclamation-sign')
                ).add_to(m)

            res_col1, res_col2 = st.columns([2,1])
            with res_col1:
                st.subheader("Final Defect Map")
                st_folium(m, use_container_width=True, height=500)
            with res_col2:
                st.subheader("List of Defect Coordinates")
                st.dataframe(filtered_df, use_container_width=True, height=500)
        else:
             st.warning("No defects in the selected range.")

    else:
        st.warning("Analysis complete, but no defects with valid GPS coordinates were found.")

elif not uploaded_file:
    st.info("Awaiting video upload to begin analysis.")