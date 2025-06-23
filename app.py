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

# --- Page Configuration ---
st.set_page_config(
    page_title="Road Defect Detector",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model and OCR Loading ---
@st.cache_resource
def load_models():
    """Loads the YOLOv8 and EasyOCR models from disk."""
    yolo_model = YOLO('best.pt')
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return yolo_model, ocr_reader

# --- Helper Functions ---
def parse_gps_from_text(text):
    text = text.replace(" ", "")
    lat, lon = None, None
    lat_match = re.search(r'([NS])(\d+\.\d+)', text, re.IGNORECASE)
    lon_match = re.search(r'([EW])(\d+\.\d+)', text, re.IGNORECASE)
    if lat_match:
        direction, value = lat_match.groups()
        lat = float(value) if direction.upper() == 'N' else -float(value)
    if lon_match:
        direction, value = lon_match.groups()
        lon = float(value) if direction.upper() == 'E' else -float(value)
    return lat, lon

# --- NEW: Session State Management ---
# Initialize state variables if they don't exist
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# Function to reset the state when a new file is uploaded
def reset_analysis():
    st.session_state.analysis_complete = False
    st.session_state.results_df = pd.DataFrame()

# --- Main Application UI ---
st.title("🚧 Road Defect Detection & Mapping")
st.markdown("""
    **Drag and drop a dashcam video below.** The app will analyze it once and display the results.
    You can then interact with the results without re-running the analysis.
""")

with st.spinner('Loading AI models, this may take a moment...'):
    yolo_model, ocr_reader = load_models()

# NEW: The on_change callback will run our reset_analysis function
uploaded_file = st.file_uploader(
    "Choose a video file to analyze",
    type=['mp4', 'mov', 'avi', 'mkv'],
    on_change=reset_analysis 
)

# --- Main Logic: Run analysis ONLY if a file is present and not yet analyzed ---
if uploaded_file is not None and not st.session_state.analysis_complete:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    st.success(f"Video '{uploaded_file.name}' uploaded. Starting analysis...")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Video Analysis")
        stframe = st.empty()
    with col2:
        st.subheader("Detected Defect Map")
        map_placeholder = st.empty() # We'll fill this in after the loop

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Analyzing video...")
    
    defect_locations = []

    for frame_idx, result in enumerate(yolo_model.predict(source=video_path, stream=True, device='cpu')):
        stframe.image(result.plot(), channels="BGR", use_container_width=True)
        progress_text = f"Analyzing video... Frame {frame_idx + 1}/{total_frames}"
        progress_bar.progress((frame_idx + 1) / total_frames, text=progress_text)
        
        if result.boxes:
            orig_frame = result.orig_img
            height, width, _ = orig_frame.shape
            crop = orig_frame[int(height * 0.85):height, 0:int(width * 0.50)]
            ocr_results = ocr_reader.readtext(crop, detail=0, paragraph=True)
            if ocr_results:
                full_text = " ".join(ocr_results)
                lat, lon = parse_gps_from_text(full_text)
                if lat and lon:
                    defect_locations.append({'latitude': lat, 'longitude': lon})

    video.release()
    progress_bar.progress(1.0, text="Analysis Complete!")
    time.sleep(2)
    progress_bar.empty()

    # --- NEW: Save results to session state and mark analysis as complete ---
    if defect_locations:
        st.session_state.results_df = pd.DataFrame(defect_locations).drop_duplicates().reset_index(drop=True)
    st.session_state.analysis_complete = True
    st.rerun() # Rerun the script one last time to display the final results cleanly

# --- Display Final Results (This part runs on every interaction) ---
if st.session_state.analysis_complete:
    st.header("Analysis Results")
    df_locations = st.session_state.results_df

    if not df_locations.empty:
        st.success(f"Found and mapped {len(df_locations)} unique defect locations!")

        map_center = [df_locations['latitude'].mean(), df_locations['longitude'].mean()]
        
        try:
            GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
            tiles = f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_MAPS_API_KEY}"
            attr = "Google"
        except (FileNotFoundError, KeyError):
            st.warning("Google Maps API Key not found. Using default OpenStreetMap.")
            tiles = "OpenStreetMap"
            attr = "OpenStreetMap"

        m = folium.Map(location=map_center, zoom_start=16, tiles=tiles, attr=attr)
        
        for index, loc in df_locations.iterrows():
            lat, lon = loc['latitude'], loc['longitude']
            google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"
            popup_html = f"""<b>Defect #{index + 1}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br><br><a href="{google_maps_url}" target="_blank">Open in Google Maps</a>"""
            iframe = folium.IFrame(popup_html, width=220, height=120)
            popup = folium.Popup(iframe, max_width=220)
            
            folium.Marker(
                location=[lat, lon], popup=popup, tooltip="Click for details",
                icon=folium.Icon(color='red', icon='exclamation-sign')
            ).add_to(m)
        
        # Display map and dataframe
        st_folium(m, use_container_width=True, height=500)
        st.subheader("List of Defect Coordinates")
        st.dataframe(df_locations, use_container_width=True)

    else:
        st.warning("Analysis complete, but no defects with valid GPS coordinates were found.")
else:
    # This message shows only when the app first loads
    st.info("Awaiting video upload to begin analysis.")