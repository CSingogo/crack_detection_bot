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
# This should be the first Streamlit command in your script
st.set_page_config(
    page_title="Road Defect Detector",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model and OCR Loading ---
# We cache these models to prevent reloading them every time the user interacts with the app
@st.cache_resource
def load_models():
    """Loads the YOLOv8 and EasyOCR models from disk."""
    yolo_model = YOLO('best.pt')
    ocr_reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if your hosting environment has a GPU
    return yolo_model, ocr_reader

# --- Helper Functions ---
def parse_gps_from_text(text):
    """Parses GPS coordinates from a string using regular expressions."""
    lat_match = re.search(r'(?:Lat|Latitude):\s*(-?\d+\.\d+)', text, re.IGNORECASE)
    lon_match = re.search(r'(?:Lon|Longitude):\s*(-?\d+\.\d+)', text, re.IGNORECASE)
    
    if lat_match and lon_match:
        lat = float(lat_match.group(1))
        lon = float(lon_match.group(1))
        return lat, lon
    return None, None

# --- Main Application UI and Logic ---
st.title("🚧 Road Defect Detection & Mapping")
st.markdown("""
    **Drag and drop a dashcam video below.** The app will:
    1.  Process the video frame by frame to detect road defects.
    2.  Extract GPS coordinates from the video frames.
    3.  Plot the defect locations on an interactive map.
""")

# Load models with a user-friendly spinner
with st.spinner('Loading AI models, this may take a moment...'):
    yolo_model, ocr_reader = load_models()

# File Uploader UI
uploaded_file = st.file_uploader(
    "Choose a video file to analyze",
    type=['mp4', 'mov', 'avi', 'mkv']
)

if uploaded_file is not None:
    # Use a temporary file to save the uploaded video for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    st.success(f"Video '{uploaded_file.name}' uploaded successfully. Starting analysis...")
    
    # --- UI Layout for Video Processing ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Video Analysis")
        stframe = st.empty() # Placeholder for the video player
    
    with col2:
        st.subheader("Detected Defect Map")
        map_placeholder = st.empty() # Placeholder for the map

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Analyzing video...")
    
    defect_locations = []
    
    # --- The Main Processing Loop ---
    # The YOLO model's predict function with stream=True is a generator
    for frame_idx, result in enumerate(yolo_model.predict(source=video_path, stream=True, device='cpu')):
        
        # Display the frame with bounding boxes drawn on it
        frame_with_boxes = result.plot()
        stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        # Update the progress bar based on the frame number
        progress_text = f"Analyzing video... Frame {frame_idx + 1}/{total_frames}"
        progress_bar.progress((frame_idx + 1) / total_frames, text=progress_text)
        
        # --- OCR Logic: run only if defects are detected ---
        if result.boxes:
            orig_frame = result.orig_img
            height, width, _ = orig_frame.shape
            
            # Crop the bottom-left corner of the frame to isolate GPS text
            crop = orig_frame[int(height*0.85):height, 0:int(width*0.3)]
            
            # Run OCR on the cropped image
            ocr_results = ocr_reader.readtext(crop, detail=0, paragraph=True)
            
            if ocr_results:
                full_text = " ".join(ocr_results)
                lat, lon = parse_gps_from_text(full_text)
                if lat and lon:
                    defect_locations.append({'latitude': lat, 'longitude': lon})

    video.release()
    progress_bar.progress(1.0, text="Analysis Complete!")
    time.sleep(2) # Pause to let the user see the completion message
    progress_bar.empty() # Remove the progress bar

    # --- Display Final Results ---
    st.header("✨ Analysis Results")
    if defect_locations:
        # Create and display the Folium map
        map_center = [defect_locations[0]['latitude'], defect_locations[0]['longitude']]
        
        # Use Google Maps tiles if API key is available, otherwise fallback to default
        try:
            GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
            tiles = f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_MAPS_API_KEY}"
            attr = "Google"
        except (FileNotFoundError, KeyError):
            st.warning("Google Maps API Key not found in secrets.toml. Using default OpenStreetMap.")
            tiles = "OpenStreetMap"
            attr = "OpenStreetMap"

        m = folium.Map(location=map_center, zoom_start=16, tiles=tiles, attr=attr)
        
        for loc in defect_locations:
            lat, lon = loc['latitude'], loc['longitude']
            google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"
            popup_html = f"""
            <b>Defect Location</b><br>
            Lat: {lat:.5f}, Lon: {lon:.5f}<br><br>
            <a href="{google_maps_url}" target="_blank">Open in Google Maps</a>
            """
            iframe = folium.IFrame(popup_html, width=220, height=100)
            popup = folium.Popup(iframe, max_width=220)
            
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                tooltip="Click for details",
                icon=folium.Icon(color='red', icon='exclamation-sign')
            ).add_to(m)
        
        # Display the map in its placeholder
        with map_placeholder:
            st_folium(m, use_container_width=True)
            
        st.subheader("List of Defect Coordinates")
        st.dataframe(pd.DataFrame(defect_locations), use_container_width=True)

    else:
        st.warning("No defects with valid GPS coordinates were found in the video.")
        map_placeholder.info("No data to display on the map.")

else:
    st.info("Awaiting video upload to begin analysis.")