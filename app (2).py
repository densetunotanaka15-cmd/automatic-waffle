
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import pyttsx3
from collections import Counter

# --- 1. Initialize text-to-speech engine ---
@st.cache_resource
def get_tts_engine():
    engine = pyttsx3.init()
    # Optional: Adjust rate and volume
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)
    return engine

engine = get_tts_engine()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# --- 2. Define HSV color ranges and color_threshold ---
# These are reused from the notebook's successful steps.
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

lower_green = np.array([50, 100, 100])
upper_green = np.array([80, 255, 255])

color_threshold = 20 # Threshold for considering a color active (sum of non-zero pixels in the ROI)

# --- 3. Function to determine active color using HSV ---
def determine_active_color_hsv(cropped_image, color_threshold_val):
    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        return 'none'

    hsv_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    mask_red1_cropped = cv2.inRange(hsv_cropped_image, lower_red1, upper_red1)
    mask_red2_cropped = cv2.inRange(hsv_cropped_image, lower_red2, upper_red2)
    mask_yellow_cropped = cv2.inRange(hsv_cropped_image, lower_yellow, upper_yellow)
    mask_green_cropped = cv2.inRange(hsv_cropped_image, lower_green, upper_green)

    mask_red_cropped = cv2.add(mask_red1_cropped, mask_red2_cropped)

    red_intensity = cv2.countNonZero(mask_red_cropped)
    yellow_intensity = cv2.countNonZero(mask_yellow_cropped)
    green_intensity = cv2.countNonZero(mask_green_cropped)

    active_color = 'none'
    max_intensity = color_threshold_val

    # Prioritize red > yellow > green for activation if intensities are similar or within a small range
    # This is a heuristic and can be adjusted.
    if red_intensity > max_intensity:
        max_intensity = red_intensity
        active_color = 'red'

    if yellow_intensity > max_intensity:
        # Consider yellow if it's significantly higher or if red wasn't strongly detected
        if active_color == 'none' or yellow_intensity > red_intensity * 1.2: # Yellow needs to be 20% more intense than red to override
            max_intensity = yellow_intensity
            active_color = 'yellow'

    if green_intensity > max_intensity:
        # Consider green if it's significantly higher or if red/yellow weren't strongly detected
        if active_color == 'none' or (green_intensity > red_intensity * 1.2 and green_intensity > yellow_intensity * 1.2): # Green needs to be 20% more intense than others
            max_intensity = green_intensity
            active_color = 'green'

    return active_color

# --- 4. Load the pre-trained YOLOv8 model ---
@st.cache_resource # Cache the model to avoid reloading on each rerun
def load_yolo_model():
    return YOLO('yolov8n.pt')

model = load_yolo_model()

# --- 5. Streamlit application interface ---
st.set_page_config(page_title="Visually Impaired Traffic Light Detector", layout="wide")
st.title("🚦 Visually Impaired Traffic Light Detector")
st.write("Upload an image to detect traffic lights and determine their active colors. "
         "The results will be displayed visually and announced via text-to-speech.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image_bytes = uploaded_file.getvalue()
    pil_image = Image.open(io.BytesIO(image_bytes))
    image_rgb = np.array(pil_image) # Convert to numpy array (RGB)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV

    st.subheader("Original Image")
    st.image(image_rgb, caption='Original Image', use_column_width=True)

    # Run YOLOv8 prediction
    st.subheader("Processing Image...")
    results = model.predict(source=image_bgr, conf=0.25, verbose=False)

    # Extract bounding boxes and class IDs, then filter for traffic lights
    detected_objects_yolo = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            detected_objects_yolo.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': cls,
                'class_name': model.names[cls]
            })

    # Correctly find the class ID for 'traffic light'
    traffic_light_class_id = None
    for class_id, class_name in model.names.items():
        if class_name == 'traffic light':
            traffic_light_class_id = class_id
            break

    traffic_lights_yolo = []
    if traffic_light_class_id is not None:
        for obj in detected_objects_yolo:
            if obj['class_id'] == traffic_light_class_id:
                traffic_lights_yolo.append(obj)

    st.info(f"YOLOv8 detected {len(traffic_lights_yolo)} potential traffic lights.")

    # Annotate image and determine active colors
    annotated_image_bgr = image_bgr.copy()

    yolo_final_traffic_lights = []

    # Define colors for drawing (BGR format)
    color_map = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'none': (255, 255, 255)
    }

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    for tl_yolo_info in traffic_lights_yolo:
        x1, y1, x2, y2 = map(int, tl_yolo_info['bbox'])

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_bgr.shape[1], x2)
        y2 = min(image_bgr.shape[0], y2)

        # Crop the traffic light region from the original image
        cropped_image = image_bgr[y1:y2, x1:x2]

        # Determine active color using the helper function
        active_color = determine_active_color_hsv(cropped_image, color_threshold)

        # Store results
        yolo_final_traffic_lights.append({'bbox': tl_yolo_info['bbox'], 'active_color': active_color})

        # Draw rectangle and text on the annotated image
        draw_color = color_map.get(active_color, color_map['none'])

        cv2.rectangle(annotated_image_bgr, (x1, y1), (x2, y2), draw_color, 2)

        text_size = cv2.getTextSize(active_color, font, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y2 + text_size[1] + 5

        cv2.putText(annotated_image_bgr, active_color, (text_x, text_y), font, font_scale, draw_color, font_thickness, cv2.LINE_AA)

    # Display annotated image
    st.subheader("Detection Results")
    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, caption='Annotated Traffic Lights', use_column_width=True)

    # Summarize results
    if yolo_final_traffic_lights:
        color_counts = Counter([tl['active_color'] for tl in yolo_final_traffic_lights])
        total_detected = len(yolo_final_traffic_lights)

        summary_parts = [f"Total traffic lights detected: {total_detected}."]
        if color_counts['red'] > 0: summary_parts.append(f"Red: {color_counts['red']}")
        if color_counts['yellow'] > 0: summary_parts.append(f"Yellow: {color_counts['yellow']}")
        if color_counts['green'] > 0: summary_parts.append(f"Green: {color_counts['green']}")
        if color_counts['none'] > 0: summary_parts.append(f"None: {color_counts['none']}")

        summary_string = " ".join(summary_parts)
        st.success(summary_string)
        speak_text(summary_string)
    else:
        st.warning("No traffic lights were detected in the image.")
        speak_text("No traffic lights were detected in the image.")

else:
    st.info("Please upload an image to get started!")

