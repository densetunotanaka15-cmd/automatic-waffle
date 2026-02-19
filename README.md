# Visually Impaired Traffic Light Detector

## Overview
This project implements a real-time traffic light detection and active color classification system designed to assist visually impaired individuals. It uses a pre-trained YOLOv8 model for object detection to identify traffic lights in an image, and then applies a custom HSV color analysis method to determine the active color (red, yellow, or green) of each detected traffic light. The results are presented visually and announced via text-to-speech for enhanced accessibility.

## Features
- **YOLOv8 Detection**: Leverages the YOLOv8n model for accurate traffic light detection.
- **HSV Color Analysis**: Determines the active color of traffic lights (red, yellow, green, or none) using HSV color space segmentation.
- **Visual Annotation**: Displays bounding boxes and active color labels on the input image.
- **Text-to-Speech**: Announces detection summary for accessibility.
- **Streamlit Web Application**: Provides an easy-to-use web interface for uploading images and viewing results.

## How to Run Locally

### Prerequisites
Make sure you have Python 3.8+ installed.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/traffic-light-detector.git
cd traffic-light-detector
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
```

### 3. Install Dependencies
Install the required Python packages. The `yolov8n.pt` model weights will be downloaded automatically by `ultralytics` on first run.
```bash
pip install -r requirements.txt
# For text-to-speech, you might also need espeak on your system:
# sudo apt-get update && sudo apt-get install espeak
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
This will open the application in your web browser. You can then upload an image to test the detector.

## Deployment to Streamlit Cloud

To deploy this application to Streamlit Cloud, follow these steps:

### 1. Prepare your Repository
Ensure your GitHub repository contains the following files:
- `app.py`: The main Streamlit application script.
- `requirements.txt`: Lists all Python dependencies.
- `yolov8n.pt`: The YOLOv8 model weights (or let `ultralytics` download it at runtime, ensuring your `requirements.txt` is correct).

### 2. Create a Streamlit Cloud Account
If you don't have one, sign up at [streamlit.io/cloud](https://streamlit.io/cloud).

### 3. Deploy the App
- Go to your Streamlit Cloud dashboard.
- Click on the "New app" button.
- Select your GitHub repository, the branch, and the `app.py` file as the main file path.
- Click "Deploy!"

Streamlit Cloud will automatically install the dependencies from `requirements.txt` and launch your application. You can monitor the deployment process from the dashboard.

## Example Usage
Upload an image containing traffic lights. The application will display the original image, an annotated image with detected traffic lights and their active colors, and a text summary that is also spoken aloud.

---
