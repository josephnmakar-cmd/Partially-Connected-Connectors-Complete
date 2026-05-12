# Seat Connector Inspection System - Project Overview

This project is a computer vision-based inspection system developed for a Master's Thesis in AI and Automation at University West. It is designed to perform real-time quality control of automotive seat connectors by analyzing gap width and the presence of a notch feature.

## Core Technologies
- **Language**: Python 3.7+
- **Computer Vision**: OpenCV (cv2)
- **Image Processing**: NumPy, Pillow (PIL)
- **GUI**: Tkinter
- **OS**: Windows (based on shell paths and .bat files)

## Architecture & Algorithms
The system follows a tracking-then-inspection pipeline:
1.  **Tracking**: Uses Template Matching (cv2.matchTemplate) to locate the connector body in the video feed.
2.  **Gap Analysis**: Measures the distance between connector halves using Sobel edge detection to identify vertical edges.
3.  **Notch Detection**: Scores the presence of a notch feature using Canny edge detection and pixel counting (texture score).
4.  **Logging**: Automatically saves defect images to the defect_logs/ directory for audit and analysis.

## Project Structure
- User_HMI.py: The primary entry point. Provides a Tkinter-based GUI for calibration and production monitoring.
- live_inspection.py: A headless version of the inspection system for automated environments.
- Calib_interface.py: A specialized component for interactive ROI (Region of Interest) calibration and threshold tuning.
- edge_detection.py: A standalone test script for refining edge detection and sensor fusion logic.
- AquirePhoto.py: Utilities for frame acquisition and camera initialization.

## Building and Running
### Prerequisites
Ensure you have Python 3.7+ installed.

### Installation
`ash
pip install -r requirements.txt
`

### Running the System
- **Main GUI Mode**: python User_HMI.py
- **Headless Mode**: python live_inspection.py
- **Calibration/Tuning**: python Calib_interface.py

## Development Conventions
- **Camera Configuration**: CAMERA_INDEX is typically set to 1 in source files. Adjust this in the code if your camera is on a different index.
- **ROI Calibration**: The system relies on initial manual ROI selection for the anchor (tracking), gap, and notch regions.
- **Thresholding**: Thresholds for MAX_ACCEPTABLE_GAP and NOTCH_THRESHOLD are defined as constants in the scripts and can be tuned via trackbars in calibration mode.
- **Data Storage**: Defect logs are stored in defect_logs/. Research data is stored in 	hesis_data/.

## Data Management
- defect_logs/: Images of failed connectors.
- 	hesis_data/: Dataset used for the Master's thesis, including full frames and cropped ROIs.
- .venv/: Local Python virtual environment.
