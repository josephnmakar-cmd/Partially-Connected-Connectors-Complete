# Seat Connector Inspection System

A computer vision-based system for real-time inspection of seat connectors in automotive manufacturing. This project uses OpenCV and Python to automatically detect whether seat connectors are properly seated by analyzing the gap width between connector halves and the presence of a notch feature.

## Features

- **Real-time Video Inspection**: Live camera feed with automatic connector tracking
- **Graphical User Interface**: Tkinter-based interface for calibration and monitoring
- **Automatic Defect Detection**: Analyzes gap width and notch texture to determine seating status
- **Defect Logging**: Saves images of failed connectors for analysis
- **Calibration Mode**: Interactive ROI selection for anchor, gap, and notch regions
- **Edge Detection Algorithms**: Uses Sobel and Canny edge detection for precise measurements

## Project Structure

- `User_HMI.py`: Main GUI application for calibration and production inspection
- `live_inspection.py`: Headless inspection script with automatic logging
- `edge_detection.py`: Basic edge detection implementation
- `AquirePhoto.py`: Photo acquisition utilities
- `Calib_interface.py`: Calibration interface components
- `settings.json`: VS Code workspace settings

## Requirements

- Python 3.7+
- OpenCV
- Pillow (PIL)
- NumPy
- Tkinter (usually included with Python)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/seat-connector-inspection.git
   cd seat-connector-inspection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### GUI Mode (Recommended)
Run the main inspection interface:
```bash
python User_HMI.py
```

1. Click "START CALIBRATION" to enter calibration mode
2. Select the anchor region (connector body)
3. Select the gap region
4. Select the notch region
5. Adjust threshold values if needed
6. Click "LOCK & RUN PRODUCTION" to start inspection

### Headless Mode
For automated inspection without GUI:
```bash
python live_inspection.py
```

### Basic Edge Detection
For testing edge detection algorithms:
```bash
python edge_detection.py
```

## Algorithm Overview

The system uses template matching to track connectors in the video feed, then analyzes two key features:

1. **Gap Analysis**: Measures the distance between connector edges using Sobel edge detection
2. **Notch Detection**: Counts edge pixels in the notch region using Canny edge detection

Connectors are classified as:
- **PASS**: Properly seated (gap within limits, notch present)
- **FAIL - GAP ERROR**: Gap too wide
- **FAIL - PARTIAL**: Notch partially present
- **FAIL - DISCONNECTED**: Notch missing

## Configuration

Adjust parameters in the code:
- `MAX_ACCEPTABLE_GAP`: Maximum allowed gap in pixels
- `NOTCH_THRESHOLD`: Minimum edge pixels for notch detection
- Camera settings in `CAMERA_INDEX`

## Data Folders

- `defect_logs/`: Saved images of failed inspections
- `thesis_data/full_frames/`: Full frame captures
- `thesis_data/roi_crops/`: Cropped regions of interest

## License

This project is part of a Master's thesis at University West.

## Author

[Your Name] - University West, Master Thesis in AI and Automation