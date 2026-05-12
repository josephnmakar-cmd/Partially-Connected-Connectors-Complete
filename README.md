# Seat Connector Inspection System

A comprehensive computer vision and deep learning-based system for real-time quality inspection of automotive seat connectors. Developed as part of a Master's Thesis in AI and Automation at University West, this system ensures connectors are properly seated by analyzing physical features and utilizing neural network classification.

## Overview

The system employs two primary methodologies for inspection:
1.  **Classical Computer Vision**: Uses template matching, Sobel edge detection for gap measurement, and Canny-based texture scoring for notch detection.
2.  **Deep Learning (AI)**: Utilizes Convolutional Neural Networks (ResNet18-based) to classify the seating state directly from full frames or cropped Regions of Interest (ROIs).

## Key Features

- **Real-time Inspection**: High-speed video analysis with live visual feedback.
- **Dual Pipeline**: Traditional CV for precise measurement and AI for robust classification.
- **Interactive Calibration**: GUI-driven ROI selection and threshold tuning.
- **Automatic Logging**: Captures and archives images of defective connectors for audit trails.
- **Performance Evaluation**: Scripts for generating confusion matrices and classification reports.
- **Data Augmentation**: Training pipeline includes heavy augmentation (flips, rotations, color jitter) for robust AI models.

## Project Structure

### Core Application & GUI
- `User_HMI.py`: The main production interface. Integrated GUI for calibration and monitoring.
- `live_inspection.py`: Headless version for high-performance deployment.
- `Calib_interface.py`: Specialized tool for fine-tuning ROIs and edge detection parameters.

### AI & Deep Learning
- `CNN.py`: Definition and training logic for the neural network classifier.
- `Train_FullImage.py`: Script to train the model on the full frame dataset.
- `evaluate model.py`: Generates performance metrics, including confusion matrices and precision/recall reports.
- `ai_live_view.py`: Real-time AI-based inspection interface.
- `full_frame_live_view.py`: Live AI inference on full camera frames with logging.

### Utilities & Data Collection
- `collect_data.py` / `collect_full_data.py`: Tools for capturing and organizing images for dataset creation.
- `AquirePhoto.py`: Hardware abstraction for camera initialization and frame capture.
- `edge_detection.py`: Prototyping script for testing new filter kernels and thresholding logic.

## Requirements

### Hardware
- USB Industrial Camera or standard Webcam.
- Windows-compatible PC.

### Software (Python 3.7+)
- `opencv-python`: Core vision processing.
- `torch` & `torchvision`: Neural network training and inference.
- `scikit-learn`: Performance metrics and evaluation.
- `matplotlib` & `seaborn`: Visualization of results.
- `pillow`, `numpy`: Image handling and numerical operations.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/josephnmakar-cmd/Partially-Connected-Connectors-Complete.git
   cd Partially-Connected-Connectors-Complete
   ```

2. **Set up environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Traditional Inspection (CV Mode)
Run the HMI for interactive setup:
```bash
python User_HMI.py
```
*   **Calibration**: Click "START CALIBRATION" to define the Anchor (tracking), Gap, and Notch ROIs.
*   **Production**: Click "LOCK & RUN PRODUCTION" to start real-time monitoring.

### 2. Deep Learning Inspection (AI Mode)
Run the AI live viewer (requires a trained `full_image_ai.pth` model):
```bash
python full_frame_live_view.py
```

### 3. Model Training & Evaluation
To train a new model on your dataset:
```bash
python Train_FullImage.py
```
To evaluate model performance on the test set:
```bash
python "evaluate model.py"
```

## Data Management

- `connector_dataset/`: Image dataset for ROI-based classification.
- `Full_ConnectorFrame_Dataset/`: Dataset containing full-frame images.
- `defect_logs/`: Archive of failures detected during traditional inspection.
- `inspection_logs/`: Archive of images and predictions from the AI pipeline.

## Authors

**Joseph Makar** - Master Thesis in AI and Automation, University West.

## License

This project is proprietary research code developed for academic purposes at University West.
