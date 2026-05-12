# Seat Connector Inspection System - Project Overview

This project is a computer vision and deep learning-based inspection system developed for a Master's Thesis in AI and Automation at University West. It is designed to perform real-time quality control of automotive seat connectors.

## Core Technologies
- **Language**: Python 3.7+
- **Deep Learning**: PyTorch (torch, torchvision)
- **Computer Vision**: OpenCV (cv2)
- **Image Processing**: NumPy, Pillow (PIL)
- **Visualization**: Matplotlib, Seaborn
- **GUI**: Tkinter

## Architecture & Algorithms
The system supports two inspection modes:

### 1. Traditional CV Pipeline
- **Tracking**: Template Matching (cv2.matchTemplate) for localization.
- **Gap Analysis**: Sobel edge detection for distance measurement.
- **Notch Detection**: Canny edge detection and texture scoring.

### 2. Deep Learning Pipeline
- **Model**: ResNet18 (Transfer Learning) with custom classification heads.
- **Training**: Stochastic Gradient Descent (SGD) with heavy data augmentation.
- **Evaluation**: Automated confusion matrix generation and classification reporting.

## Project Structure
- `User_HMI.py`: Primary production GUI.
- `live_inspection.py`: Headless production script.
- `CNN.py` / `Train_FullImage.py`: Neural network definition and training.
- `evaluate model.py`: Performance validation script.
- `full_frame_live_view.py`: AI-driven live inspection.
- `AquirePhoto.py`: Camera interface.

## Data Management
- `connector_dataset/` / `Full_ConnectorFrame_Dataset/`: Training and validation data.
- `defect_logs/` / `inspection_logs/`: Real-time audit logs for failed parts.
- `thesis_data/`: Raw research data and frame captures.

## Development Conventions
- **Camera Configuration**: `CAMERA_INDEX` is configurable (default 1).
- **Model Weights**: Saved as `.pth` files (e.g., `full_image_ai.pth`).
- **Input Resolution**: Neural network expects 224x224 input.

