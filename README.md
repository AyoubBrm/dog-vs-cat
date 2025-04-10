# Real-Time Cat/Dog Classifier 🐱🐶

A lightweight Python script that classifies cats and dogs in real-time using a fine-tuned YOLOv11 model and OpenCV.

## Features
- Real-time classification from webcam feed
- Confidence thresholding (0.8) to filter uncertain predictions
- Minimal dependencies (YOLOv11, OpenCV, NumPy)
- Clean visualization with on-screen labels

## Requirements
```bash
pip install ultralytics opencv-python numpy
```
## Usage
```bash
python run.py
```
## Controls:
Press q to quit the application

## How It Works:
1- Captures frames from your default webcam (index 0)

2- Processes each frame through your fine-tuned YOLOv11 model

3-Displays predictions with these rules:

* Shows class name if confidence > 80%

* Shows "none" if confidence < 80%

* Displays the annotated frame in real-time

## Customization:
* To change model path: Modify best.pt path in the script

* To adjust confidence threshold: Change 0.8 value in the if condition

* To change display text: Modify the cv.putText() parameters
