!pip install ultralytics
import cv2
from ultralytics import YOLO
import os

# Load a model
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)