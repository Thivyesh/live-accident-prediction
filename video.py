from ultralytics import YOLO

# Load an official or custom model
model = YOLO("yolo11n.pt")  # Load an official Detect model

# Perform tracking with the model
results = model.track("https://www.youtube.com/watch?v=H9QSGqioYww", show=True, tracker="bytetrack.yaml")  # with ByteTrack