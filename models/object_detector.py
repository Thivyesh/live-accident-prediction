from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.yolo = YOLO(model_path)

    def detect(self, frame):
        """Run YOLO object detection with tracking"""
        results = self.yolo.track(frame, persist=True, device="mps", verbose=False)[0]
        detected = []
        
        # Get boxes and track IDs
        boxes = results.boxes.xywh.cpu()
        track_ids = results.boxes.id
        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()
        
        # Get detected objects with their track IDs
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls)
            conf = float(box.conf)
            class_name = results.names[class_id]
            if conf > 0.5:
                x, y, w, h = box.xywh[0].tolist()
                current_track_id = int(track_ids[i]) if track_ids is not None else None
                detected.append({
                    'name': class_name,
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'track_id': current_track_id
                })
        
        return detected, results 