import cv2
import time
import json
from datetime import datetime
import os

class VideoProcessor:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(output_dir, self.timestamp), exist_ok=True)
        self.risk_history = []

    def save_analysis_results(self, frame_count, analysis):
        """Save analysis results to JSON"""
        try:
            result = {
                'frame': frame_count,
                'timestamp': time.time(),
                'analysis': analysis
            }
            self.risk_history.append(result)
            
            output_path = f"{self.output_dir}/{self.timestamp}/analysis.json"
            print(f"Saving analysis to {output_path}")  # Debug logging
            print(f"Analysis content: {result}")  # Debug logging
            
            with open(output_path, 'w') as f:
                json.dump(self.risk_history, f, indent=4)
                
        except Exception as e:
            print(f"Error saving analysis results: {str(e)}") 