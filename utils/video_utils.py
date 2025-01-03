import cv2
import time
import json
from datetime import datetime
import os

class VideoProcessor:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_path = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.output_path, exist_ok=True)
        self.risk_history = []
        
        # Create or load existing analysis file
        self.analysis_file = os.path.join(self.output_path, "analysis.json")
        if os.path.exists(self.analysis_file):
            with open(self.analysis_file, 'r') as f:
                self.risk_history = json.load(f)

    def save_analysis_results(self, frame_count, analysis):
        """Save analysis results to JSON"""
        try:
            # Create new result entry
            result = {
                'frame': frame_count,
                'timestamp': time.time(),
                'analysis': analysis
            }
            
            # Append to history
            self.risk_history.append(result)
            
            # Write entire history to file
            with open(self.analysis_file, 'w') as f:
                json.dump(self.risk_history, f, indent=4)
                
            print(f"Saved analysis for frame {frame_count}")  # Debug logging
            
        except Exception as e:
            print(f"Error saving analysis results: {str(e)}")
            import traceback
            print(traceback.format_exc()) 