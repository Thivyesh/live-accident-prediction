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
        self.verification_path = os.path.join(self.output_path, "verification_images")
        
        # Create directories
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.verification_path, exist_ok=True)
        
        self.risk_history = []
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

    def save_verification_image(self, frame_count, frame, detected_objects, involved_vehicles):
        """Save verification image when accidents are detected"""
        try:
            # Create RiskVisualizer instance if not already created
            if not hasattr(self, 'visualizer'):
                from visualization.risk_visualizer import RiskVisualizer
                self.visualizer = RiskVisualizer()
            
            # Generate verification frame
            verification_frame = self.visualizer.draw_accident_verification(
                frame, detected_objects, involved_vehicles
            )
            
            # Save the image
            filename = f"verification_frame_{frame_count}.jpg"
            filepath = os.path.join(self.verification_path, filename)
            cv2.imwrite(filepath, verification_frame)
            
            # Add verification image info to analysis
            for result in self.risk_history:
                if result['frame'] == frame_count:
                    result['verification_image'] = filename
                    break
            
            # Update analysis.json
            with open(self.analysis_file, 'w') as f:
                json.dump(self.risk_history, f, indent=4)
                
            print(f"Saved verification image for frame {frame_count}")
            
        except Exception as e:
            print(f"Error saving verification image: {str(e)}")
            import traceback
            print(traceback.format_exc()) 