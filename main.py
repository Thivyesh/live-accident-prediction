from models.object_detector import ObjectDetector
from models.trajectory_tracker import TrajectoryTracker
from models.risk_analyzer import RiskAnalyzer
from visualization.risk_visualizer import RiskVisualizer
from utils.video_utils import VideoProcessor
from dotenv import load_dotenv
import cv2
import os
import time

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize components
    detector = ObjectDetector()
    tracker = TrajectoryTracker()
    analyzer = RiskAnalyzer(
        provider='ollama',
        model='llava:latest',
        ollama_host='http://localhost:11434'
    )
    visualizer = RiskVisualizer()
    video_processor = VideoProcessor()

    # Video capture setup
    cap = cv2.VideoCapture("../accident-data/Skjermopptak 2024-12-25 kl. 15.24.58.mov")
    
    last_analysis_time = time.time()
    frame_count = 0
    analysis_interval = 3.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects and update trajectories
        detected_objects, results = detector.detect(frame)
        tracker.update(detected_objects)
        
        # Perform risk analysis
        current_time = time.time()
        if current_time - last_analysis_time >= analysis_interval and not analyzer.processing:
            print(f"Starting new analysis at frame {frame_count}")  # Debug logging
            analyzer.analyze_scene(frame, detected_objects, tracker.get_recent_tracks())
            last_analysis_time = current_time
        
        # Update current analysis
        current_analysis = analyzer.update_current_analysis()
        if current_analysis:
            print(f"Got new analysis at frame {frame_count}")  # Debug logging
        
        # Visualize results
        annotated_frame = visualizer.draw_frame(
            frame, results, tracker.track_history, 
            analyzer.current_analysis, detected_objects
        )
        
        # Save results periodically
        if frame_count % 300 == 0:
            print(f"Saving analysis results at frame {frame_count}")  # Debug logging
            video_processor.save_analysis_results(frame_count, analyzer.current_analysis)
        
        cv2.imshow('Accident Prediction Feed', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    # Save final results
    video_processor.save_analysis_results(frame_count, analyzer.current_analysis)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 