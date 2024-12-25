from ultralytics import YOLO
import cv2
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
import threading
from queue import Queue, Empty
import time
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

class AccidentPredictor:
    def __init__(self):
        # Initialize YOLO model
        self.yolo = YOLO("yolo11n.pt")
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)

        # Add risk analysis parameters
        self.risk_threshold = 0.7
        self.risk_history = []
        
        # Initialize queues for different analysis types
        self.scene_queue = Queue()
        self.critical_queue = Queue()
        self.risk_queue = Queue()
        self.processing = False
        
        # Setup output directory
        self.output_dir = "output"
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(self.output_dir, self.timestamp), exist_ok=True)

        # Current analysis results
        self.current_analysis = {
            'scene_description': "",
            'critical_objects': "",
            'risk_assessment': "",
            'risk_score': 0
        }
        
    def detect_objects(self, frame):
        """Run YOLO object detection"""
        results = self.yolo.track(frame, persist=True)[0]
        detected = []
        
        for box in results.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            class_name = results.names[class_id]
            if conf > 0.5:
                detected.append(f"{class_name} ({conf:.2f})")
        
        return detected, results

    def frame_to_base64(self, frame):
        """Convert frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def start_analysis(self, frame, detected_objects):
        """Start all analysis threads"""
        if not self.processing:
            self.processing = True
            base64_image = self.frame_to_base64(frame)
            
            # Start separate threads for each analysis type
            threads = [
                threading.Thread(target=self._analyze_scene, args=(base64_image, detected_objects)),
                threading.Thread(target=self._analyze_critical_objects, args=(base64_image,)),
                threading.Thread(target=self._analyze_risk, args=(base64_image, detected_objects))
            ]
            
            for thread in threads:
                thread.daemon = True
                thread.start()

    def _analyze_scene(self, base64_image, detected_objects):
        """Analyze scene in separate thread"""
        try:
            prompt = """Analyze this traffic scene. Describe:
            1. Overall traffic flow
            2. Number of vehicles
            3. Any immediate hazards
            4. Road conditions"""
            
            response = self._get_vision_analysis(prompt, base64_image)
            self.scene_queue.put(response)
        except Exception as e:
            print(f"Scene analysis error: {e}")

    def _analyze_critical_objects(self, base64_image):
        """Analyze critical objects in separate thread"""
        try:
            prompt = """Identify vehicles or situations that could lead to accidents:
            1. Vehicles moving erratically
            2. Dangerous proximity between vehicles
            3. Vehicles breaking traffic rules"""
            
            response = self._get_vision_analysis(prompt, base64_image)
            self.critical_queue.put(response)
        except Exception as e:
            print(f"Critical objects analysis error: {e}")

    def _analyze_risk(self, base64_image, detected_objects):
        """Analyze risk in separate thread"""
        try:
            prompt = f"""Given these objects: {', '.join(detected_objects)}
            Rate the accident risk from 1-10 and explain why:
            1. Immediate collision risk
            2. Contributing environmental factors
            3. Recommended preventive actions"""
            
            response = self._get_vision_analysis(prompt, base64_image)
            self.risk_queue.put(response)
        except Exception as e:
            print(f"Risk analysis error: {e}")
        finally:
            self.processing = False

    def _get_vision_analysis(self, prompt, base64_image):
        """Helper method to get vision analysis"""
        response = self.client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=250
        )
        return response.choices[0].message.content

    def update_current_analysis(self):
        """Update current analysis from queues"""
        try:
            self.current_analysis['scene_description'] = self.scene_queue.get_nowait()
        except Empty:
            pass

        try:
            self.current_analysis['critical_objects'] = self.critical_queue.get_nowait()
        except Empty:
            pass

        try:
            risk_assessment = self.risk_queue.get_nowait()
            self.current_analysis['risk_assessment'] = risk_assessment
            # Try to extract risk score
            try:
                # Updated regex to match "Accident Risk Assessment: 8/10" format
                import re
                score_match = re.search(r'(?:Accident Risk Assessment:|Risk Assessment:|risk:?)\s*(\d+)(?:/10)?', 
                                    risk_assessment, re.IGNORECASE)
                if score_match:
                    self.current_analysis['risk_score'] = float(score_match.group(1))
                else:
                    print("No risk score found in:", risk_assessment[:100])  # Debug print
            except (IndexError, ValueError) as e:
                print(f"Error extracting risk score: {e}")  # Debug print
        except Empty:
            pass

def main():
    cap = cv2.VideoCapture("../Skjermopptak 2024-12-25 kl. 15.24.58.mov")  # Use 0 for webcam or provide video file path
    predictor = AccidentPredictor()
    
    last_analysis_time = time.time()
    frame_count = 0
    analysis_interval = 3.0  # Seconds between analyses
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        detected_objects, results = predictor.detect_objects(frame)
        
        # Start new analysis if enough time has passed
        current_time = time.time()
        if current_time - last_analysis_time >= analysis_interval and not predictor.processing:
            predictor.start_analysis(frame, detected_objects)
            last_analysis_time = current_time
        
        # Update current analysis from queues
        predictor.update_current_analysis()
        
        # Visualize results
        annotated_frame = results.plot()
        
        # Add analysis visualization
        analysis = predictor.current_analysis
        y_offset = 30
        cv2.putText(annotated_frame, f"Objects: {', '.join(detected_objects)}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if analysis['risk_score'] > 0:
            y_offset += 30
            risk_color = (0, 255, 0) if analysis['risk_score'] < 5 else \
                        (0, 165, 255) if analysis['risk_score'] < 7 else (0, 0, 255)
            cv2.putText(annotated_frame, f"Risk Score: {analysis['risk_score']}/10", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        
        # Save analysis results periodically
        if frame_count % 300 == 0:  # Every 300 frames
            result = {
                'frame': frame_count,
                'timestamp': time.time(),
                'analysis': predictor.current_analysis
            }
            predictor.risk_history.append(result)
            
            with open(f"{predictor.output_dir}/{predictor.timestamp}/analysis.json", 'w') as f:
                json.dump(predictor.risk_history, f, indent=2)
        
        cv2.imshow('Accident Prediction Feed', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # Save final analysis results
    with open(f"{predictor.output_dir}/{predictor.timestamp}/analysis.json", 'w') as f:
        json.dump(predictor.risk_history, f, indent=2)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()