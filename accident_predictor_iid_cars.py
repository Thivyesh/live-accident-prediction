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
from collections import defaultdict
import numpy as np


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
        
        # Add track history
        self.track_history = defaultdict(lambda: [])
        self.max_track_length = 30  # Number of frames to retain trajectory

    def detect_objects(self, frame):
        """Run YOLO object detection with tracking"""
        results = self.yolo.track(frame, persist=True, device="mps", verbose=False)[0]
        detected = []
        
        # Get boxes and track IDs
        boxes = results.boxes.xywh.cpu()
        track_ids = results.boxes.id
        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()
            
            # Update trajectory history
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > self.max_track_length:
                    track.pop(0)
        
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

    def frame_to_base64(self, frame):
        """Convert frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def start_analysis(self, frame, detected_objects):
        """Start all analysis threads"""
        if not self.processing:
            self.processing = True
            base64_image = self.frame_to_base64(frame)
            
            if len(detected_objects) > 5:  # Only analyze if more than 5 objects detected
                threads = [
                    threading.Thread(target=self._analyze_scene, args=(base64_image, detected_objects)),
                    threading.Thread(target=self._analyze_risk, args=(base64_image, detected_objects))
                ]
                
                for thread in threads:
                    thread.daemon = True
                    thread.start()
            else:
                self.processing = False

    def _analyze_scene(self, base64_image, detected_objects):
        """Analyze scene in separate thread"""
        try:
            prompt = """You are an expert traffic analyst. Analyze this traffic scene.
            1. Return only the status and a short description of the scene where STATUS should be:
            - COLLISION_RISK (if there is a risk of collision)
            - DAMAGED (if visual damage is detected on a vehicle)
            - COLLIDING (if collision is imminent or occurring)
            - SAFE (If there is no risk of collision or damage)
            
            The response should be in the following format:
            STATUS: <status>
            <description>"""
                    
            response = self._get_vision_analysis(prompt, base64_image)
            self.scene_queue.put(response)
        except Exception as e:
            print(f"Scene analysis error: {e}")

    def _analyze_risk(self, base64_image, detected_objects):
        """Analyze risk in separate thread"""
        try:
            # Calculate trajectory insights
            trajectory_insights, collision_areas = self._analyze_trajectories()
            
            detected_objects_str = ', '.join([f"{obj['name']} at ({obj['x']}, {obj['y']})" for obj in detected_objects])

            prompt = f"""Given these objects: {detected_objects_str}

Vehicle Movement Analysis:
{trajectory_insights}

Rate the accident risk from 1-10 and explain why, considering:
1. Immediate collision risk based on vehicle trajectories and speeds
2. Contributing environmental factors
3. Dangerous movement patterns identified
4. Recommended preventive actions
"""
            
            response = self._get_vision_analysis(prompt, base64_image)
            self.risk_queue.put(response)
        except Exception as e:
            print(f"Risk analysis error: {e}")
        finally:
            self.processing = False

    def _analyze_trajectories(self):
        """Analyze trajectory data to identify potential risks"""
        collision_areas = []
        vehicle_states = {}
        
        # Process only the most recent tracks (last 10 frames)
        recent_tracks = {k: v[-10:] for k, v in self.track_history.items() if len(v) >= 2}
        
        for track_id, track in recent_tracks.items():
            # Quick speed calculation using only last two points
            last_pos = np.array(track[-1])
            prev_pos = np.array(track[-2])
            speed = np.linalg.norm(last_pos - prev_pos)
            
            # Simplified direction change detection
            direction_changes = 0
            if len(track) > 3:
                # Check only the last 3 points for direction changes
                vec1 = np.array(track[-2]) - np.array(track[-3])
                vec2 = np.array(track[-1]) - np.array(track[-2])
                angle = np.arctan2(np.cross(vec1, vec2), np.dot(vec1, vec2))
                if abs(angle) > 0.5:
                    direction_changes = 1
            
            # Quick movement type classification
            movement_type = "stable"
            if speed < 5:
                movement_type = "stationary"
            elif direction_changes > 0:
                movement_type = "erratic"
            elif speed > 20:
                movement_type = "fast"
            
            vehicle_states[track_id] = {
                'position': last_pos,
                'speed': speed,
                'movement_type': movement_type
            }
        
        # Only check proximity for vehicles that are not stationary
        moving_vehicles = {k: v for k, v in vehicle_states.items() 
                          if v['movement_type'] != "stationary"}
        
        # Quick proximity check for moving vehicles
        if len(moving_vehicles) >= 2:
            collision_areas = self._identify_collision_areas(moving_vehicles)
        
        return "", collision_areas  # Skip generating text insights for performance

    def _identify_collision_areas(self, vehicle_states):
        """Identify vehicles with collision risks or damage"""
        risk_vehicles = {}
        tracked_ids = list(vehicle_states.keys())
        positions = np.array([vehicle_states[id]['position'] for id in tracked_ids])
        
        # Compute all pairwise distances at once
        if len(positions) >= 2:
            distances = np.linalg.norm(positions[:, None] - positions, axis=2)
            
            # Find pairs of vehicles that are close to each other
            close_pairs = np.where(distances < 100)  # Risk threshold distance
            
            for i, j in zip(*close_pairs):
                if i < j:  # Avoid duplicate pairs
                    id1, id2 = tracked_ids[i], tracked_ids[j]
                    speed1 = vehicle_states[id1]['speed']
                    speed2 = vehicle_states[id2]['speed']
                    
                    # Determine risk level based on speed and distance
                    distance = distances[i][j]
                    if distance < 30:  # Very close proximity
                        risk_level = 'COLLIDING' if (speed1 > 10 or speed2 > 10) else 'DAMAGED'
                    else:
                        risk_level = 'HIGH_RISK' if (speed1 > 15 or speed2 > 15) else 'COLLISION_RISK'
                    
                    # Store risk assessment for both vehicles
                    risk_vehicles[id1] = risk_level
                    risk_vehicles[id2] = risk_level
        
        return risk_vehicles

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
                # Updated regex to match markdown formatting
                import re
                score_match = re.search(r'(?:Accident Risk Rating:|Risk Rating:|Risk Assessment:)\s*\*?\*?(\d+)(?:/10)?', 
                                    risk_assessment, re.IGNORECASE)
                if score_match:
                    self.current_analysis['risk_score'] = float(score_match.group(1))
                else:
                    print("No risk score found in:", risk_assessment)  # Debug print
            except (IndexError, ValueError) as e:
                print(f"Error extracting risk score: {e}")  # Debug print
        except Empty:
            pass

    def visualize_risks(self, frame, results):
        """Visualize vehicle risks with bounding boxes and status tags"""
        annotated_frame = frame
        
        # Get status directly from scene_description
        frame_status = self.current_analysis.get('scene_description', 'STATUS: SAFE')
        if not frame_status:
            frame_status = 'STATUS: SAFE'
        # Add status to lower left corner
        frame_height, frame_width = annotated_frame.shape[:2]
        status_color = {
            'SAFE': (0, 255, 0),
            'COLLISION_RISK': (0, 165, 255),
            'HIGH_RISK': (0, 0, 255),
            'COLLIDING': (0, 0, 255),
            'DAMAGED': (128, 0, 128)
        }.get(frame_status.split(': ')[1], (0, 255, 0))
        
        # Get text size for positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        status_size = cv2.getTextSize(frame_status, font, font_scale, thickness)[0]
        
        # Calculate position (20 pixels from left edge, 30 pixels from bottom)
        status_pos = (20, frame_height - 30)
        
        # Add dark background for better visibility
        padding = 5
        cv2.rectangle(annotated_frame, 
                     (status_pos[0] - padding, status_pos[1] - status_size[1] - padding),
                     (status_pos[0] + status_size[0] + padding, status_pos[1] + padding),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated_frame, frame_status, status_pos, font, font_scale, status_color, thickness)

        # Color scheme for different statuses (BGR format)
        color_map = {
            'SAFE': (0, 255, 0),          # Green
            'COLLISION_RISK': (0, 165, 255), # Orange
            'HIGH_RISK': (0, 0, 255),     # Red
            'COLLIDING': (0, 0, 255),     # Red
            'DAMAGED': (128, 0, 128)      # Purple
        }

        # Process vehicle states for collision detection
        vehicle_states = {}
        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes.xywh, results.boxes.id):
                if track_id is not None:
                    track_id = int(track_id)
                    x, y, w, h = box.tolist()
                    
                    # Get trajectory data
                    trajectory = self.track_history.get(track_id, [])
                    if len(trajectory) >= 2:
                        # Calculate speed from last two points
                        last_pos = np.array(trajectory[-1])
                        prev_pos = np.array(trajectory[-2])
                        speed = np.linalg.norm(last_pos - prev_pos)
                    else:
                        speed = 0
                        last_pos = np.array([x, y])
                    
                    vehicle_states[track_id] = {
                        'position': last_pos,
                        'speed': speed,
                        'box': (x, y, w, h)
                    }

        # Get risk assessment from collision detection
        risk_vehicles = self._identify_collision_areas(vehicle_states)

        # Draw boxes and trajectories
        for track_id, state in vehicle_states.items():
            x, y, w, h = state['box']
            
            # Determine status and color
            status = risk_vehicles.get(track_id, 'SAFE')
            color = color_map.get(status, (0, 255, 0))  # Default to green if status not found

            # Draw box
            cv2.rectangle(annotated_frame,
                        (int(x - w/2), int(y - h/2)),
                        (int(x + w/2), int(y + h/2)),
                        color, 2)

            # Draw status label only if not safe
            if status != 'SAFE':
                cv2.putText(annotated_frame,
                        f"ID:{track_id} {status}",
                        (int(x - w/2), int(y - h/2 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

            # Draw trajectory
            if track_id in self.track_history:
                track = self.track_history[track_id][-10:]  # Last 10 points
                if len(track) > 1:
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points],
                                isClosed=False,
                                color=color,
                                thickness=1)

        return annotated_frame

    def save_analysis_results(self):
        """Save analysis results to a JSON file"""
        output_path = f"{self.output_dir}/{self.timestamp}/analysis.json"
        with open(output_path, 'w') as f:
            json.dump(self.risk_history, f, indent=4)

def main():
    cap = cv2.VideoCapture("../accident-data/Skjermopptak 2024-12-25 kl. 15.24.58.mov")  # Use 0 for webcam or provide video file path
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
        
        # Replace results.plot() with visualize_risks
        annotated_frame = predictor.visualize_risks(frame, results)
        
        # Draw trajectory lines
        for track_id, track in predictor.track_history.items():
            if len(track) > 1:
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, 
                            color=(230, 230, 230), thickness=2)
        
        # Add analysis visualization
        analysis = predictor.current_analysis
        y_offset = 30
        
        # # Convert detected objects to a string representation
        # detected_objects_str = ', '.join([f"{obj['name']} at ({obj['x']}, {obj['y']})" for obj in detected_objects])
        # cv2.putText(annotated_frame, f"Objects: {detected_objects_str}", 
        #             (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
            
            predictor.save_analysis_results()
        
        cv2.imshow('Accident Prediction Feed', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # Save final analysis results
    predictor.save_analysis_results()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()