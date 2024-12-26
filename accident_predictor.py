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
        
        # Get detected objects as before
        for box in results.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            class_name = results.names[class_id]
            if conf > 0.5:
                # Ensure you are accessing the correct elements of the tensor
                x, y, w, h = box.xywh[0].tolist()  # Convert the tensor to a list
                detected.append({
                    'name': class_name,
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
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
            
            # Check if analysis is necessary
            if self.should_analyze(detected_objects):
                # Start separate threads for each analysis type
                threads = [
                    threading.Thread(target=self._analyze_scene, args=(base64_image, detected_objects)),
                    threading.Thread(target=self._analyze_critical_objects, args=(base64_image,)),
                    threading.Thread(target=self._analyze_risk, args=(base64_image, detected_objects)),
                    threading.Thread(target=self._generate_bounding_boxes, args=(base64_image, detected_objects, frame))
                ]
                
                for thread in threads:
                    thread.daemon = True
                    thread.start()
            else:
                self.processing = False

    def should_analyze(self, detected_objects):
        """Determine if analysis should be performed"""
        # Example condition: only analyze if more than a certain number of objects are detected
        return len(detected_objects) > 5

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
            # Calculate trajectory insights
            trajectory_insights, collision_areas = self._analyze_trajectories()
            
            # Convert detected objects to a string representation
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
        """Simplified collision area identification"""
        collision_areas = []
        tracked_ids = list(vehicle_states.keys())
        
        # Use numpy arrays for faster computation
        positions = np.array([vehicle_states[id]['position'] for id in tracked_ids])
        
        # Compute all pairwise distances at once
        if len(positions) >= 2:
            distances = np.linalg.norm(positions[:, None] - positions, axis=2)
            
            # Find pairs of vehicles that are close to each other
            close_pairs = np.where(distances < 100)  # Fixed risk radius for simplicity
            
            for i, j in zip(*close_pairs):
                if i < j:  # Avoid duplicate pairs
                    pos1 = positions[i]
                    pos2 = positions[j]
                    
                    # Quick bounding box calculation
                    x_min, y_min = np.minimum(pos1, pos2)
                    x_max, y_max = np.maximum(pos1, pos2)
                    
                    # Fixed padding for performance
                    padding = 40
                    collision_areas.append({
                        'x': int(x_min - padding),
                        'y': int(y_min - padding),
                        'width': int(x_max - x_min + 2 * padding),
                        'height': int(y_max - y_min + 2 * padding),
                        'risk_level': 'high' if vehicle_states[tracked_ids[i]]['speed'] > 15 
                                             or vehicle_states[tracked_ids[j]]['speed'] > 15 
                                          else 'medium'
                    })
        
        return collision_areas

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
        """Optimized risk visualization with combined predictions"""
        annotated_frame = results.plot()
        overlay = annotated_frame.copy()
        
        # Get trajectory-based collision areas
        _, trajectory_areas = self._analyze_trajectories()
        
        # Get LLM boxes from current analysis
        llm_boxes = self.current_analysis.get('bounding_boxes', []) if hasattr(self, 'current_analysis') else []
        
        # Combine predictions
        combined_areas = self._combine_predictions(trajectory_areas, llm_boxes)
        
        # Draw all areas
        for area in combined_areas:
            if area.get('combined', False):
                color = (255, 165, 0)  # Orange for combined predictions
            elif area['risk_level'] == 'high':
                color = (0, 0, 255)  # Red for high risk
            elif area['risk_level'] == 'llm':
                color = (255, 0, 0)  # Blue for LLM
            else:
                color = (0, 165, 255)  # Light orange for medium risk
            
            # Draw filled rectangle
            cv2.rectangle(
                overlay,
                (area['x'], area['y']),
                (area['x'] + area['width'], area['y'] + area['height']),
                color,
                -1
            )
            
            # Add label
            label = "COMBINED" if area.get('combined', False) else area['risk_level'].upper()
            cv2.putText(
                annotated_frame,
                f"{label} RISK",
                (area['x'], area['y'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Apply transparency for all boxes at once
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
        
        return annotated_frame

    def save_analysis_results(self):
        """Save analysis results to a JSON file"""
        output_path = f"{self.output_dir}/{self.timestamp}/analysis.json"
        with open(output_path, 'w') as f:
            json.dump(self.risk_history, f, indent=4)  # Use indent=4 for readability

    def _generate_bounding_boxes(self, base64_image, detected_objects, frame):
        """Generate bounding boxes in a separate thread with trajectory insights"""
        try:
            # Get trajectory analysis results
            trajectory_insights, collision_areas = self._analyze_trajectories()
            
            # Format detected objects
            object_details = []
            for obj in detected_objects:
                object_details.append(f"{obj['name']} at ({obj['x']}, {obj['y']})")
            
            # Format collision areas from trajectory analysis
            collision_details = []
            for area in collision_areas:
                risk_level = area['risk_level']
                collision_details.append(
                    f"Collision risk area ({risk_level}): "
                    f"x={area['x']}, y={area['y']}, "
                    f"width={area['width']}, height={area['height']}"
                )

            # Create enhanced prompt with trajectory analysis
            prompt = f"""Analyze this traffic scene and provide collision area bounding boxes.

DETECTED OBJECTS:
{', '.join(object_details)}

TRAJECTORY ANALYSIS RESULTS:
- Detected collision areas: {len(collision_areas)}
{chr(10).join(collision_details)}

Based on the image and trajectory analysis above, provide ONLY the bounding box coordinates for collision areas in this exact format:
(x, y, width, height)

Example: (100, 200, 50, 50)

Rules:
1. Consider both moving and stationary vehicles
2. Pay special attention to areas identified by trajectory analysis
3. Look for additional potential collision areas not caught by trajectory analysis
4. Return ONLY the coordinates, no explanation needed
5. Each line should contain exactly one set of coordinates
"""

            # Get bounding box data from LLM
            response = self._get_vision_analysis(prompt, base64_image)
            
            # Save the response for debugging
            self._save_bounding_box_response(response)

            # Extract bounding box coordinates from response
            import re
            bbox_pattern = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
            matches = re.findall(bbox_pattern, response)
            
            bounding_boxes = []
            for match in matches:
                x, y, width, height = map(int, match)
                bounding_boxes.append((x, y, width, height))
            
            # Store bounding boxes in current analysis
            self.current_analysis['bounding_boxes'] = bounding_boxes

        except Exception as e:
            print(f"Bounding box generation error: {e}")
        finally:
            self.processing = False

    def _save_bounding_box_response(self, response):
        """Save the bounding box response to a JSON file for debugging"""
        output_dir = os.path.join(self.output_dir, self.timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "bounding_box_responses.json")

        # Load existing responses if the file exists
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                responses = json.load(f)
        else:
            responses = []

        # Append the new response
        responses.append(response)

        # Save the updated responses
        with open(output_path, 'w') as f:
            json.dump(responses, f, indent=4)

    def _combine_predictions(self, trajectory_areas, llm_boxes):
        """Combine trajectory and LLM predictions using weighted IoU"""
        if not llm_boxes:
            return trajectory_areas
        
        if not trajectory_areas:
            return [{'x': x, 'y': y, 'width': w, 'height': h, 'risk_level': 'llm'} 
                    for (x, y, w, h) in llm_boxes]

        combined_areas = []
        
        # Convert LLM boxes to same format as trajectory areas
        llm_areas = [{'x': x, 'y': y, 'width': w, 'height': h, 'risk_level': 'llm'} 
                     for (x, y, w, h) in llm_boxes]
        
        # Calculate IoU matrix between trajectory and LLM boxes
        iou_matrix = np.zeros((len(trajectory_areas), len(llm_areas)))
        for i, traj_box in enumerate(trajectory_areas):
            for j, llm_box in enumerate(llm_areas):
                iou_matrix[i, j] = self._calculate_iou(traj_box, llm_box)
        
        # Find matching boxes using IoU threshold
        iou_threshold = 0.2  # Adjust as needed
        matched_pairs = np.where(iou_matrix > iou_threshold)
        
        # Process matched pairs
        processed_traj_indices = set()
        processed_llm_indices = set()
        
        for traj_idx, llm_idx in zip(*matched_pairs):
            if traj_idx in processed_traj_indices or llm_idx in processed_llm_indices:
                continue
            
            traj_box = trajectory_areas[traj_idx]
            llm_box = llm_areas[llm_idx]
            
            # Calculate confidence weights
            traj_conf = 0.7 if traj_box['risk_level'] == 'high' else 0.5
            llm_conf = 0.6  # Base LLM confidence
            
            # Combine boxes using weighted average
            combined_box = self._weighted_combine_boxes(traj_box, llm_box, traj_conf, llm_conf)
            combined_areas.append(combined_box)
            
            processed_traj_indices.add(traj_idx)
            processed_llm_indices.add(llm_idx)
        
        # Add unmatched trajectory boxes
        for i, traj_box in enumerate(trajectory_areas):
            if i not in processed_traj_indices:
                combined_areas.append(traj_box)
        
        # Add unmatched LLM boxes
        for i, llm_box in enumerate(llm_areas):
            if i not in processed_llm_indices:
                combined_areas.append(llm_box)
        
        return combined_areas

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        # Convert to x1, y1, x2, y2 format
        box1_x1 = box1['x']
        box1_y1 = box1['y']
        box1_x2 = box1['x'] + box1['width']
        box1_y2 = box1['y'] + box1['height']
        
        box2_x1 = box2['x']
        box2_y1 = box2['y']
        box2_x2 = box2['x'] + box2['width']
        box2_y2 = box2['y'] + box2['height']
        
        # Calculate intersection
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0

    def _weighted_combine_boxes(self, traj_box, llm_box, traj_conf, llm_conf):
        """Combine two boxes using weighted average based on confidence"""
        total_conf = traj_conf + llm_conf
        w1 = traj_conf / total_conf
        w2 = llm_conf / total_conf
        
        # Weighted average of coordinates
        x = int(w1 * traj_box['x'] + w2 * llm_box['x'])
        y = int(w1 * traj_box['y'] + w2 * llm_box['y'])
        width = int(w1 * traj_box['width'] + w2 * llm_box['width'])
        height = int(w1 * traj_box['height'] + w2 * llm_box['height'])
        
        # Determine combined risk level
        risk_level = 'high' if traj_box['risk_level'] == 'high' else 'medium'
        
        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'risk_level': risk_level,
            'combined': True  # Mark as combined prediction
        }

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
        # Convert detected objects to a string representation
        detected_objects_str = ', '.join([f"{obj['name']} at ({obj['x']}, {obj['y']})" for obj in detected_objects])
        cv2.putText(annotated_frame, f"Objects: {detected_objects_str}", 
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