import numpy as np
from openai import OpenAI
from queue import Queue, Empty
import threading
import base64
import cv2
import re
from typing import Literal, Optional
import requests
import json

class RiskAnalyzer:
    def __init__(self, api_key: Optional[str] = None, provider: Literal['openai', 'ollama'] = 'openai', 
                 model: str = None, ollama_host: str = "http://localhost:11434"):
        self.provider = provider
        self.ollama_host = ollama_host
        
        if provider == 'openai':
            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-4o-mini"
        else:
            self.model = model or "llama3.2-vision:latest"
        
        # Rest of the initialization remains the same
        self.scene_queue = Queue()
        self.risk_queue = Queue()
        self.processing = False
        self.risk_threshold = 0.7
        
        self.current_analysis = {
            'scene_description': "",
            'risk_assessment': "",
            'risk_score': 0
        }

    def analyze_scene(self, frame, detected_objects, trajectory_data):
        """Start risk analysis for the current frame"""
        if not self.processing:
            self.processing = True
            base64_image = self._frame_to_base64(frame)
            
            if len(detected_objects) > 5:
                threads = [
                    threading.Thread(target=self._analyze_scene, args=(base64_image, detected_objects)),
                    threading.Thread(target=self._analyze_risk, args=(base64_image, detected_objects, trajectory_data))
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

    def _analyze_risk(self, base64_image, detected_objects, trajectory_data):
        """Analyze risk in separate thread"""
        try:
            # Calculate trajectory insights
            trajectory_insights = self._analyze_trajectories(trajectory_data)
            
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

    def _analyze_trajectories(self, trajectory_data):
        """Analyze trajectory data to identify potential risks"""
        insights = []
        
        # Process only vehicles with sufficient tracking history
        for track_id, track in trajectory_data.items():
            if len(track) >= 3:
                # Calculate speed and direction changes
                speeds = []
                direction_changes = 0
                
                for i in range(1, len(track)):
                    prev_pos = np.array(track[i-1])
                    curr_pos = np.array(track[i])
                    speed = np.linalg.norm(curr_pos - prev_pos)
                    speeds.append(speed)
                
                    if i > 1:
                        vec1 = track[i-1] - track[i-2]
                        vec2 = track[i] - track[i-1]
                        angle = np.arctan2(np.cross(vec1, vec2), np.dot(vec1, vec2))
                        if abs(angle) > 0.5:  # ~30 degrees
                            direction_changes += 1
                
                avg_speed = np.mean(speeds)
                
                # Generate insight based on movement pattern
                if direction_changes > 1:
                    insights.append(f"Vehicle {track_id}: Erratic movement detected")
                elif avg_speed > 20:
                    insights.append(f"Vehicle {track_id}: High speed movement")
                elif avg_speed < 2:
                    insights.append(f"Vehicle {track_id}: Stationary or very slow")
        
        return "\n".join(insights) if insights else "No significant movement patterns detected"

    def _identify_collision_risks(self, vehicle_states):
        """Identify vehicles with collision risks"""
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
        """Helper method to get vision analysis from either OpenAI or Ollama"""
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
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
        else:
            try:
                payload = {
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                        "images": [base64_image]
                    }]
                }
                
                print(f"Sending request to Ollama at: {self.ollama_host}/api/chat")
                print(f"Using model: {self.model}")
                print(f"Prompt length: {len(prompt)}")
                print(f"Image data length: {len(base64_image)}")
                
                response = requests.post(
                    f"{self.ollama_host}/api/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                print(f"Ollama Response Status: {response.status_code}")
                print(f"Ollama Response Headers: {response.headers}")
                print(f"Ollama Response: {response.text[:500]}...")
                
                if response.status_code == 200:
                    response_data = response.json()
                    print(f"Parsed response data: {response_data}")
                    if 'message' in response_data and 'content' in response_data['message']:
                        return response_data['message']['content']
                    else:
                        print("Response format not as expected:", response_data)
                        return ""
                else:
                    print(f"Error from Ollama API: {response.status_code} - {response.text}")
                    return ""
                    
            except Exception as e:
                print(f"Error getting vision analysis: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return ""

    def _frame_to_base64(self, frame):
        """Convert frame to base64 string with resizing"""
        # Resize image to reduce processing time
        max_dimension = 800
        height, width = frame.shape[:2]
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        # Compress image
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')

    def update_current_analysis(self):
        """Update current analysis from queues"""
        updated = False
        
        try:
            scene_desc = self.scene_queue.get_nowait()
            if scene_desc:
                self.current_analysis['scene_description'] = scene_desc
                print("Updated scene description:", scene_desc)  # Debug logging
                updated = True
        except Empty:
            pass

        try:
            risk_assessment = self.risk_queue.get_nowait()
            if risk_assessment:
                self.current_analysis['risk_assessment'] = risk_assessment
                print("Updated risk assessment:", risk_assessment)  # Debug logging
                # Try to extract risk score
                try:
                    score_match = re.search(r'(?:Accident Risk Rating:|Risk Rating:|Risk Assessment:)\s*\*?\*?(\d+)(?:/10)?', 
                                        risk_assessment, re.IGNORECASE)
                    if score_match:
                        self.current_analysis['risk_score'] = float(score_match.group(1))
                        print("Extracted risk score:", self.current_analysis['risk_score'])  # Debug logging
                    else:
                        print("No risk score found in:", risk_assessment)
                except (IndexError, ValueError) as e:
                    print(f"Error extracting risk score: {e}")
                updated = True
        except Empty:
            pass

        return self.current_analysis if updated else None 