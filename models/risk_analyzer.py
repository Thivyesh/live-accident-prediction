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
            'risk_score': 0,
            'previous_status': "SAFE"
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

                # Start accident vehicle analysis if needed
                def check_and_analyze_accidents():
                    try:
                        scene_analysis = self.scene_queue.get(timeout=10)  # Wait for scene analysis
                        if scene_analysis and scene_analysis.get('status') in ['COLLIDING', 'DAMAGED']:
                            accident_analysis = self._analyze_accident_vehicles(
                                base64_image, detected_objects, scene_analysis
                            )
                            if accident_analysis:
                                self.current_analysis['involved_vehicles'] = accident_analysis
                    except Empty:
                        print("Timeout waiting for scene analysis")
                    except Exception as e:
                        print(f"Error in accident analysis: {e}")
                    finally:
                        self.processing = False

                accident_thread = threading.Thread(target=check_and_analyze_accidents)
                accident_thread.daemon = True
                accident_thread.start()
            else:
                self.processing = False

    def _analyze_scene(self, base64_image, detected_objects):
        """Analyze scene in separate thread"""
        try:
            # Get the complete previous analysis
            previous_analysis = "No previous analysis available."
            if self.current_analysis['scene_description']:
                previous_analysis = self.current_analysis['scene_description']

            prompt = f"""You are an expert traffic analyst. Analyze this traffic scene.
            Previous frame analysis was:
            {previous_analysis}

            Return your analysis in the following JSON format without any markdown formatting or code block syntax:
            {{
                "status": "<COLLISION_RISK|DAMAGED|COLLIDING|SAFE>",
                "description": "detailed description of the scene",
                "changes": "description of changes from previous frame"
            }}

            Status definitions:
            - COLLISION_RISK: if there is a risk of collision
            - DAMAGED: if visual damage is detected on a vehicle
            - COLLIDING: if collision is imminent or occurring
            - SAFE: if there is no risk of collision or damage

            Consider the previous analysis when analyzing the current frame. If the situation 
            is evolving, explain the changes in the 'changes' field.
            
            Important: Return only the JSON object, without any markdown formatting or explanation."""
                    
            response = self._get_vision_analysis(prompt, base64_image)
            
            # Clean up response - remove markdown code block if present
            if response.startswith('```'):
                response = response.split('\n', 1)[1]  # Remove first line
                response = response.rsplit('\n', 1)[0]  # Remove last line
                response = response.replace('```json\n', '').replace('```', '').strip()
            
            # Try to parse JSON response
            try:
                parsed_response = json.loads(response)
                self.scene_queue.put(parsed_response)
                print(f"Parsed response: {parsed_response}")
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                # print(f"Raw response: {response}")
                self.scene_queue.put(None)
                
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
        except Exception:
            # print(f"Risk analysis error: {e}")
            pass
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
                    "prompt": prompt,
                    "images": [base64_image],
                    "stream": False
                }
                
                print(f"Sending request to Ollama at: {self.ollama_host}/api/generate")
                print(f"Using model: {self.model}")
                
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                print(f"Ollama Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    print(f"Parsed response data: {response_data}")
                    
                    # Extract response from the new format
                    if 'response' in response_data:
                        return response_data['response']
                    else:
                        print("Unexpected response format:", response_data)
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
                # Store the entire JSON response
                self.current_analysis['scene_description'] = scene_desc
                # Update previous status
                self.current_analysis['previous_status'] = scene_desc.get('status', 'SAFE')
                print("Updated scene description:", scene_desc)  # Debug logging
                updated = True
        except Empty:
            pass

        try:
            risk_assessment = self.risk_queue.get_nowait()
            if risk_assessment:
                self.current_analysis['risk_assessment'] = risk_assessment
                # print("Updated risk assessment:", risk_assessment)  # Debug logging
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

    def _analyze_accident_vehicles(self, base64_image, detected_objects, scene_analysis):
        """Analyze which vehicles are involved in an accident when collision or damage is detected"""
        try:
            if scene_analysis.get('status') in ['COLLIDING', 'DAMAGED']:
                # Format detected objects for the prompt
                objects_info = "\n".join([
                    f"Vehicle ID {obj['track_id']}: at position ({obj['x']}, {obj['y']}), "
                    f"size {obj['width']}x{obj['height']}"
                    for obj in detected_objects if obj['track_id'] is not None
                ])

                prompt = f"""Given the following scene analysis:
                {json.dumps(scene_analysis, indent=2)}

                And these detected vehicles:
                {objects_info}

                Return a JSON response identifying the vehicles involved in the accident:
                {{
                    "involved_vehicles": [
                        {{
                            "track_id": <vehicle_id>,
                            "role": "<COLLIDING|DAMAGED>",
                            "position": {{"x": <x_coord>, "y": <y_coord>}},
                            "confidence": <0.0-1.0>
                        }}
                    ],
                    "explanation": "brief explanation of why these vehicles were selected"
                }}
                
                Only include vehicles that you are confident are involved in the accident based on 
                their position and the scene context."""

                response = self._get_vision_analysis(prompt, base64_image)
                
                try:
                    parsed_response = json.loads(response)
                    return parsed_response
                except json.JSONDecodeError as e:
                    print(f"Failed to parse accident vehicles JSON response: {e}")
                    print(f"Raw response: {response}")
                    return None

        except Exception as e:
            print(f"Accident vehicles analysis error: {e}")
            return None 