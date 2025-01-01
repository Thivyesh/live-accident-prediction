import cv2
import numpy as np

class RiskVisualizer:
    def __init__(self):
        self.color_map = {
            'SAFE': (0, 255, 0),          # Green
            'COLLISION_RISK': (0, 165, 255), # Orange
            'HIGH_RISK': (0, 0, 255),     # Red
            'COLLIDING': (0, 0, 255),     # Red
            'DAMAGED': (128, 0, 128)      # Purple
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2

    def draw_frame(self, frame, results, track_history, risk_assessment, detected_objects):
        """Draw all visualizations on the frame"""
        annotated_frame = frame.copy()
        
        # Draw status
        self._draw_status(annotated_frame, risk_assessment)
        
        # Draw vehicles and trajectories
        self._draw_vehicles(annotated_frame, results, track_history, risk_assessment)
        
        # Draw risk score
        if risk_assessment.get('risk_score', 0) > 0:
            self._draw_risk_score(annotated_frame, risk_assessment['risk_score'])
        
        return annotated_frame

    def _draw_status(self, frame, risk_assessment):
        """Draw overall status on the frame"""
        frame_status = risk_assessment.get('scene_description', 'STATUS: SAFE')
        if not frame_status:
            frame_status = 'STATUS: SAFE'
        
        # Get status color
        status = frame_status.split(': ')[1] if ': ' in frame_status else 'SAFE'
        status_color = self.color_map.get(status, self.color_map['SAFE'])
        
        # Calculate position and size
        frame_height, frame_width = frame.shape[:2]
        status_size = cv2.getTextSize(frame_status, self.font, self.font_scale, self.thickness)[0]
        status_pos = (20, frame_height - 30)
        
        # Draw background rectangle
        padding = 5
        cv2.rectangle(frame, 
                     (status_pos[0] - padding, status_pos[1] - status_size[1] - padding),
                     (status_pos[0] + status_size[0] + padding, status_pos[1] + padding),
                     (0, 0, 0), -1)
        
        # Draw status text
        cv2.putText(frame, frame_status, status_pos, 
                    self.font, self.font_scale, status_color, self.thickness)

    def _draw_vehicles(self, frame, results, track_history, risk_assessment):
        """Draw vehicle boxes and trajectories"""
        if results.boxes.id is not None:
            # Process vehicle states for visualization
            vehicle_states = self._get_vehicle_states(results, track_history)
            
            # Get risk assessment
            risk_vehicles = self._get_risk_states(vehicle_states)
            
            # Draw boxes and trajectories for each vehicle
            for track_id, state in vehicle_states.items():
                x, y, w, h = state['box']
                
                # Determine status and color
                status = risk_vehicles.get(track_id, 'SAFE')
                color = self.color_map.get(status, self.color_map['SAFE'])

                # Draw bounding box
                cv2.rectangle(frame,
                            (int(x - w/2), int(y - h/2)),
                            (int(x + w/2), int(y + h/2)),
                            color, 2)

                # Draw status label if not safe
                if status != 'SAFE':
                    self._draw_vehicle_label(frame, track_id, status, x, y, h, color)

                # Draw trajectory
                self._draw_trajectory(frame, track_id, track_history, color)

    def _draw_vehicle_label(self, frame, track_id, status, x, y, h, color):
        """Draw vehicle ID and status label"""
        label = f"ID:{track_id} {status}"
        label_pos = (int(x - h/2), int(y - h/2 - 5))
        
        # Get label size for background
        label_size = cv2.getTextSize(label, self.font, 0.5, 2)[0]
        
        # Draw background rectangle
        cv2.rectangle(frame,
                     (label_pos[0], label_pos[1] - label_size[1] - 2),
                     (label_pos[0] + label_size[0], label_pos[1] + 2),
                     (0, 0, 0), -1)
        
        # Draw label
        cv2.putText(frame, label, label_pos,
                    self.font, 0.5, color, 2)

    def _draw_trajectory(self, frame, track_id, track_history, color):
        """Draw trajectory line for a vehicle"""
        if track_id in track_history:
            track = track_history[track_id][-10:]  # Last 10 points
            if len(track) > 1:
                # Convert track points to integer coordinates
                points = np.array([point for point in track], dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(frame, [points],
                            isClosed=False,
                            color=color,
                            thickness=1)

    def _draw_risk_score(self, frame, risk_score):
        """Draw risk score on the frame"""
        risk_color = (0, 255, 0) if risk_score < 5 else \
                    (0, 165, 255) if risk_score < 7 else (0, 0, 255)
        
        score_text = f"Risk Score: {risk_score}/10"
        cv2.putText(frame, score_text,
                    (10, 30), self.font, self.font_scale, risk_color, self.thickness)

    def _get_vehicle_states(self, results, track_history):
        """Process vehicle states for visualization"""
        vehicle_states = {}
        
        for box, track_id in zip(results.boxes.xywh, results.boxes.id):
            if track_id is not None:
                track_id = int(track_id)
                x, y, w, h = box.tolist()
                
                # Calculate speed from trajectory if available
                trajectory = track_history.get(track_id, [])
                speed = 0
                if len(trajectory) >= 2:
                    # trajectory points are already numpy arrays
                    last_pos = trajectory[-1]
                    prev_pos = trajectory[-2]
                    speed = np.linalg.norm(last_pos - prev_pos)
                
                vehicle_states[track_id] = {
                    'position': np.array([x, y]),
                    'speed': speed,
                    'box': (x, y, w, h)
                }
        
        return vehicle_states

    def _get_risk_states(self, vehicle_states):
        """Determine risk states for vehicles"""
        risk_vehicles = {}
        tracked_ids = list(vehicle_states.keys())
        
        if len(tracked_ids) >= 2:
            # Ensure positions are numpy arrays
            positions = np.array([vehicle_states[id]['position'] for id in tracked_ids])
            
            # Calculate pairwise distances
            distances = np.linalg.norm(positions[:, None] - positions, axis=2)
            
            close_pairs = np.where(distances < 100)
            for i, j in zip(*close_pairs):
                if i < j:  # Avoid duplicate pairs
                    id1, id2 = tracked_ids[i], tracked_ids[j]
                    speed1 = vehicle_states[id1]['speed']
                    speed2 = vehicle_states[id2]['speed']
                    
                    distance = distances[i][j]
                    if distance < 30:
                        risk_level = 'COLLIDING' if (speed1 > 10 or speed2 > 10) else 'DAMAGED'
                    else:
                        risk_level = 'HIGH_RISK' if (speed1 > 15 or speed2 > 15) else 'COLLISION_RISK'
                    
                    risk_vehicles[id1] = risk_level
                    risk_vehicles[id2] = risk_level
        
        return risk_vehicles