from collections import defaultdict
import numpy as np

class TrajectoryTracker:
    def __init__(self, max_track_length=30):
        self.track_history = defaultdict(lambda: [])
        self.max_track_length = max_track_length

    def update(self, detected_objects):
        """Update trajectory history with new detections"""
        for obj in detected_objects:
            if obj['track_id'] is not None:
                track = self.track_history[obj['track_id']]
                # Store position as numpy array
                position = np.array([float(obj['x']), float(obj['y'])])
                track.append(position)
                if len(track) > self.max_track_length:
                    track.pop(0)

    def get_recent_tracks(self, num_frames=10):
        """Get recent tracking history"""
        return {k: v[-num_frames:] for k, v in self.track_history.items() if len(v) >= 2} 