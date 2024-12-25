from ultralytics import YOLO
import cv2
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
import threading
from queue import Queue, Empty
import time
import pygame
import sounddevice as sd
import numpy as np

# Load environment variables from .env file
load_dotenv()

class SceneDescriber:
    def __init__(self):
        # Initialize YOLO model
        self.yolo = YOLO("yolo11n.pt")
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        
        # Initialize threading components
        self.description_queue = Queue()
        self.processing = False
        
        # Simplified audio initialization
        self.current_audio = None
        pygame.init()
    
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
    
    def describe_scene_async(self, frame, detected_objects):
        """Start async scene description"""
        if not self.processing:
            self.processing = True
            thread = threading.Thread(
                target=self._describe_scene_thread,
                args=(frame, detected_objects)
            )
            thread.daemon = True
            thread.start()
    
    def _stream_audio(self, response):
        """Stream audio in real-time using sounddevice"""
        try:
            # Convert response content to numpy array and play
            audio_array = np.frombuffer(response.content, dtype=np.float32)
            sd.play(audio_array, samplerate=24000, blocking=False)
            
        except Exception as e:
            print(f"Error streaming audio: {e}")
    
    def _describe_scene_thread(self, frame, detected_objects):
        """Thread function for scene description"""
        try:
            objects_text = ", ".join(detected_objects)
            base64_image = self.frame_to_base64(frame)
            
            print("Sending request to OpenAI...")  # Debug print
            
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"I see these objects: {objects_text}. Describe what these objects are doing in a brief sentence."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            
            description = response.choices[0].message.content
            print(f"Got description: {description}")  # Debug print
            
            # Generate speech in a separate thread
            self._generate_and_play_speech(description)
            
            self.description_queue.put(description)
        except Exception as e:
            print(f"Error in description thread: {e}")
        finally:
            self.processing = False
    
    def _generate_and_play_speech(self, text):
        """Generate and stream speech using OpenAI's TTS"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            # Start audio streaming in a separate thread
            audio_thread = threading.Thread(
                target=self._stream_audio,
                args=(response,)
            )
            audio_thread.daemon = True
            audio_thread.start()
            
        except Exception as e:
            print(f"Error generating or streaming speech: {e}")
    
    def __del__(self):
        """Cleanup audio resources"""
        sd.stop()

def main():
    cap = cv2.VideoCapture(0)
    describer = SceneDescriber()
    
    last_description_time = time.time()
    description = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run object detection on every frame
        detected_objects, results = describer.detect_objects(frame)
        
        # Request new description every 3 seconds if not currently processing
        current_time = time.time()
        if current_time - last_description_time >= 3 and detected_objects and not describer.processing:
            print(f"Detected objects: {detected_objects}")  # Debug print
            describer.describe_scene_async(frame, detected_objects)
            last_description_time = current_time
        
        # Check for new descriptions
        try:
            while True:  # Get the most recent description
                description = describer.description_queue.get_nowait()
        except Empty:
            pass
        
        # Visualize YOLO detections
        annotated_frame = results.plot()
        
        # Display the detected objects and description
        cv2.putText(annotated_frame, f"Objects: {', '.join(detected_objects)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Description: {description}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Webcam Feed', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()