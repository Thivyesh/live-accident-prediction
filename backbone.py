import cv2
import os
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor



            
class AccidentPredictor:
    def __init__(self, args):
        super(AccidentPredictor, self).__init__()
        self.model_path = args.model_id
        self.init_model(args=args)
    
    def init_model(self, args=None):
        if "Llama-3.2" in self.model_path:
            self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
            self.processor = AutoProcessor.from_pretrained(self.model_path)

        elif "gpt" in self.model_path:
            self.api_key = args.api_key
                
    def scene_analysis(self, image_path):
        # Adapt from Scenedescription
        prompt = "Analyze the traffic scene and identify potential risk factors."
        return self.vlm_inference(text=prompt, image_path=image_path)
    
    def get_critical_objects(self, image_path):
        # Modify from get_objects
        prompt = "Identify vehicles showing dangerous behavior or positioning, specify their location and movement patterns."
        return self.vlm_inference(text=prompt, image_path=image_path)
    
    def risk_assessment(self, detected_objects, image_path):
        # New method for risk scoring
        prompt = f"Given these objects and their behaviors: {detected_objects}\nRate the collision risk on a scale of 1-10 and explain why."
        return self.vlm_inference(text=prompt, image_path=image_path)

    def spatial_reasoning(self, image_path):
        # Keep and modify for overhead perspective
        prompt = "Detect all vehicles and their trajectories, noting any irregular patterns or unsafe distances."
        return self.vlm_inference(text=prompt, image_path=image_path)

class VideoProcessor:
    def frames_to_video(self, input_folder, output_file, fps):
        images = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
        images.sort()  # Sort files by name to maintain order

        frame = cv2.imread(os.path.join(input_folder, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        for image in images:
            img_path = os.path.join(input_folder, image)
            frame = cv2.imread(img_path)
            out.write(frame) 
    
        out.release()
        print(f"Video saved as {output_file}")

    def extract_frames(self, video_path, output_folder, frame_rate=1):
        # New method for processing video feeds
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_rate == 0:
                cv2.imwrite(f"{output_folder}/frame_{frame_count}.jpg", frame)
            frame_count += 1
        cap.release()
        
class TrajectoryAnalyzer:
    def analyze_vehicle_trajectories(self, trajectories):
        # New method for identifying dangerous patterns
        risk_factors = {
            'sudden_stops': [],
            'erratic_movements': [],
            'close_encounters': [],
            'speed_violations': []
        }
        # Add analysis logic here
        return risk_factors

    def compute_collision_probability(self, vehicle_paths):
        # New method for predicting potential collisions
        collision_risks = []
        # Add collision prediction logic here
        return collision_risks