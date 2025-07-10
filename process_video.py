import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import insightface
from tqdm import tqdm

# --- DeepSORT Imports ---
from deep_sort_realtime.deep_sort_realtime import DeepSort

# --- Configuration ---
# Directories
WORKSPACE_DIR = "workspace"
VIDEOS_DIR = os.path.join(WORKSPACE_DIR, "uploaded_videos")
OBJECT_CROPS_DIR = os.path.join(WORKSPACE_DIR, "object_crops")
ANALYSIS_DIR = os.path.join(WORKSPACE_DIR, "analysis_data")
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OBJECT_CROPS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Processing settings
FRAME_INTERVAL = 5  # Process every 5th frame to speed up, 1 for max detail

# --- Model Loading ---
def load_models():
    """Loads all necessary models and sets up the device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # YOLOv8 for Object Detection
    yolo_model = YOLO("yolov8l.pt") # 'l' for large, good balance

    # DeepSORT for Object Tracking
    deepsort = DeepSort(
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True if device == 'cuda' else False
    )
    
    # CLIP for Appearance Embeddings
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # InsightFace for Face Recognition Embeddings
    face_analysis_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analysis_model.prepare(ctx_id=0, det_size=(640, 640))

    models = {
        'device': device,
        'yolo': yolo_model,
        'deepsort': deepsort,
        'clip_model': clip_model,
        'clip_processor': clip_processor,
        'face_model': face_analysis_model
    }
    return models


def get_appearance_embedding(crop_image, models):
    """Generates a CLIP embedding for a cropped object image."""
    device = models['device']
    image = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
    inputs = models['clip_processor'](images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = models['clip_model'].get_image_features(**inputs)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()

def get_face_embedding(crop_image, models):
    """Detects a face in a crop and returns an ArcFace embedding."""
    faces = models['face_model'].get(crop_image)
    if faces and len(faces) > 0:
        # Return embedding of the largest face found in the crop
        return faces[0].normed_embedding
    return None


def process_video(video_path, models):
    """
    Processes a single video file to detect, track, and analyze objects.
    Saves the analysis data to a .pkl file.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup output paths
    video_crops_dir = os.path.join(OBJECT_CROPS_DIR, video_name)
    os.makedirs(video_crops_dir, exist_ok=True)
    analysis_file_path = os.path.join(ANALYSIS_DIR, f"{video_name}_analysis.pkl")

    all_objects_data = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to speed up processing
        if frame_idx % FRAME_INTERVAL != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        timestamp = frame_idx / fps

        # --- Step 1: Object Detection (YOLOv8) ---
        detections = models['yolo'](frame, verbose=False)[0]
        
        # Prepare detections for DeepSORT: [bbox_xywh, confidence, class_id]
        results_for_deepsort = []
        for det in detections.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            class_name = models['yolo'].names[cls]

            # We only care about tracking certain objects
            if class_name in ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']:
                xywh = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
                results_for_deepsort.append((xywh, conf, cls))

        # --- Step 2: Object Tracking (DeepSORT) ---
        if len(results_for_deepsort) > 0:
            # Convert to format expected by deep_sort_realtime
            detections = []
            for xywh, conf, cls in results_for_deepsort:
                detections.append((xywh, conf, models['yolo'].names[cls]))
            
            tracks = models['deepsort'].update_tracks(detections, frame=frame)
            outputs = []
            for track in tracks:
                if track.is_confirmed():
                    bbox = track.to_tlbr()  # top left bottom right
                    track_id = track.track_id
                    # Find the class_id from the original detections
                    class_id = None
                    for xywh, conf, cls in results_for_deepsort:
                        if abs(bbox[0] - xywh[0]) < 10 and abs(bbox[1] - xywh[1]) < 10:  # Rough matching
                            class_id = cls
                            break
                    if class_id is not None:
                        outputs.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id, class_id])
        else:
            outputs = []

        # --- Step 3: Crop, Enrich, and Embed ---
        if len(outputs) > 0:
            for track in outputs:
                x1, y1, x2, y2, track_id, cls_id = map(int, track)
                
                # Crop the object from the frame
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Save the crop
                crop_filename = f"track_{track_id}_frame_{frame_idx}.jpg"
                crop_path = os.path.join(video_crops_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

                # Generate Embeddings
                appearance_emb = get_appearance_embedding(crop, models)
                face_emb = None
                class_name = models['yolo'].names[cls_id]
                if class_name == 'person':
                    face_emb = get_face_embedding(crop, models)

                # Store all data for this object instance
                object_data = {
                    "track_id": f"{video_name}_track_{track_id}",
                    "base_class": class_name,
                    "timestamp": timestamp,
                    "object_crop_path": crop_path,
                    "bounding_box": [x1, y1, x2, y2],
                    "appearance_embedding": appearance_emb,
                    "face_embedding": face_emb
                }
                all_objects_data.append(object_data)
        
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # --- Step 4: Save Analysis File ---
    print(f"\nSaving analysis data for {video_name} to {analysis_file_path}...")
    with open(analysis_file_path, 'wb') as f:
        pickle.dump(all_objects_data, f)
    
    print("Processing complete.")


if __name__ == "__main__":
    # Put the name of the video you want to process here
    # Make sure the video is in the 'workspace/uploaded_videos' folder
    target_video_name = "your_video_here.mp4" 
    video_path = os.path.join(VIDEOS_DIR, target_video_name)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please place your video in the 'workspace/uploaded_videos' directory.")
    else:
        print("Loading all AI models. This may take a moment...")
        models = load_models()
        process_video(video_path, models)