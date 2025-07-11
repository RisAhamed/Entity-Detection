import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
import insightface
from tqdm import tqdm

# Using the modern, well-maintained deep-sort-realtime library
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Configuration ---
WORKSPACE_DIR = "workspace"
VIDEOS_DIR = os.path.join(WORKSPACE_DIR, "uploaded_videos")
OBJECT_CROPS_DIR = os.path.join(WORKSPACE_DIR, "object_crops")
ANALYSIS_DIR = os.path.join(WORKSPACE_DIR, "analysis_data")
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OBJECT_CROPS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Process every frame for best tracking accuracy. Increase to 2 or 3 for more speed at the cost of accuracy.
FRAME_INTERVAL = 1

# --- Model Loading ---
def load_models():
    """Loads all necessary models and sets up the device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # YOLOv8 for Object Detection
    yolo_model = YOLO("yolov8l.pt")

    # DeepSORT for Object Tracking
    deepsort_tracker = DeepSort(
        max_age=30,  # Max frames to keep a track without detection
        n_init=3,    # Number of consecutive detections to start a track
        nms_max_overlap=1.0,
        max_iou_distance=0.7,
        embedder='mobilenet',
        half=True,
        bgr=False, # We will feed RGB frames
        embedder_gpu=(device == "cuda")
    )
    
    # CLIP for Appearance Embeddings
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval() # Set to eval mode
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # InsightFace for Face Recognition Embeddings
    face_analysis_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analysis_model.prepare(ctx_id=0, det_size=(640, 640))

    models = {
        'device': device,
        'yolo': yolo_model,
        'deepsort': deepsort_tracker,
        'clip_model': clip_model,
        'clip_processor': clip_processor,
        'face_model': face_analysis_model
    }
    return models

# --- BATCH Embedding Functions (for performance) ---

def get_batch_appearance_embeddings(crop_images, models):
    """Generates CLIP embeddings for a batch of cropped object images."""
    if not crop_images:
        return []

    device = models['device']
    pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in crop_images]
    
    inputs = models['clip_processor'](images=pil_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = models['clip_model'].get_image_features(**inputs)
    
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy()

def get_face_embedding(crop_image, models):
    """Detects a face in a single crop and returns an ArcFace embedding."""
    # Face detection is typically fast enough to run individually
    faces = models['face_model'].get(crop_image)
    if faces:
        return faces[0].normed_embedding
    return None

# --- Main Processing Function ---

def process_video(video_path, models):
    """Processes a video to detect, track, and create analysis data."""
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
    pbar = tqdm(total=total_frames, desc=f"Processing '{video_name}'")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        timestamp = frame_idx / fps
        
        # --- Step 1: Object Detection (YOLOv8) ---
        # We give the model an RGB frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = models['yolo'](frame_rgb, verbose=False)[0]

        # --- Step 2: Format Detections for DeepSORT ---
        results_for_deepsort = []
        for det in detections.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls_name = models['yolo'].names[int(det.cls[0])]
            
            # Filter for classes we want to track
            if cls_name in ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']:
                # Format: ([left, top, width, height], confidence, class_name)
                ltwh = [x1, y1, x2 - x1, y2 - y1]
                results_for_deepsort.append((ltwh, conf, cls_name))

        # --- Step 3: Object Tracking (DeepSORT) ---
        tracks = models['deepsort'].update_tracks(results_for_deepsort, frame=frame_rgb)
        
        # --- Step 4: Process Confirmed Tracks (Batch-wise) ---
        frame_crops = []
        track_info_for_frame = []
        
        for track in tracks:
            # We only process confirmed tracks
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            class_name = track.get_det_class()
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            
            # Crop the object from the original BGR frame
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            frame_crops.append(crop)
            track_info_for_frame.append({
                "track_id": track_id,
                "class_name": class_name,
                "bbox": [x1, y1, x2, y2],
                "crop": crop # Temporarily store the crop
            })
            
        # --- Step 5: BATCH Generate Embeddings ---
        if frame_crops:
            batch_appearance_embeddings = get_batch_appearance_embeddings(frame_crops, models)

            # Now, iterate through the processed tracks and assign their embeddings
            for i, track_info in enumerate(track_info_for_frame):
                track_id = track_info['track_id']
                class_name = track_info['class_name']
                x1, y1, x2, y2 = track_info['bbox']
                crop = track_info['crop']

                # Save the crop image
                crop_filename = f"track_{track_id}_frame_{frame_idx}.jpg"
                crop_path = os.path.join(video_crops_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

                # Get the corresponding embedding from the batch
                appearance_emb = batch_appearance_embeddings[i]
                
                # Get face embedding (individually) only if it's a person
                face_emb = get_face_embedding(crop, models) if class_name == 'person' else None

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

    # --- Step 6: Save Analysis File ---
    print(f"\nSaving analysis for {len(all_objects_data)} object instances to {analysis_file_path}...")
    with open(analysis_file_path, 'wb') as f:
        pickle.dump(all_objects_data, f)
    
    print("Processing complete.")

if __name__ == "__main__":
    # --- IMPORTANT ---

    target_video_name = "crowd_vedio2.mp4" 

    if not os.path.exists(target_video_name):
        print(f"Error: Video file not found at '{target_video_name}'")
        print("Please make sure the video is in the correct path.")
    else:
        # Move the video to the workspace folder for consistency
        video_workspace_path = os.path.join(VIDEOS_DIR, os.path.basename(target_video_name))
        if not os.path.exists(video_workspace_path):
            import shutil
            shutil.copy(target_video_name, video_workspace_path)

        print("Loading all AI models. This may take a moment...")
        models = load_models()
        process_video(video_workspace_path, models)