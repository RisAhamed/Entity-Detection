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
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR / "workspace"
VIDEOS_DIR = WORKSPACE_DIR / "uploaded_videos"
OBJECT_CROPS_DIR = WORKSPACE_DIR / "object_crops"
ANALYSIS_DIR = WORKSPACE_DIR / "analysis_data"
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OBJECT_CROPS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# New Configurable Parameters
FRAME_INTERVAL = 1  # Default frame interval
FRAME_SKIP = 1      # New parameter for frame skipping (default: process every frame)
DESIRED_CLASSES = ['person', 'car']  # Custom classes to detect (configurable)

# --- Model Loading ---
def load_models(yolo_model='yolov8l.pt'):
    """Loads all necessary models and sets up the device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yolo_model = YOLO(yolo_model)
    deepsort_tracker = DeepSort(
        max_age=30, n_init=3, nms_max_overlap=1.0, max_iou_distance=0.7,
        embedder='mobilenet', half=True, bgr=False, embedder_gpu=(device == "cuda")
    )
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    face_analysis_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analysis_model.prepare(ctx_id=0, det_size=(640, 640))

    models = {
        'device': device, 'yolo': yolo_model, 'deepsort': deepsort_tracker,
        'clip_model': clip_model, 'clip_processor': clip_processor, 'face_model': face_analysis_model
    }
    return models

# --- Batch Embedding Functions ---
def get_batch_appearance_embeddings(crop_images, models):
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
    faces = models['face_model'].get(crop_image)
    if faces:
        return faces[0].normed_embedding
    return None

# --- Main Processing Function ---
def process_video(video_path, models, desired_classes=DESIRED_CLASSES, frame_skip=FRAME_SKIP):
    """Processes a video with custom class filtering and frame skipping."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

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

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        timestamp = (frame_idx * frame_skip) / fps
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = models['yolo'](frame_rgb, verbose=False)[0]

        # Filter detections by desired classes
        results_for_deepsort = []
        for det in detections.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls_name = models['yolo'].names[int(det.cls[0])]
            if cls_name in desired_classes:
                ltwh = [x1, y1, x2 - x1, y2 - y1]
                results_for_deepsort.append((ltwh, conf, cls_name))

        # Skip frame if no desired objects detected
        if not results_for_deepsort:
            frame_idx += 1
            pbar.update(1)
            continue

        tracks = models['deepsort'].update_tracks(results_for_deepsort, frame=frame_rgb)
        
        frame_crops = []
        track_info_for_frame = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            class_name = track.get_det_class()
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            frame_crops.append(crop)
            track_info_for_frame.append({
                "track_id": track_id, "class_name": class_name, "bbox": [x1, y1, x2, y2], "crop": crop
            })

        if frame_crops:
            batch_appearance_embeddings = get_batch_appearance_embeddings(frame_crops, models)
            for i, track_info in enumerate(track_info_for_frame):
                track_id = track_info['track_id']
                class_name = track_info['class_name']
                x1, y1, x2, y2 = track_info['bbox']
                crop = track_info['crop']
                crop_filename = f"track_{track_id}_frame_{frame_idx}.jpg"
                crop_path = os.path.join(video_crops_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

                appearance_emb = batch_appearance_embeddings[i]
                face_emb = get_face_embedding(crop, models) if class_name == 'person' else None

                object_data = {
                    "track_id": f"{video_name}_track_{track_id}", "base_class": class_name,
                    "timestamp": timestamp, "object_crop_path": str(Path(crop_path).relative_to(SCRIPT_DIR)),
                    "bounding_box": [x1, y1, x2, y2], "appearance_embedding": appearance_emb,
                    "face_embedding": face_emb, "video_name": video_name  # Added for multi-video search
                }
                all_objects_data.append(object_data)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\nSaving analysis for {len(all_objects_data)} object instances to {analysis_file_path}...")
    with open(analysis_file_path, 'wb') as f:
        pickle.dump(all_objects_data, f)
    print("Processing complete.")

if __name__ == "__main__":
    target_video_name = "10secvideo.mp4"
    if not os.path.exists(target_video_name):
        print(f"Error: Video file not found at '{target_video_name}'")
    else:
        video_workspace_path = os.path.join(VIDEOS_DIR, os.path.basename(target_video_name))
        if not os.path.exists(video_workspace_path):
            import shutil
            shutil.copy(target_video_name, video_workspace_path)
        print("Loading all AI models. This may take a moment...")
        models = load_models(yolo_model='yolov8l.pt')
        process_video(video_workspace_path, models)