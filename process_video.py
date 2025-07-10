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
from rembg import remove
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
WORKSPACE_DIR = "workspace"
VIDEOS_DIR = os.path.join(WORKSPACE_DIR, "uploaded_videos")
OBJECT_CROPS_DIR = os.path.join(WORKSPACE_DIR, "object_crops")
ANALYSIS_DIR = os.path.join(WORKSPACE_DIR, "analysis_data")
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OBJECT_CROPS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FRAME_INTERVAL = 3  # Reduced from 5 for better detection

# --- Model Loading ---
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yolo_model = YOLO("yolov8l.pt")
    deepsort = DeepSort(
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=(device == "cuda")
    )
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    face_analysis_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analysis_model.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)  # Lower threshold

    return {
        'device': device,
        'yolo': yolo_model,
        'deepsort': deepsort,
        'clip_model': clip_model,
        'clip_processor': clip_processor,
        'face_model': face_analysis_model
    }

def remove_background(image_crop):
    pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    output_pil = remove(pil_image)
    background = Image.new('RGB', output_pil.size, (255, 255, 255))
    background.paste(output_pil, mask=output_pil.split()[3])
    return cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)

def get_appearance_embedding(crop_image, models):
    device = models['device']
    image = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
    inputs = models['clip_processor'](images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = models['clip_model'].get_image_features(**inputs)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()

def get_face_embedding(crop_image, models):
    faces = models['face_model'].get(crop_image)
    return faces[0].normed_embedding if faces else None

def process_video(video_path, models):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
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
        if frame_idx % FRAME_INTERVAL != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        timestamp = frame_idx / fps
        detections = models['yolo'](frame, verbose=False)[0]

        # Visualize detections (optional, for debugging)
        """
        for det in detections.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            class_name = models['yolo'].names[int(det.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(video_crops_dir, f"debug_frame_{frame_idx}.jpg"), frame)
        """

        results_for_deepsort = []
        for det in detections.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            class_name = models['yolo'].names[cls]
            if class_name in ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']:
                xywh = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
                results_for_deepsort.append((xywh, conf, cls))

        outputs = []
        if results_for_deepsort:
            detections = [(xywh, conf, models['yolo'].names[cls]) for xywh, conf, cls in results_for_deepsort]
            tracks = models['deepsort'].update_tracks(detections, frame=frame)
            for track in tracks:
                if track.is_confirmed():
                    bbox = track.to_tlbr()
                    track_id = track.track_id
                    class_id = None
                    for xywh, conf, cls in results_for_deepsort:
                        if abs(bbox[0] - xywh[0]) < 10 and abs(bbox[1] - xywh[1]) < 10:
                            class_id = cls
                            break
                    if class_id is not None:
                        outputs.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id, class_id])

        for track in outputs:
            x1, y1, x2, y2, track_id, cls_id = map(int, track)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            segmented_crop = remove_background(crop)
            crop_filename = f"track_{track_id}_frame_{frame_idx}.jpg"
            crop_path = os.path.join(video_crops_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            # Save segmented crop for debugging
            cv2.imwrite(os.path.join(video_crops_dir, f"segmented_{crop_filename}"), segmented_crop)

            appearance_emb = get_appearance_embedding(segmented_crop, models)
            class_name = models['yolo'].names[cls_id]
            face_emb = get_face_embedding(crop, models) if class_name == 'person' else None

            # Debug embedding consistency
            if frame_idx > 0:
                prev_obj = next((o for o in all_objects_data if o['track_id'] == f"{video_name}_track_{track_id}" and o['timestamp'] != timestamp), None)
                if prev_obj and appearance_emb is not None:
                    sim = cosine_similarity(appearance_emb.reshape(1, -1), prev_obj['appearance_embedding'].reshape(1, -1))[0][0]
                    print(f"Track {track_id} similarity to previous: {sim:.2f}")

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

    print(f"Saving analysis data to {analysis_file_path}...")
    with open(analysis_file_path, 'wb') as f:
        pickle.dump(all_objects_data, f)
    print("Processing complete.")

if __name__ == "__main__":
    from pathlib import Path
    video_path = Path('workspace/uploaded_videos/crowd_vedio2.mp4')  # Update path as needed
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
    else:
        print("Loading models...")
        models = load_models()
        process_video(video_path, models)