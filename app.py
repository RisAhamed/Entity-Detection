
import streamlit as st
import os
import cv2
import numpy as np
import torch
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import logging

# --- Setup logging ---
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, "vsearch.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'), # Overwrite log on each run
        logging.StreamHandler()
    ]
)

logging.info("Starting V-Search application V2.0")

# --- 1. Configuration and Setup ---
st.set_page_config(layout="wide", page_title="V-Search Engine")

# Define local directories
WORKSPACE_DIR = "workspace"
VIDEOS_DIR = os.path.join(WORKSPACE_DIR, "uploaded_videos")
KEYFRAMES_DIR = os.path.join(WORKSPACE_DIR, "keyframes")
EMBEDDINGS_DIR = os.path.join(WORKSPACE_DIR, "embeddings")
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(KEYFRAMES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# --- 2. Model and Device Loading (Crucial for Performance) ---
@st.cache_resource
def load_model_and_device():
    """
    Loads the CLIP model, processor, and determines the processing device.
    Uses a more powerful model: 'openai/clip-vit-large-patch14'.
    """
    logging.info("Cache miss: Loading CLIP model and determining device...")
    st.write("Cache miss: Loading a more powerful CLIP model (this may take a minute on first run)...")
    
    # --- IMPROVEMENT 1: Use a more powerful model ---
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # --- IMPROVEMENT 2: Use GPU if available ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    logging.info(f"CLIP model '{model_name}' loaded successfully to device: {device}")
    st.success(f"Model loaded onto device: {device.upper()}")
    return model, processor, device

# --- 3. Core Processing Functions ---

def save_uploaded_file(uploaded_file, target_dir):
    file_path = os.path.join(target_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_keyframes_scenedetect(video_path, threshold=30.0):
    """
    --- IMPROVEMENT 3: Smarter Keyframe Extraction ---
    Extracts keyframes based on significant changes between frames (scene detection).
    This avoids saving many identical frames from static scenes.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keyframe_subfolder = os.path.join(KEYFRAMES_DIR, video_name)
    os.makedirs(keyframe_subfolder, exist_ok=True)
    
    logging.info(f"Starting scene-detect keyframe extraction for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.error("Failed to read video FPS.")
        return None, 0

    prev_frame = None
    saved_frame_count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Always save the first frame
        if prev_frame is None:
            is_new_scene = True
        else:
            # Convert frames to grayscale for faster comparison
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference and its mean
            diff = cv2.absdiff(gray_frame, gray_prev_frame)
            mean_diff = np.mean(diff)
            
            is_new_scene = mean_diff > threshold

        if is_new_scene:
            timestamp_sec = frame_count / fps
            timestamp_str = f"{int(timestamp_sec // 3600):02d}-{int((timestamp_sec % 3600) // 60):02d}-{int(timestamp_sec % 60):02d}"
            keyframe_path = os.path.join(keyframe_subfolder, f"frame_{timestamp_str}.jpg")
            cv2.imwrite(keyframe_path, frame)
            saved_frame_count += 1
            logging.info(f"Scene change detected. Saved keyframe: {keyframe_path} (diff: {mean_diff if prev_frame is not None else 'N/A'})")

        prev_frame = frame
        frame_count += 1

    cap.release()
    logging.info(f"Scene-detect extraction complete: {saved_frame_count} frames saved.")
    return keyframe_subfolder, saved_frame_count

@st.cache_data(show_spinner=False)
def get_embeddings_from_keyframes(_model, _processor, _device, keyframe_folder):
    """
    Generates and saves embeddings for keyframes. Now device-aware.
    """
    video_name = os.path.basename(keyframe_folder)
    embeddings_file_path = os.path.join(EMBEDDINGS_DIR, f"{video_name}_embeddings_large.pkl")

    if os.path.exists(embeddings_file_path):
        logging.info(f"Loading cached embeddings from {embeddings_file_path}")
        with open(embeddings_file_path, "rb") as f:
            return pickle.load(f)

    logging.info(f"Generating new embeddings for {keyframe_folder}")
    image_files = sorted([os.path.join(keyframe_folder, f) for f in os.listdir(keyframe_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    embeddings_data = []
    
    progress_bar = st.progress(0, text="Generating embeddings...")
    for i, img_path in enumerate(image_files):
        try:
            image = Image.open(img_path).convert("RGB")
            # --- Move inputs to the correct device ---
            inputs = _processor(images=image, return_tensors="pt").to(_device)
            with torch.no_grad():
                image_features = _model.get_image_features(**inputs)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embeddings_data.append({"path": img_path, "embedding": image_features.cpu().numpy().flatten()})
            progress_bar.progress((i + 1) / len(image_files), text=f"Generating embeddings: {i+1}/{len(image_files)}")
        except Exception as e:
            logging.warning(f"Could not process image {img_path}: {e}")
    
    progress_bar.empty()
    with open(embeddings_file_path, "wb") as f:
        pickle.dump(embeddings_data, f)
    logging.info(f"Saved {len(embeddings_data)} new embeddings to {embeddings_file_path}")
    return embeddings_data

def get_query_embedding(_model, _processor, _device, text=None, image=None):
    """
    --- IMPROVEMENT 4: Unified & Robust Query Processing ---
    Generates a normalized embedding for either a text or image query.
    """
    if text:
        # For text, we can use templates to make the query more robust
        templates = [f"a photo of {text}", f"a scene with {text}", f"an image of {text}"]
        inputs = _processor(text=templates, return_tensors="pt", padding=True, truncation=True).to(_device)
        with torch.no_grad():
            text_features = _model.get_text_features(**inputs)
        # Average the embeddings from the templates
        text_features = text_features.mean(dim=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()
    elif image:
        query_pil_image = Image.open(image).convert("RGB")
        inputs = _processor(images=query_pil_image, return_tensors="pt").to(_device)
        with torch.no_grad():
            image_features = _model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()
    return None

def find_best_matches(query_embedding, keyframe_embeddings, top_k=6):
    if not keyframe_embeddings: return []
    keyframe_vectors = np.array([d['embedding'] for d in keyframe_embeddings])
    similarities = cosine_similarity(query_embedding.reshape(1, -1), keyframe_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for i in top_indices:
        results.append((keyframe_embeddings[i]['path'], similarities[i]))
        logging.info(f"Match: {os.path.basename(keyframe_embeddings[i]['path'])}, Score: {similarities[i]:.4f}")
    return results

# --- 4. Streamlit User Interface ---

model, processor, device = load_model_and_device()

st.title("🎬 V-Search: Local Video Query Engine (V2)")
st.markdown("This enhanced version uses a more powerful model and smarter processing for better results.")

# --- UI: Video Selection and Processing ---
st.header("Step 1: Select or Upload a Video")
video_files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
selected_video_name = st.selectbox("Choose a video:", ["Upload a new video..."] + video_files)

uploaded_video = None
if selected_video_name == "Upload a new video...":
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.session_state.video_path = save_uploaded_file(uploaded_video, VIDEOS_DIR)
else:
    st.session_state.video_path = os.path.join(VIDEOS_DIR, selected_video_name)

if 'video_path' in st.session_state and st.session_state.video_path:
    st.video(st.session_state.video_path)
    
    st.header("Step 2: Process the Video")
    st.write("Using scene-change detection for smarter keyframe extraction.")
    if st.button("Process Video & Generate Embeddings"):
        with st.spinner("Detecting scenes and extracting keyframes..."):
            keyframe_folder, frame_count = extract_keyframes_scenedetect(st.session_state.video_path)
            if keyframe_folder:
                st.success(f"Extracted {frame_count} keyframes to `{keyframe_folder}`")
                st.session_state.keyframe_folder = keyframe_folder
        
        if 'keyframe_folder' in st.session_state:
            # Note: arguments to cached function must be consistent
            st.session_state.embeddings = get_embeddings_from_keyframes(model, processor, device, st.session_state.keyframe_folder)
            st.success(f"Successfully generated and saved {len(st.session_state.embeddings)} embeddings!")

# --- UI: Search Interface ---
video_name_for_search = os.path.splitext(os.path.basename(st.session_state.get('video_path', '')))[0]
embedding_file_for_search = os.path.join(EMBEDDINGS_DIR, f"{video_name_for_search}_embeddings_large.pkl")

if os.path.exists(embedding_file_for_search):
    st.header(f"Step 3: Search in '{video_name_for_search}'")

    if 'embeddings' not in st.session_state or st.session_state.get('current_video') != video_name_for_search:
        with open(embedding_file_for_search, "rb") as f:
            st.session_state.embeddings = pickle.load(f)
        st.session_state.current_video = video_name_for_search

    tab1, tab2 = st.tabs(["🔍 Text Search", "🖼️ Image Search"])
    with tab1:
        text_query = st.text_input("Enter text query:", "A person wearing a blue shirt")
        if st.button("Search with Text", key="text_btn"):
            if text_query and 'embeddings' in st.session_state:
                with st.spinner("Searching..."):
                    query_emb = get_query_embedding(model, processor, device, text=text_query)
                    results = find_best_matches(query_emb, st.session_state.embeddings, top_k=6)
                    st.subheader("Top Results:")
                    if not results: st.write("No matches found.")
                    else:
                        cols = st.columns(3)
                        for i, (path, score) in enumerate(results):
                            with cols[i % 3]: st.image(path, caption=f"Match: {score:.2f}", use_column_width=True)

    with tab2:
        image_query = st.file_uploader("Upload image query", type=["jpg", "jpeg", "png"], key="img_uploader")
        if image_query:
            st.image(image_query, caption="Query image", width=200)
            if st.button("Search with Image", key="img_btn"):
                with st.spinner("Searching..."):
                    query_emb = get_query_embedding(model, processor, device, image=image_query)
                    results = find_best_matches(query_emb, st.session_state.embeddings, top_k=6)
                    st.subheader("Top Results:")
                    if not results: st.write("No matches found.")
                    else:
                        cols = st.columns(3)
                        for i, (path, score) in enumerate(results):
                            with cols[i % 3]: st.image(path, caption=f"Match: {score:.2f}", use_column_width=True)