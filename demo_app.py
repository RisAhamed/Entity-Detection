import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# --- Configuration & Path Handling (CRUCIAL FIX) ---
st.set_page_config(layout="wide", page_title="Forensic V-Search")

# Use Pathlib to get the directory of the current script
# This makes all paths relative to the script's location, not the current working directory
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR / "workspace"
ANALYSIS_DIR = WORKSPACE_DIR / "analysis_data"

# --- Model Loading (No changes needed here) ---
@st.cache_resource
def load_search_models():
    # To make it cleaner for users, let's hide the loading messages in the terminal
    # and only show Streamlit's messages.
    st.write("Loading search models (this may take a moment on first run)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    face_analysis_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analysis_model.prepare(ctx_id=0, det_size=(640, 640))
    
    st.success(f"Search models loaded successfully on {device.upper()}.")
    return {
        'device': device,
        'clip_model': clip_model,
        'clip_processor': clip_processor,
        'face_model': face_analysis_model
    }

# --- Data Loading (No changes needed here) ---
@st.cache_data
def load_analysis_data(analysis_file_path):
    st.write(f"Loading analysis data from '{os.path.basename(analysis_file_path)}'...")
    if not os.path.exists(analysis_file_path):
        st.error(f"Analysis file not found at {analysis_file_path}! Please process the video first using 'process_video.py'.")
        return None
    with open(analysis_file_path, 'rb') as f:
        data = pickle.load(f)
    st.success(f"Loaded {len(data)} object instances.")
    return data

# --- Query Embedding (No changes needed here) ---
def get_query_embedding(models, text=None, image=None, face_image=None):
    device = models['device']
    if text:
        inputs = models['clip_processor'](text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = models['clip_model'].get_text_features(**inputs)
    elif image:
        pil_image = Image.open(image).convert("RGB")
        inputs = models['clip_processor'](images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = models['clip_model'].get_image_features(**inputs)
    elif face_image:
        pil_image = Image.open(face_image).convert("RGB")
        cv2_image = np.array(pil_image)[:, :, ::-1]
        faces = models['face_model'].get(cv2_image)
        return faces[0].normed_embedding if faces else None
    
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()

# --- Search Function (Added a minimum score threshold) ---
def search_objects(query_embedding, analysis_data, models, text_query=None, search_type='appearance', top_k=5, min_score=0.2): # Added min_score
    if query_embedding is None:
        return []

    candidate_objects = [obj for obj in analysis_data if search_type == 'appearance' or (search_type == 'face' and obj.get('face_embedding') is not None)]
    if not candidate_objects:
        return []

    embedding_key = 'appearance_embedding' if search_type == 'appearance' else 'face_embedding'
    candidate_embeddings = np.array([obj[embedding_key] for obj in candidate_objects])
    similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
    
    # Use the same scores for ranking and filtering to avoid confusion
    final_scores = similarities

    # Get all indices above the minimum score threshold
    top_indices = np.argsort(final_scores)[::-1]
    filtered_indices = [i for i in top_indices if final_scores[i] >= min_score]

    best_results_by_track = {}
    for i in filtered_indices:
        obj = candidate_objects[i]
        track_id = obj['track_id']
        score = final_scores[i]
        if track_id not in best_results_by_track or score > best_results_by_track[track_id]['score']:
            best_results_by_track[track_id] = {'object': obj, 'score': score}

    sorted_unique_results = sorted(best_results_by_track.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    return sorted_unique_results

# --- Main UI ---
st.title("👁️ Forensic V-Search")
st.markdown("An interactive tool to search pre-processed videos using natural language, example images, or faces.")

models = load_search_models()

# --- UI: Video Selection (CRUCIAL FIX) ---
st.header("1. Select Processed Video")

# Automatically find all analysis files
try:
    processed_files = [f.name for f in ANALYSIS_DIR.iterdir() if f.name.endswith("_analysis.pkl")]
except FileNotFoundError:
    processed_files = []

if not processed_files:
    st.warning(f"No processed videos found in '{ANALYSIS_DIR}'. Please run 'process_video.py' first.")
else:
    selected_file_name = st.selectbox("Choose a video analysis file:", processed_files)
    
    # Construct the full, absolute path to the selected file
    selected_file_path = ANALYSIS_DIR / selected_file_name
    analysis_data = load_analysis_data(selected_file_path)

    if analysis_data:
        st.header("2. Perform Search")
        # Add a slider for the user to control the confidence threshold
        score_threshold = st.slider("Minimum Confidence Score:", 0.1, 1.0, 0.3, 0.05)
        
        tab1, tab2, tab3 = st.tabs(["📝 Text Search", "🖼️ Image Similarity Search", "🧑 Face Recognition Search"])

        def display_results(results):
            if not results:
                st.write("No matches found above the selected confidence score.")
            else:
                st.write(f"Found {len(results)} unique objects matching the criteria.")
                cols = st.columns(4)
                for i, res in enumerate(results):
                    with cols[i % 4]:
                        # --- CRUCIAL FIX: Construct absolute path for st.image ---
                        # The path in the .pkl is relative to the workspace, so we join it with the SCRIPT_DIR
                        image_path = SCRIPT_DIR / res['object']['object_crop_path']
                        if os.path.exists(image_path):
                            st.image(str(image_path), caption=f"Score: {res['score']:.2f}", use_container_width=True) # FIX: use_container_width
                        else:
                            st.warning(f"Img not found: {image_path}")

        with tab1:
            text_query = st.text_input("Enter your query:", "a person wearing a blue shirt")
            if st.button("Search by Text", key="text_search_btn"):
                with st.spinner("Searching..."):
                    query_emb = get_query_embedding(models, text=text_query)
                    results = search_objects(query_emb, analysis_data, models, min_score=score_threshold, search_type='appearance')
                st.subheader(f"Top Results for: '{text_query}'")
                display_results(results)

        with tab2:
            image_query = st.file_uploader("Upload an image of an object:", type=['jpg', 'jpeg', 'png'], key="img_sim_uploader")
            if image_query:
                st.image(image_query, "Query image:", width=150)
                if st.button("Search by Image", key="img_search_btn"):
                    with st.spinner("Searching..."):
                        query_emb = get_query_embedding(models, image=image_query)
                        results = search_objects(query_emb, analysis_data, models, min_score=score_threshold, search_type='appearance')
                    st.subheader("Top Similarity Results:")
                    display_results(results)

        with tab3:
            face_query = st.file_uploader("Upload a face image:", type=['jpg', 'jpeg', 'png'], key="face_uploader")
            if face_query:
                st.image(face_query, "Query face:", width=150)
                if st.button("Search by Face", key="face_search_btn"):
                    with st.spinner("Searching..."):
                        query_emb = get_query_embedding(models, face_image=face_query)
                        if query_emb is None:
                            st.error("No face detected in the uploaded image.")
                        else:
                            # Face recognition scores are different, a higher threshold is usually needed
                            face_score_threshold = st.slider("Face Match Threshold:", 0.1, 1.0, 0.4, 0.05, key="face_thresh")
                            results = search_objects(query_emb, analysis_data, models, min_score=face_score_threshold, search_type='face')
                            st.subheader("Top Face Recognition Results:")
                            display_results(results)