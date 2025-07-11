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

# --- Configuration & Path Handling ---
st.set_page_config(layout="wide", page_title="Forensic V-Search")

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR / "workspace"
ANALYSIS_DIR = WORKSPACE_DIR / "analysis_data"

# --- Model Loading ---
@st.cache_resource
def load_search_models():
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

# --- Data Loading ---
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

# --- Helper Function for Timestamp Formatting ---
def format_timestamp(seconds):
    """Converts seconds into HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# --- Search Functions ---
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

def search_by_text_zero_shot(text_query, analysis_data, models, top_k=12):
    if not text_query: return []
    prompts = [f"a photo of {text_query}", "a photo of a person wearing different clothes", "a photo of a car or a truck", "a photo of a building or a tree", "a blurry photo or a logo", "a pole having flag or Borads",
    "a person having dogs","a vehicel carying people"]
    device = models['device']
    text_inputs = models['clip_processor'](text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = models['clip_model'].get_text_features(**text_inputs)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    
    candidate_objects = [obj for obj in analysis_data]
    if not candidate_objects: return []
    
    image_embeddings = np.array([obj['appearance_embedding'] for obj in candidate_objects])
    image_embeddings = torch.from_numpy(image_embeddings).to(device)

    with torch.no_grad():
        similarity_matrix = (100.0 * image_embeddings @ text_embeddings.T).softmax(dim=-1)
    
    scores_for_main_query = similarity_matrix[:, 0].cpu().numpy()
    top_indices = np.argsort(scores_for_main_query)[::-1]
    
    best_results_by_track = {}
    for i in top_indices:
        obj = candidate_objects[i]
        track_id = obj['track_id']
        score = scores_for_main_query[i]
        if track_id not in best_results_by_track or score > best_results_by_track[track_id]['score']:
            best_results_by_track[track_id] = {'object': obj, 'score': score}

    sorted_unique_results = sorted(best_results_by_track.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    return sorted_unique_results

def search_objects(query_embedding, analysis_data, models, search_type='appearance', top_k=12, min_score=0.2):
    if query_embedding is None: return []

    candidate_objects = [obj for obj in analysis_data if search_type == 'appearance' or (search_type == 'face' and obj.get('face_embedding') is not None)]
    if not candidate_objects: return []

    embedding_key = 'appearance_embedding' if search_type == 'appearance' else 'face_embedding'
    candidate_embeddings = np.array([obj[embedding_key] for obj in candidate_objects])
    similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
    final_scores = similarities
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

st.header("1. Select Processed Video")
try:
    processed_files = [f.name for f in ANALYSIS_DIR.iterdir() if f.name.endswith("_analysis.pkl")]
except FileNotFoundError:
    processed_files = []

if not processed_files:
    st.warning(f"No processed videos found in '{ANALYSIS_DIR}'. Please run 'process_video.py' first.")
else:
    selected_file_name = st.selectbox("Choose a video analysis file:", processed_files)
    selected_file_path = ANALYSIS_DIR / selected_file_name
    analysis_data = load_analysis_data(selected_file_path)

    if analysis_data:
        st.header("2. Perform Search")
        
        # --- NEW: Centralized display function with timestamp ---
        def display_results(results):
            if not results:
                st.write("No matches found above the selected confidence score.")
            else:
                st.write(f"Found {len(results)} unique objects matching the criteria.")
                # Displaying results in a more structured way
                for i, res in enumerate(results):
                    # Create two columns: one for the image, one for the info
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        image_path = SCRIPT_DIR / res['object']['object_crop_path']
                        if os.path.exists(image_path):
                            st.image(str(image_path), use_container_width=True)
                        else:
                            st.warning(f"Img not found")

                    with col2:
                        score = res['score']
                        # The score from zero-shot is a probability (0-1), others are cosine similarity (-1 to 1)
                        # We can display them differently if needed, but for now, showing as is.
                        st.metric(label="Confidence Score", value=f"{score:.2%}") # Display as percentage
                        
                        # --- THE KEY ADDITION ---
                        ts = res['object']['timestamp']
                        st.markdown(f"**Timestamp:** {format_timestamp(ts)} (at {ts:.2f} seconds)")
                        st.markdown(f"**Track ID:** `{res['object']['track_id']}`")

                    st.divider() # Add a line between results

        tab1, tab2, tab3 = st.tabs(["📝 Text Search", "🖼️ Image Similarity Search", "🧑 Face Recognition Search"])

        with tab1:
            text_query = st.text_input("Enter your query:", "a person wearing a blue shirt")
            text_score_threshold = st.slider("Minimum Confidence Score:", 0.0, 1.0, 0.5, 0.05, key="text_thresh", help="Zero-shot scores are probabilities. A higher value means a more confident match.")
            if st.button("Search by Text", key="text_search_btn"):
                with st.spinner("Searching..."):
                    results = search_by_text_zero_shot(text_query, analysis_data, models)
                    # Filter results by the slider value
                    filtered_results = [r for r in results if r['score'] >= text_score_threshold]
                st.subheader(f"Top Results for: '{text_query}'")
                display_results(filtered_results)

        with tab2:
            image_query = st.file_uploader("Upload an image of an object:", type=['jpg', 'jpeg', 'png'], key="img_sim_uploader")
            img_score_threshold = st.slider("Minimum Confidence Score:", 0.1, 1.0, 0.3, 0.05, key="img_thresh")
            if image_query:
                st.image(image_query, "Query image:", width=150)
                if st.button("Search by Image", key="img_search_btn"):
                    with st.spinner("Searching..."):
                        query_emb = get_query_embedding(models, image=image_query)
                        results = search_objects(query_emb, analysis_data, models, min_score=img_score_threshold, search_type='appearance')
                    st.subheader("Top Similarity Results:")
                    display_results(results)

        with tab3:
            face_query = st.file_uploader("Upload a face image:", type=['jpg', 'jpeg', 'png'], key="face_uploader")
            face_score_threshold = st.slider("Face Match Threshold:", 0.1, 1.0, 0.4, 0.05, key="face_thresh")
            if face_query:
                st.image(face_query, "Query face:", width=150)
                if st.button("Search by Face", key="face_search_btn"):
                    with st.spinner("Searching..."):
                        query_emb = get_query_embedding(models, face_image=face_query)
                        if query_emb is None:
                            st.error("No face detected in the uploaded image.")
                        else:
                            results = search_objects(query_emb, analysis_data, models, min_score=face_score_threshold, search_type='face')
                            st.subheader("Top Face Recognition Results:")
                            display_results(results)