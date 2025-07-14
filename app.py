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

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Forensic V-Search")
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR / "workspace"
ANALYSIS_DIR = WORKSPACE_DIR / "analysis_data"

# --- Model Loading ---
@st.cache_resource
def load_search_models():
    st.write("Loading search models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    face_analysis_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analysis_model.prepare(ctx_id=0, det_size=(640, 640))
    st.success(f"Search models loaded on {device.upper()}.")
    return {
        'device': device, 'clip_model': clip_model, 'clip_processor': clip_processor, 'face_model': face_analysis_model
    }

# --- Data Loading ---
@st.cache_data
def load_analysis_data(selected_file_paths):
    all_data = []
    for file_path in selected_file_paths:
        st.write(f"Loading analysis data from '{os.path.basename(file_path)}'...")
        if not os.path.exists(file_path):
            st.error(f"Analysis file not found at {file_path}!")
            continue
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    st.success(f"Loaded {len(all_data)} object instances from {len(selected_file_paths)} files.")
    return all_data

# --- Helper Function ---
def format_timestamp(seconds):
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
    if not text_query:
        return []
    prompts = [f"a photo of {text_query}", "a photo of a person wearing different clothes", "a photo of a car or a truck",
               "a photo of a building or a tree", "a blurry photo or a logo", "a pole having flag or boards",
               "a person having dogs", "a vehicle carrying people"]
    device = models['device']
    text_inputs = models['clip_processor'](text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = models['clip_model'].get_text_features(**text_inputs)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    
    candidate_objects = [obj for obj in analysis_data]
    if not candidate_objects:
        return []
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
    return sorted(best_results_by_track.values(), key=lambda x: x['score'], reverse=True)[:top_k]

def search_objects(query_embedding, analysis_data, models, search_type='appearance', top_k=12, min_score=0.2):
    if query_embedding is None:
        return []
    candidate_objects = [obj for obj in analysis_data if search_type == 'appearance' or (search_type == 'face' and obj.get('face_embedding') is not None)]
    if not candidate_objects:
        return []
    embedding_key = 'appearance_embedding' if search_type == 'appearance' else 'face_embedding'
    candidate_embeddings = np.array([obj[embedding_key] for obj in candidate_objects])
    similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
    top_indices = np.argsort(similarities)[::-1]
    filtered_indices = [i for i in top_indices if similarities[i] >= min_score]
    best_results_by_track = {}
    for i in filtered_indices:
        obj = candidate_objects[i]
        track_id = obj['track_id']
        score = similarities[i]
        if track_id not in best_results_by_track or score > best_results_by_track[track_id]['score']:
            best_results_by_track[track_id] = {'object': obj, 'score': score}
    return sorted(best_results_by_track.values(), key=lambda x: x['score'], reverse=True)[:top_k]

# --- Main UI ---
st.title("👁️ Forensic V-Search")
st.markdown("Search pre-processed videos using text, images, or faces.")

models = load_search_models()

st.header("1. Select Processed Videos")
try:
    processed_files = [f.name for f in ANALYSIS_DIR.iterdir() if f.name.endswith("_analysis.pkl")]
except FileNotFoundError:
    processed_files = []

if not processed_files:
    st.warning(f"No processed videos found in '{ANALYSIS_DIR}'. Run 'main.py' first.")
else:
    selected_file_names = st.multiselect("Choose video analysis files:", processed_files, default=processed_files[:1])
    selected_file_paths = [ANALYSIS_DIR / fname for fname in selected_file_names]
    analysis_data = load_analysis_data(selected_file_paths)

    if analysis_data:
        st.header("2. Perform Search")
        
        def display_results(results, title="Search Results"):
            if not results:
                st.write("No matches found.")
            else:
                st.write(f"Found {len(results)} unique objects.")
                for i, res in enumerate(results):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        image_path = SCRIPT_DIR / res['object']['object_crop_path']
                        if os.path.exists(image_path):
                            st.image(str(image_path), use_container_width=True)
                        else:
                            st.warning("Image not found")
                    with col2:
                        st.metric(label="Confidence Score", value=f"{res['score']:.2%}")
                        ts = res['object']['timestamp']
                        st.markdown(f"**Timestamp:** {format_timestamp(ts)} ({ts:.2f}s)")
                        st.markdown(f"**Track ID:** `{res['object']['track_id']}`")
                        st.markdown(f"**Video:** {res['object']['video_name']}")
                    st.divider()

        tab1, tab2, tab3 = st.tabs(["📝 Text Search", "🖼️ Image Search", "🧑 Face Search"])

        with tab1:
            text_query = st.text_input("Enter query:", "a person wearing a blue shirt")
            text_score_threshold = st.slider("Min Confidence:", 0.0, 1.0, 0.5, key="text_thresh")
            if st.button("Search by Text"):
                with st.spinner("Searching..."):
                    results = search_by_text_zero_shot(text_query, analysis_data, models)
                    filtered_results = [r for r in results if r['score'] >= text_score_threshold]
                    st.session_state.first_search_results = filtered_results
                st.subheader(f"Results for: '{text_query}'")
                display_results(filtered_results)

        with tab2:
            image_query = st.file_uploader("Upload object image:", type=['jpg', 'jpeg', 'png'], key="img_sim")
            img_score_threshold = st.slider("Min Confidence:", 0.1, 1.0, 0.3, key="img_thresh")
            if image_query and st.button("Search by Image"):
                with st.spinner("Searching..."):
                    query_emb = get_query_embedding(models, image=image_query)
                    results = search_objects(query_emb, analysis_data, models, search_type='appearance', min_score=img_score_threshold)
                    st.session_state.first_search_results = results
                st.subheader("Image Similarity Results:")
                display_results(results)

        with tab3:
            face_query = st.file_uploader("Upload face image:", type=['jpg', 'jpeg', 'png'], key="face")
            face_score_threshold = st.slider("Min Confidence:", 0.1, 1.0, 0.4, key="face_thresh")
            if face_query and st.button("Search by Face"):
                with st.spinner("Searching..."):
                    query_emb = get_query_embedding(models, face_image=face_query)
                    if query_emb is None:
                        st.error("No face detected.")
                    else:
                        results = search_objects(query_emb, analysis_data, models, search_type='face', min_score=face_score_threshold)
                        st.session_state.first_search_results = results
                st.subheader("Face Recognition Results:")
                display_results(results)

        # --- Layered Search ---
        if 'first_search_results' in st.session_state and st.session_state.first_search_results:
            st.header("3. Secondary Search on Results")
            secondary_search_type = st.selectbox("Choose secondary search type:", ["Text", "Image", "Face"])
            if secondary_search_type == "Text":
                sec_text_query = st.text_input("Enter secondary text query:", key="sec_text")
                sec_text_thresh = st.slider("Min Confidence:", 0.0, 1.0, 0.5, key="sec_text_thresh")
                if st.button("Secondary Text Search"):
                    with st.spinner("Searching..."):
                        sec_results = search_by_text_zero_shot(sec_text_query, [r['object'] for r in st.session_state.first_search_results], models)
                        filtered_sec_results = [r for r in sec_results if r['score'] >= sec_text_thresh]
                    st.subheader(f"Secondary Results for: '{sec_text_query}'")
                    display_results(filtered_sec_results)
            elif secondary_search_type == "Image":
                sec_image_query = st.file_uploader("Upload secondary image:", type=['jpg', 'jpeg', 'png'], key="sec_img")
                sec_img_thresh = st.slider("Min Confidence:", 0.1, 1.0, 0.3, key="sec_img_thresh")
                if sec_image_query and st.button("Secondary Image Search"):
                    with st.spinner("Searching..."):
                        query_emb = get_query_embedding(models, image=sec_image_query)
                        sec_results = search_objects(query_emb, [r['object'] for r in st.session_state.first_search_results], models, search_type='appearance', min_score=sec_img_thresh)
                    st.subheader("Secondary Image Results:")
                    display_results(sec_results)
            elif secondary_search_type == "Face":
                sec_face_query = st.file_uploader("Upload secondary face image:", type=['jpg', 'jpeg', 'png'], key="sec_face")
                sec_face_thresh = st.slider("Min Confidence:", 0.1, 1.0, 0.4, key="sec_face_thresh")
                if sec_face_query and st.button("Secondary Face Search"):
                    with st.spinner("Searching..."):
                        query_emb = get_query_embedding(models, face_image=sec_face_query)
                        if query_emb is None:
                            st.error("No face detected.")
                        else:
                            sec_results = search_objects(query_emb, [r['object'] for r in st.session_state.first_search_results], models, search_type='face', min_score=sec_face_thresh)
                    st.subheader("Secondary Face Results:")
                    display_results(sec_results)