import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import insightface
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Forensic V-Search")
WORKSPACE_DIR = "workspace"
ANALYSIS_DIR = os.path.join(WORKSPACE_DIR, "analysis_data")

# --- Model Loading ---
@st.cache_resource
def load_search_models():
    st.write("Loading search models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    face_analysis_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analysis_model.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    st.success("Search models loaded.")
    return {
        'device': device,
        'clip_model': clip_model,
        'clip_processor': clip_processor,
        'face_model': face_analysis_model
    }

# --- Data Loading ---
@st.cache_data
def load_analysis_data(analysis_file_path):
    st.write(f"Loading analysis data from {analysis_file_path}...")
    if not os.path.exists(analysis_file_path):
        st.error("Analysis file not found! Process the video first.")
        return None
    with open(analysis_file_path, 'rb') as f:
        data = pickle.load(f)
    st.success(f"Loaded {len(data)} object instances.")
    return data

# --- Query Embedding ---
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

# --- Search Function ---
def search_objects(query_embedding, analysis_data, models, text_query=None, search_type='appearance', top_k=12):
    if query_embedding is None:
        return []

    candidate_objects = [obj for obj in analysis_data if search_type == 'appearance' or (search_type == 'face' and obj.get('face_embedding') is not None)]
    if not candidate_objects:
        return []

    embedding_key = 'appearance_embedding' if search_type == 'appearance' else 'face_embedding'
    candidate_embeddings = np.array([obj[embedding_key] for obj in candidate_objects])
    similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()

    if text_query and search_type == 'appearance':
        negative_prompt = "a photo of an empty background, a logo, a blurry image, a pattern"
        negative_embedding = get_query_embedding(models, text=negative_prompt)
        negative_similarities = cosine_similarity(negative_embedding.reshape(1, -1), candidate_embeddings).flatten()
        final_scores = similarities - (0.5 * negative_similarities)
    else:
        final_scores = similarities

    top_indices = np.argsort(final_scores)[::-1]
    filtered_indices = [i for i in top_indices if final_scores[i] > 0.7][:top_k*5]  # Added threshold

    best_results_by_track = {}
    for i in filtered_indices:
        obj = candidate_objects[i]
        track_id = obj['track_id']
        score = similarities[i]
        if track_id not in best_results_by_track or score > best_results_by_track[track_id]['score']:
            best_results_by_track[track_id] = {'object': obj, 'score': score}

    sorted_unique_results = sorted(best_results_by_track.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    return sorted_unique_results

# --- Main UI ---
st.title("👁️ Forensic V-Search")
st.markdown("Search pre-processed videos using text, images, or faces.")

models = load_search_models()
processed_files = 'workspace/analysis_data/crowd_vedio2_analysis.pkl'
if not processed_files:
    st.warning("No processed videos found! Run 'process_video.py' first.")
else:
    selected_file = st.selectbox("Choose a video analysis file:", processed_files)
    analysis_data = load_analysis_data('EntityDetection/Entity-Detection/workspace/analysis_data/crowd_vedio2_analysis.pkl')

    if analysis_data:
        st.header("Perform Search")
        tab1, tab2, tab3 = st.tabs(["📝 Text Search", "🖼️ Image Similarity Search", "🧑 Face Recognition Search"])

        with tab1:
            text_query = st.text_input("Enter your query:", "a person wearing a red shirt")
            if st.button("Search by Text", key="text_search_btn"):
                with st.spinner("Searching..."):
                    query_emb = get_query_embedding(models, text=text_query)
                    results = search_objects(query_emb, analysis_data, models, text_query=text_query, search_type='appearance')
                st.subheader(f"Top Results for: '{text_query}'")
                if not results:
                    st.write("No matches found.")
                else:
                    st.write(f"Top match score: {results[0]['score']:.2f}")
                    cols = st.columns(4)
                    for i, res in enumerate(results):
                        with cols[i % 4]:
                            st.image(res['object']['object_crop_path'], caption=f"Score: {res['score']:.2f}", use_column_width=True)

        with tab2:
            image_query = st.file_uploader("Upload an image of an object:", type=['jpg', 'jpeg', 'png'], key="img_sim_uploader")
            if image_query:
                st.image(image_query, "Query image:", width=150)
                if st.button("Search by Image", key="img_search_btn"):
                    with st.spinner("Searching..."):
                        query_emb = get_query_embedding(models, image=image_query)
                        results = search_objects(query_emb, analysis_data, models, search_type='appearance')
                    st.subheader("Top Similarity Results:")
                    if not results:
                        st.write("No matches found.")
                    else:
                        st.write(f"Top match score: {results[0]['score']:.2f}")
                        cols = st.columns(4)
                        for i, res in enumerate(results):
                            with cols[i % 4]:
                                st.image(res['object']['object_crop_path'], caption=f"Score: {res['score']:.2f}", use_column_width=True)

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
                            results = search_objects(query_emb, analysis_data, models, search_type='face')
                            st.subheader("Top Face Recognition Results:")
                            if not results:
                                st.write("No matching faces found.")
                            else:
                                st.write(f"Top match score: {results[0]['score']:.2f}")
                                cols = st.columns(4)
                                for i, res in enumerate(results):
                                    with cols[i % 4]:
                                        st.image(res['object']['object_crop_path'], caption=f"Score: {res['score']:.2f}", use_column_width=True)