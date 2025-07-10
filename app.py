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

# --- Model Loading (Cached for UI performance) ---
@st.cache_resource
def load_search_models():
    """Loads models needed for generating query embeddings."""
    st.write("Cache miss: Loading search models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for search: {device}")

    # CLIP for Text & Image Queries
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # InsightFace for Face Queries
    face_analysis_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analysis_model.prepare(ctx_id=0, det_size=(640, 640))

    models = {
        'device': device,
        'clip_model': clip_model.eval(), # Set to evaluation mode
        'clip_processor': clip_processor,
        'face_model': face_analysis_model
    }
    st.success("Search models loaded.")
    return models

# --- Data Loading ---
@st.cache_data
def load_analysis_data(analysis_file_path):
    """Loads the pre-processed analysis data for a video."""
    st.write(f"Cache miss: Loading analysis data from {analysis_file_path}...")
    if not os.path.exists(analysis_file_path):
        st.error(f"Analysis file not found! Please process the video first using 'process_video.py'.")
        return None
    with open(analysis_file_path, 'rb') as f:
        data = pickle.load(f)
    st.success(f"Loaded {len(data)} object instances.")
    return data


def get_query_embedding(models, text=None, image=None, face_image=None):
    """Generates an embedding for the user's query."""
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
        cv2_image = np.array(pil_image)[:, :, ::-1] # Convert to BGR for insightface
        faces = models['face_model'].get(cv2_image)
        if faces and len(faces) > 0:
            return faces[0].normed_embedding # Return as numpy array directly
        else:
            return None # No face found in query image

    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()



# The function signature will now accept the models dictionary
def search_objects(query_embedding, analysis_data, models, text_query=None, search_type='appearance', top_k=12):
    """
    Searches through the analysis data, with an optional re-scoring step for text queries.
    """
    if query_embedding is None: return []

    candidate_objects = []
    for obj in analysis_data:
        if search_type == 'appearance':
            candidate_objects.append(obj)
        elif search_type == 'face' and obj.get('face_embedding') is not None:
            candidate_objects.append(obj)
    
    if not candidate_objects:
        return []

    # Extract the relevant embeddings from our candidates
    embedding_key = 'appearance_embedding' if search_type == 'appearance' else 'face_embedding'


    
    
    candidate_embeddings = np.array([obj[embedding_key] for obj in candidate_objects])
    similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
    
    # --- KEY IMPROVEMENT: Re-scoring for Text Queries ---
    if text_query and search_type == 'appearance':
        print("Performing re-scoring with negative prompt...")
        # Define a generic negative prompt
        negative_prompt = "a photo of an empty background, a logo, a blurry image, a pattern"
        negative_embedding = get_query_embedding(models, text=negative_prompt)
        
        # Calculate similarity to the negative prompt
        negative_similarities = cosine_similarity(negative_embedding.reshape(1, -1), candidate_embeddings).flatten()
        
        # Adjust the original scores. We penalize images that are also similar to the negative prompt.
        # The 0.5 is a weight, can be tuned.
        final_scores = similarities - (0.5 * negative_similarities)
    else:
        final_scores = similarities

    # Now use `final_scores` for ranking instead of `similarities`
    top_indices = np.argsort(final_scores)[::-1][:top_k*5] 
    
    # ... (the de-duplication part remains the same, but uses final_scores) ...
    best_results_by_track = {}
    for i in top_indices:
        obj = candidate_objects[i]
        track_id = obj['track_id']
        score = final_scores[i] # Use the new score
        
        if track_id not in best_results_by_track or score > best_results_by_track[track_id]['score']:
            # Also store the original similarity for display, as the new score is not intuitive (can be < 0)
            best_results_by_track[track_id] = {'object': obj, 'score': similarities[i]}
    
    # Sort by the ORIGINAL score for user display
    sorted_unique_results = sorted(best_results_by_track.values(), key=lambda x: x['score'], reverse=True)
    
    return sorted_unique_results[:top_k]

# --- Main Application UI ---
st.title("👁️ Forensic V-Search")
st.markdown("An interactive tool to search pre-processed videos using text, images, or faces.")

# --- Load Models ---
models = load_search_models()

# --- UI: Video Selection ---
st.header("1. Select Processed Video")
processed_files = [f for f in os.listdir(ANALYSIS_DIR) if f.endswith("_analysis.pkl")]
if not processed_files:
    st.warning("No processed videos found! Please run 'process_video.py' first.")
else:
    selected_file = st.selectbox("Choose a video analysis file:", processed_files)
    
    # Load data for the selected video
    analysis_data = load_analysis_data(os.path.join(ANALYSIS_DIR, selected_file))
    
    if analysis_data:
        # --- UI: Search Tabs ---
        st.header("2. Perform Search")
        tab1, tab2, tab3 = st.tabs(["📝 Text Search", "🖼️ Image Similarity Search", "🧑 Face Recognition Search"])

        # --- Text Search Tab ---
        with tab1:
            text_query = st.text_input("Enter your free-text query:", "a person wearing a red shirt")
            if st.button("Search by Text", key="text_search_btn"):
                with st.spinner("Generating text embedding and searching..."):
                    query_emb = get_query_embedding(models, text=text_query)
                    results = search_objects(query_emb, analysis_data, models, text_query=text_query, search_type='appearance')

                    # results = search_objects(query_emb, analysis_data, search_type='appearance')
                
                st.subheader(f"Top Results for: '{text_query}'")
                if not results:
                    st.write("No matches found.")
                else:
                    cols = st.columns(4)
                    for i, res in enumerate(results):
                        with cols[i % 4]:
                            st.image(res['object']['object_crop_path'], caption=f"Score: {res['score']:.2f}", use_column_width=True)

        # --- Image Similarity Search Tab ---
        with tab2:
            image_query = st.file_uploader("Upload an image of an object (person, car, etc.)", type=['jpg', 'jpeg', 'png'], key="img_sim_uploader")
            if image_query:
                st.image(image_query, "Your query image:", width=150)
                if st.button("Search by Image", key="img_search_btn"):
                    with st.spinner("Generating image embedding and searching..."):
                        query_emb = get_query_embedding(models, image=image_query)
                        results = search_objects(query_emb, analysis_data, models, search_type='...')
                    
                    st.subheader(f"Top Similarity Results:")
                    if not results:
                        st.write("No matches found.")
                    else:
                        cols = st.columns(4)
                        for i, res in enumerate(results):
                            with cols[i % 4]:
                                st.image(res['object']['object_crop_path'], caption=f"Score: {res['score']:.2f}", use_column_width=True)
        
        # --- Face Recognition Search Tab ---
        with tab3:
            face_query = st.file_uploader("Upload a clear image of a face", type=['jpg', 'jpeg', 'png'], key="face_uploader")
            if face_query:
                st.image(face_query, "Your query face:", width=150)
                if st.button("Search by Face", key="face_search_btn"):
                    with st.spinner("Generating face embedding and searching..."):
                        query_emb = get_query_embedding(models, face_image=face_query)
                        if query_emb is None:
                            st.error("No face detected in the uploaded image. Please try another one.")
                        else:
                            results = search_objects(query_emb, analysis_data, search_type='...')

                            st.subheader(f"Top Face Recognition Results:")
                            if not results:
                                st.write("No matching faces found in the video.")
                            else:
                                cols = st.columns(4)
                                for i, res in enumerate(results):
                                    with cols[i % 4]:
                                        st.image(res['object']['object_crop_path'], caption=f"Score: {res['score']:.2f}", use_column_width=True)


