# Entity-Detection
# Forensic V-Search: AI-Powered Video Search Engine

Forensic V-Search is a powerful, local-first computer vision tool that allows you to perform deep, content-based searches on your video files. Go beyond simple metadata and search for what's *inside* your videos using natural language, example images, or even faces.

This project uses a sophisticated AI pipeline to analyze videos offline, creating a searchable index of every object and person. You can then use the interactive web interface to find specific moments in seconds.

<!-- TODO: Replace with a real screenshot of your app -->

## Features

- **Text-to-Video Search:** Find moments in a video using natural language queries (e.g., "a person in a blue shirt walking a dog").
- **Image Similarity Search:** Find objects (people, vehicles) that look visually similar to an image you provide.
- **Face Recognition Search:** Upload a photo of a person's face to find all instances of that person in the video.
- **Object Tracking:** Each object is tracked across frames, so you get unique results for each individual person or vehicle.
- **Interactive UI:** A simple and intuitive web interface built with Streamlit for easy searching and viewing of results.

---

## How It Works: The Project Workflow

The project is split into two main components: an **Offline Processing Engine** and an **Interactive Search UI**.

### 1. The Processing Engine (`process_video.py`)

This is the heavy-lifting part. For each video you want to analyze, this script performs a deep, frame-by-frame analysis:
- **Object Detection:** A YOLOv8 model identifies all objects like people, cars, and bicycles in every frame.
- **Object Tracking:** The DeepSORT algorithm tracks each detected object, assigning it a unique ID as it moves through the video.
- **Data Extraction:** Each tracked object is cropped from its frame.
- **AI "Fingerprinting" (Embedding Generation):**
- An **Appearance Embedding** is created for each object crop using OpenAI's CLIP model. This captures the object's visual essence.
- A **Face Embedding** is created for every detected face using the ArcFace model. This is a highly accurate facial signature.
- **Indexing:** All this extracted data—track IDs, timestamps, bounding boxes, and AI embeddings—is saved to a single, efficient `.pkl` analysis file in the `workspace/analysis_data/` directory.

### 2. The Search UI (`app.py`)

This is the interactive Streamlit application that you run to search your processed videos.
- **Load Data:** You select a processed video, and the app loads its corresponding analysis file into memory.
- **Process Query:** When you type a text query or upload an image:
- The same AI models (CLIP, ArcFace) are used to convert your query into a compatible embedding.
- **Vector Search:** The app performs a high-speed vector similarity search, comparing your query's embedding against all the object embeddings stored in the analysis file.
- **Display Results:** The top-matching objects are displayed in a clean gallery, complete with their confidence score and the exact timestamp of their appearance in the original video.

---

## Getting Started: Setup & Installation

Follow these steps to set up the project environment and run the application on your local machine.

### Prerequisites

- An NVIDIA GPU with CUDA installed is **highly recommended** for acceptable processing speed. The project will run on a CPU, but it will be very slow.
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
- [Git](https://git-scm.com/) installed.

### 1. Clone the Repository

First, clone this repository to your local machine.

```bash
git clone https://github.com/RisAhamed/Entity-Detection
cd Entity-Detection
 ```


## Step 1: Create the virtual environment

Run this command in your terminal or command prompt:

```
python -m venv vence
```
On Windows (CMD)
```
vence\Scripts\activate.bat
```
On Linux or macOS (bash/zsh)
```
source vence/bin/activate
```
2. Create and Activate the Conda Environment
We will create a dedicated Conda environment to keep all dependencies isolated. The requirements.txt file specifies all the necessary packages. Create a new conda environment named 'vsearch' with Python 3.10
```
conda create --name vsearch python=3.10 -y
```
 Activate the newly created environment
 ```
conda activate vsearch
 ```
 
3. Install Dependencies
Now, install all the required Python packages using pip and the provided requirements.txt file.

Important: The PyTorch installation command depends on your CUDA version. Visit the PyTorch website to get the correct command for your system.

Example for CUDA 11.8 (REPLACE if your version is different)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other packages from requirements.txt
pip install -r requirements.txt
```
(Note: The first time you run the scripts, some models like InsightFace and YOLO will automatically download their pre-trained weights, which may take a few moments.)


How to Use the Application
Using the application is a two-step process:

Step 1: Process a Video (One-time)
Place the video file you want to analyze (e.g., my_video.mp4) inside the project's root directory.
Open the process_video.py script and change the target_video_name variable to match your video's filename.

Generated python# In process_video.py
target_video_name = "my_video.mp4"
 

Run the processing script from your terminal (make sure your vsearch conda environment is active):

```
python process_video.py
 ```
 
 
This will take time. Once complete, you will find a new _analysis.pkl file in the workspace/analysis_data/ directory.
Step 2: Launch the Search Interface
Once you have at least one processed video, launch the Streamlit application:

```
streamlit run app.py
 ```
 

Your web browser will open with the search interface.
Select the processed video from the dropdown menu.
Use the tabs to search by text, image, or face.
Enjoy exploring your video content like never before!


