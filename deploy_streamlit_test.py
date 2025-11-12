from torch import device
from deploy_streamlit import generate_video_token, get_noa_metadata, load_nuscenes_data, predict
from train_utils import *
from wrap_model import WrapModel
from detect_anything.datasets.utils import ResizeLongestSide, rotation_6d_to_matrix, compute_3d_bbox_vertices, project_to_image, draw_bbox_2d, box_cxcywh_to_xyxy

from PIL import Image
import cv2
import yaml
import os
import torch.nn as nn
import argparse
from box import Box
import random
import torch.nn.functional as F 

import streamlit as st
import numpy as np 

from groundingdino.util.inference import load_model
from groundingdino.util.inference import predict as dino_predict

from torchvision.ops import box_convert
import colorsys
import json
import hashlib
import io
import pickle
from collections import defaultdict
import time
from scipy.optimize import linear_sum_assignment

# --- Global Constants (Unchanged) ---
BOX_TRESHOLD = 0.37
TEXT_TRESHOLD = 0.25

# ... (Previous imports, get_noa_metadata, calculate_iou_3d, parse_args, device, load_config, cfg, generate_image_token, generate_video_token, load_nuscenes_data, get_color_for_label, distributed setup, load_detany3d_model, load_dino_model, sam_trans, BOX_TRESHOLD, TEXT_TRESHOLD, convert_image, crop_hw, preprocess, preprocess_dino, draw_text, predict - all UNCHANGED) ...

# NOTE: The 'predict' function is complex and remains unchanged, as the request only affects the loading logic.
# However, you must ensure that 'decode_bboxes' (not defined in the snippet but used in 'predict') is correctly available via 'from train_utils import *'.

# ----------------------------------------------------------------------
# MODIFIED FUNCTIONS START HERE
# ----------------------------------------------------------------------

# --- Utility to get ALL Pinhole Images (New or Modified Caching Utility) ---
@st.cache_data
def get_all_noa_pinhole_images():
    """Traverses the NOA root directory and collects paths to ALL pinhole images."""
    NOA_ROOT_DIR = './Traces_NOA_structured'
    ALL_PINHOLE_IMAGES = []
    
    if not os.path.isdir(NOA_ROOT_DIR):
        # st.error(f"NOA root directory not found at: {NOA_ROOT_DIR}")
        return ALL_PINHOLE_IMAGES
        
    # Find all pinhole image files using os.walk
    for root, dirs, files in os.walk(NOA_ROOT_DIR):
        if 'pinhole' in root:
            image_files = [f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            for img_file in image_files:
                ALL_PINHOLE_IMAGES.append(os.path.join(root, img_file))
    
    return ALL_PINHOLE_IMAGES

def load_random_nuscenes_image():
    """Function to be called by the button to load a random image (UNCHANGED)."""
    NUSCENES_PKL_PATH = './data/DA3D_pkls/omni3d23daw/nuScenes_val.pkl'
    nuscenes_data = load_nuscenes_data(NUSCENES_PKL_PATH)
    path_replace_rule = ('./data/nuscenes', '/data/nuscenes')

    if not nuscenes_data:
        return None
    
    random_item = random.choice(nuscenes_data)
    absolute_path = random_item['img_path'].replace(path_replace_rule[0], path_replace_rule[1])

    if not os.path.exists(absolute_path):
        st.error(f"Image path from pkl does not exist: {absolute_path}")
        return None
    return Image.open(absolute_path).convert("RGB")

# --- REVISED: load_random_noa_image ---
def load_random_noa_image():
    """
    Function to load a random image from the entire NOA dataset.
    REVISED to use the cached list of ALL pinhole images.
    """
    ALL_PINHOLE_IMAGES = get_all_noa_pinhole_images()

    if not ALL_PINHOLE_IMAGES:
        st.error("No pinhole image files found in NOA structure.")
        return None, None

    try:
        image_path = random.choice(ALL_PINHOLE_IMAGES)
        return Image.open(image_path).convert("RGB"), image_path
    except Exception as e:
        st.error(f"Error loading random NOA image: {e}")
        return None, None

# --- REVISED: find_and_predict_noa (Uses ALL Images) ---
def find_and_predict_noa(text_prompt, box_threshold, text_threshold, iou_threshold, progress_bar):
    """
    Generates a video by sampling a sequence of consecutive images from the 
    FULL list of NOA pinhole images, searching for a scene that contains the target object.
    
    NOTE: The scene search logic is kept for filtering, but the image collection
    will now use all frames if the criteria are met, then sample.
    """
    NOA_ROOT_DIR = './Traces_NOA_structured' 
    if not os.path.isdir(NOA_ROOT_DIR):
        st.error(f"❌ NOA root directory not found at {NOA_ROOT_DIR}. Please check paths.")
        return None

    search_category = text_prompt.strip().split('.')[0].split(' ')[-1].lower()
    if not search_category:
        st.error("Please enter a valid text prompt to search for a scene.")
        return None

    st.info(f"🔍 Searching dynamically for a scene with '{search_category}' objects in NOA...")
    progress_bar.progress(10, text=f"Searching for a matching scene...")

    found_scene_dir = None
    
    # --- Scene Search Logic (Kept to ensure object relevance) ---
    for root, dirs, files in os.walk(NOA_ROOT_DIR):
        if 'BEV.json' in files and 'bb3d' in root:
            ann_path = os.path.join(root, 'BEV.json')
            scene_path = os.path.dirname(os.path.dirname(root)) 
            
            if os.path.exists(ann_path):
                try:
                    with open(ann_path, 'r') as f:
                        annotations = json.load(f)

                    if isinstance(annotations, dict) and "objects" in annotations:
                        annotations = annotations["objects"]
                    
                    if any(search_category in (obj.get("geometry", {}).get("class") or obj.get("label", "")).lower() for obj in annotations):
                        found_scene_dir = scene_path
                        break
                except:
                    continue
        if found_scene_dir:
            break
            
    if not found_scene_dir:
        st.error(f"❌ No scenes found containing '{search_category}'.")
        return None

    # --- Image Collection (REVISED: Load all pinhole images in the found scene) ---
    img_pinhole_dir = os.path.join(found_scene_dir, 'images/pinhole')
    relevant_images = []
    if os.path.isdir(img_pinhole_dir):
        # We collect all frames from all pinhole sensors in this specific scene
        all_img_files = sorted([f for f in os.listdir(img_pinhole_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
        relevant_images = [os.path.join(img_pinhole_dir, f) for f in all_img_files]
    
    if not relevant_images:
        st.error(f"❌ Scene found, but no images in {img_pinhole_dir}.")
        return None
        
    # --- Limit frames and continue with prediction logic ---
    MAX_FRAMES = 40
    if len(relevant_images) > MAX_FRAMES:
        st.info(f"Found {len(relevant_images)} frames in scene {os.path.basename(found_scene_dir)}. Sampling first {MAX_FRAMES}.")
        relevant_images = relevant_images[:MAX_FRAMES]

    st.info(f"📸 Processing scene {os.path.basename(found_scene_dir)} with {len(relevant_images)} frames...")

    # --- Prepare video output (UNCHANGED) ---
    output_video_dir = './exps/videos'
    os.makedirs(output_video_dir, exist_ok=True)
    video_token = generate_video_token(text_prompt)
    output_video_path = os.path.join(output_video_dir, f'noa_{video_token}.avi')

    # --- Initialize safe VideoWriter (UNCHANGED) ---
    first_img = Image.open(relevant_images[0]).convert("RGB")
    width, height = first_img.size
    
    # Using DIVX for reliable AVI output
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))
    
    if not video_writer.isOpened():
        st.warning("Could not initialize VideoWriter with DIVX. Falling back to MP4V.")
        output_video_path = os.path.join(output_video_dir, f'noa_{video_token}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

    if not video_writer.isOpened():
        st.error("❌ Could not initialize video writer with any known codec. Video generation failed.")
        return None
    
    st.info(f"🎞️ Video will be saved to: {output_video_path}")
    
    valid_frames = 0
    for i, img_path in enumerate(relevant_images):
        progress_bar.progress(10 + int(90 * (i + 1) / len(relevant_images)),
                              text=f"Processing frame {i+1}/{len(relevant_images)}: {os.path.basename(img_path)}")
        try:
            # --- CALIBRATION DATA RETRIEVAL (dynamically loads calib based on image path) ---
            _, calibration_data = get_noa_metadata(img_path) 
            st.sidebar.caption(f"Frame {i}: Calib fx={calibration_data['fx']:.0f}, fy={calibration_data['fy']:.0f}")

            img = Image.open(img_path).convert("RGB")
            result_img, _ = predict(np.array(img), text_prompt, box_threshold, text_threshold, iou_threshold)

            if result_img is None:
                continue

            # Convert to uint8, resize, and convert RGB to BGR for OpenCV
            frame = np.clip(result_img * 255, 0, 255).astype(np.uint8)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height))

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            valid_frames += 1

        except Exception as e:
            st.warning(f"⚠️ Skipping {img_path} due to error: {e}")
            continue

    video_writer.release()

    if valid_frames == 0:
        st.error(f"❌ No valid frames generated for '{search_category}'. Try adjusting thresholds.")
        return None

    st.success(f"✅ Video saved to {output_video_path} ({valid_frames} valid frames).")
    return output_video_path

# ... (find_and_predict_nuscenes_video - UNCHANGED) ...

def find_and_predict(img_path_list, text_prompt, box_threshold, text_threshold, iou_threshold, progress_bar):
    """Finds an image in nuScenes based on the prompt and runs prediction (UNCHANGED)."""
    # ... (function body UNCHANGED) ...
    
    # NOTE: The search logic for nuScenes remains as it was designed to find a single
    # image instance based on the category in the text prompt.
    NUSCENES_PKL_PATH = './data/DA3D_pkls/omni3d23daw/nuScenes_val.pkl'
    nuscenes_data = load_nuscenes_data(NUSCENES_PKL_PATH)

    with open('./data/category_meta.json', 'r') as f:
        category_meta = json.load(f)
    category_id_to_name = {i: name for i, name in enumerate(category_meta['thing_classes'])}

    path_replace_rule = ('./data/nuscenes', '/data/nuscenes')
    category_to_images = defaultdict(list)
    if nuscenes_data:
        for item in nuscenes_data:
            if 'obj_list' in item and 'img_path' in item:
                absolute_path = item['img_path'].replace(path_replace_rule[0], path_replace_rule[1])
                if os.path.exists(absolute_path):
                    for obj in item['obj_list']:
                        category_id = obj.get('label')
                        if category_id is not None and category_id in category_id_to_name:
                            category_name = category_id_to_name[category_id]
                            category_to_images[category_name].append(absolute_path)

    if not nuscenes_data:
        st.error("nuScenes data not loaded. Cannot search for an image.")
        return None

    search_category = text_prompt.strip().split('.')[0].split(' ')[-1].lower()
    if not search_category:
        st.error("Please enter a valid text prompt to search for an image.")
        return None

    matched_category = None
    for name in category_to_images.keys():
        if search_category in name.lower() or name.lower() in search_category: 
              matched_category = name
              break
            
    if matched_category is None or not category_to_images[matched_category]:
        st.error(f"No images found for category matching '{search_category}' in the nuScenes validation set.")
        return None

    random_path = random.choice(category_to_images[matched_category])
    
    st.info(f"Found image with '{matched_category}' at: {os.path.basename(random_path)}. Running prediction...")
    progress_bar.progress(50, text="Image found, running prediction...")

    img = Image.open(random_path).convert("RGB")
    
    return predict(np.array(img), text_prompt, box_threshold, text_threshold, iou_threshold)[0]

def find_and_predict_nuscenes_video(text_prompt, box_threshold, text_threshold, iou_threshold, progress_bar):
    """Finds an image sequence in nuScenes based on prompt, runs prediction, and creates a video in AVI format (UNCHANGED)."""
    NUSCENES_PKL_PATH = './data/DA3D_pkls/omni3d23daw/nuScenes_val.pkl'
    nuscenes_data = load_nuscenes_data(NUSCENES_PKL_PATH)

    if not nuscenes_data:
        st.error("nuScenes data not loaded. Cannot search for an image sequence.")
        return None

    with open('./data/category_meta.json', 'r') as f:
        category_meta = json.load(f)
    category_id_to_name = {i: name for i, name in enumerate(category_meta['thing_classes'])}

    search_category = text_prompt.strip().split('.')[0].split(' ')[-1].lower()
    if not search_category:
        st.error("Please enter a valid text prompt to search for an image.")
        return None

    st.info(f"Searching for a sequence with '{search_category}' in nuScenes...")
    progress_bar.progress(10, text=f"Searching for '{search_category}' in nuScenes...")

    scene_to_images = defaultdict(list)
    path_replace_rule = ('./data/nuscenes', '/data/nuscenes')
    for item in nuscenes_data:
        if 'scene_token' in item and 'img_path' in item:
            absolute_path = item['img_path'].replace(path_replace_rule[0], path_replace_rule[1])
            if os.path.exists(absolute_path):
                scene_to_images[item['scene_token']].append(item)

    target_scene_items = []
    for scene_token, items in scene_to_images.items():
        for item in items:
            if any(search_category in category_id_to_name.get(obj.get('label'), '').lower() for obj in item.get('obj_list', [])):
                target_scene_items = sorted(items, key=lambda x: x['timestamp'])
                st.info(f"Found a relevant scene: {scene_token} with {len(target_scene_items)} images.")
                break
        if target_scene_items:
            break

    if not target_scene_items:
        st.error(f"No sequence found containing an image with category '{search_category}' in the nuScenes dataset.")
        return None

    relevant_items = target_scene_items[:50] 
    st.info(f"Generating video for the sequence with {len(relevant_items)} frames...")

    output_video_dir = './exps/deploy/videos'
    os.makedirs(output_video_dir, exist_ok=True)
    video_token = generate_video_token(text_prompt)
    output_video_path = os.path.join(output_video_dir, f'nuscenes_{video_token}.avi')

    first_img = Image.open(relevant_items[0]['img_path'].replace(path_replace_rule[0], path_replace_rule[1]))
    width, height = first_img.size
    
    # Use DIVX codec for reliable AVI writing
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 5, (width, height))
    
    if not video_writer.isOpened():
        st.error("❌ Could not initialize video writer with DIVX. Video generation failed.")
        return None


    for i, item in enumerate(relevant_items):
        progress_text = f"Processing image {i+1}/{len(relevant_items)}"
        progress_bar.progress(10 + int(90 * (i + 1) / len(relevant_items)), text=progress_text)
        img_path = item['img_path'].replace(path_replace_rule[0], path_replace_rule[1])
        img = Image.open(img_path).convert("RGB")
        result_img, _ = predict(np.array(img), text_prompt, box_threshold, text_threshold, iou_threshold)
        if result_img is not None:
            frame = (result_img * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        else:
            st.warning(f"Skipping frame {i+1} due to no detection or error.")


    video_writer.release()

    
    st.info(f"Video saved to {output_video_path}")
    return output_video_path


# ----------------------------------------------------------------------
# STREAMLIT UI LAYOUT (UNCHANGED, but uses the modified functions)
# ----------------------------------------------------------------------
st.set_page_config(page_title="DetAny3D Streamlit Demo", layout="wide")
st.title("DetAny3D: Detect Anything 3D")
st.markdown(f"**Running on Device:** `{device}`") 

# --- Global State Initialization ---
if 'image' not in st.session_state:
    st.session_state.image = None
if 'current_image_path' not in st.session_state:
    st.session_state.current_image_path = None
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = 'nuScenes'


# --- Main Layout Columns ---
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    
    # Dataset Selector
    dataset_choice = st.selectbox("Select Dataset/Mode", ["nuScenes", "NOA"], index=0 if st.session_state.current_dataset == 'nuScenes' else 1)
    st.session_state.current_dataset = dataset_choice
    
    st.markdown("---")
    
    st.markdown("#### 1. Provide an Image")
    uploaded_file = st.file_uploader("Upload your own image", type=["png", "jpg", "jpeg"])
    
    # Handle uploaded file
    if uploaded_file is not None:
        if st.session_state.image is None or uploaded_file.name != getattr(st.session_state, 'uploaded_file_name', None):
            st.session_state.image = Image.open(uploaded_file).convert("RGB")
            st.session_state.current_image_path = None
            st.session_state.uploaded_file_name = uploaded_file.name # Store file name to avoid re-reading
            st.info("Uploaded file loaded.")

    st.markdown("...or load a random one from the selected dataset:")
    
    # Dataset Specific Image Loaders
    if st.session_state.current_dataset == 'nuScenes':
        if st.button("Load Random nuScenes Image", use_container_width=True):
            st.session_state.image = load_random_nuscenes_image()
            st.session_state.current_image_path = None
    elif st.session_state.current_dataset == 'NOA':
        if st.button("Load Random NOA Image (All Pinhole)", use_container_width=True):
            image, path = load_random_noa_image()
            st.session_state.image = image
            st.session_state.current_image_path = path

    # Image Display
    if st.session_state.image:
        st.image(st.session_state.image, caption=f"Current Image ({st.session_state.current_dataset} source: {os.path.basename(st.session_state.current_image_path) if st.session_state.current_image_path else 'Upload/Random'})", use_container_width=True)

    st.markdown("#### 2. Enter a Text Prompt")
    text_input = st.text_input("Text Prompt", "a car . a person", help="Separate different objects with ' . '")

    with st.expander("Detection Thresholds"):
        box_threshold = st.slider("Box Threshold (2D)", 0.0, 1.0, BOX_TRESHOLD, 0.01)
        text_threshold = st.slider("Text Threshold (2D)", 0.0, 1.0, TEXT_TRESHOLD, 0.01)
        iou_threshold = st.slider("3D IoU Threshold", 0.0, 1.0, 0.2, 0.01, help="Filters final 3D bounding boxes based on their predicted Intersection over Union (IoU) score.")

    st.markdown("#### 3. Run Detection")
    submit_btn = st.button("Run Detection on Current Image", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Dataset Specific Search/Video Generation
    with st.expander(f"Advanced: Search {st.session_state.current_dataset} & Generate Video"):
        st.info(f"Searches the **{st.session_state.current_dataset}** dataset for a scene/image matching your prompt and generates a video (.avi/.mp4).")
        
        search_btn = None
        video_btn = None
        
        if st.session_state.current_dataset == 'nuScenes':
            search_btn = st.button("Find & Predict from nuScenes (Single Image)", key="search_nusc_img", use_container_width=True)
            video_btn = st.button("Find & Predict from nuScenes (Video Sequence)", key="search_nusc_video", use_container_width=True)
        
        elif st.session_state.current_dataset == 'NOA':
            # Updated button text to reflect the broader scope
            video_btn = st.button("Find & Predict from NOA (Video Sequence)", key="search_noa_video", use_container_width=True)


with col2:
    st.header("3D Detection Result")
    output_container = st.empty()
    output_container.info("Output will be displayed here. The first run may take a moment to load models.", icon="🖼️")


# --- BUTTON HANDLERS ---

# 1. Run Detection on Current Image
if submit_btn:
    if st.session_state.image is None:
        st.error("Please upload an image or load a random one first.")
    elif not text_input:
        st.error("Please enter a text prompt.")
    else:
        with st.spinner("Running prediction..."):
            result_image, pred_boxes_3d = predict(np.array(st.session_state.image), text_input, box_threshold, text_threshold, iou_threshold, return_boxes=True)
            if result_image is not None:
                output_container.image(result_image, caption="Detection Result", use_container_width=True)
                
                # Display Calibration info for NOA image if available
                if st.session_state.current_image_path and './Traces_NOA_structured' in st.session_state.current_image_path:
                    _, calib_data = get_noa_metadata(st.session_state.current_image_path)
                    st.sidebar.success(f"Loaded Calibration/Annotations for frame. Example Calib: fx={calib_data['fx']:.0f}, fy={calib_data['fy']:.0f}")

# 2. nuScenes Find & Predict (Single Image)
if st.session_state.current_dataset == 'nuScenes' and search_btn:
    if not text_input:
        st.error("Please enter a text prompt to search.")
    else:
        progress_bar = output_container.progress(0, text="Starting nuScenes search...")
        result_image = find_and_predict(None, text_input, box_threshold, text_threshold, iou_threshold, progress_bar) 
        if result_image is not None:
            progress_bar.progress(100, text="Done!")
            output_container.image(result_image, caption="Detection Result from nuScenes Search", use_container_width=True)
        else:
            progress_bar.empty()

# 3. Video Generation
if st.session_state.current_dataset == 'nuScenes' and video_btn: # nuScenes Video
    if not text_input:
        st.error("Please enter a text prompt to search.")
    else:
        progress_bar = output_container.progress(0, text="Starting nuScenes video search and generation...")
        video_path = find_and_predict_nuscenes_video(text_input, box_threshold, text_threshold, iou_threshold, progress_bar)
        if video_path:
            progress_bar.progress(100, text="Video generation complete! Playing...")
            mime_type = "video/mp4" if video_path.endswith('.mp4') else "video/avi"
            output_container.video(video_path) 
            
            st.download_button(
                label=f"Download nuScenes Inference Video ({'MP4' if video_path.endswith('.mp4') else 'AVI'})",
                data=open(video_path, 'rb').read(),
                file_name=os.path.basename(video_path),
                mime=mime_type
            )
        else:
            progress_bar.empty()

elif st.session_state.current_dataset == 'NOA' and video_btn: # NOA Video
    if not text_input:
        st.error("Please enter a text prompt to search.")
    else:
        progress_bar = output_container.progress(0, text="Starting NOA video search and generation...")
        video_path = find_and_predict_noa(text_input, box_threshold, text_threshold, iou_threshold, progress_bar)
        if video_path:
            progress_bar.progress(100, text="Video generation complete! Playing...")
            mime_type = "video/mp4" if video_path.endswith('.mp4') else "video/avi"
            output_container.video(video_path) 
            
            st.download_button(
                label=f"Download NOA Inference Video ({'MP4' if video_path.endswith('.mp4') else 'AVI'})",
                data=open(video_path, 'rb').read(),
                file_name=os.path.basename(video_path),
                mime=mime_type
            )
        else:
            progress_bar.empty()