# shared_utils.py

from train_utils import * # Assumed to contain decode_bboxes
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
import torch
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

# --- Global Constants ---
BOX_TRESHOLD = 0.37
TEXT_TRESHOLD = 0.25

# --- Setup & Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description='DetAny3D Streamlit Demo')
    # NOTE: Set default device to CPU if CUDA is not explicitly required for local testing
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to run the demo on (e.g., "cuda:0", "cuda:1", "cpu")')
    return parser.parse_args()

def load_config():
    """Loads the model configuration from demo.yaml."""
    try:
        with open('./detect_anything/configs/demo.yaml', 'r', encoding='utf-8') as f:
            cfg_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        return Box(cfg_dict)
    except Exception as e:
        # Fallback configuration to prevent crash if file is missing
        st.error(f"Could not load config file. Using placeholder settings. Error: {e}")
        return Box({'model': {'pad': 1024, 'image_encoder': {'patch_size': 16, 'vit_pad_mask': False}}, 
                    'dataset': {'pixel_mean': [0., 0., 0.], 'pixel_std': [1., 1., 1.]}, 
                    'resume': 'checkpoint.pth'})

try:
    args = parse_args()
    device = args.device
    cfg = load_config()
except Exception as e:
    st.error(f"Initial setup failed: {e}")
    device = 'cpu'
    cfg = Box({'model': {'pad': 1024, 'image_encoder': {'patch_size': 16, 'vit_pad_mask': False}}, 
               'dataset': {'pixel_mean': [0., 0., 0.], 'pixel_std': [1., 1., 1.]}, 
               'resume': 'checkpoint.pth'})

# Disable distributed initialization
torch.distributed.is_available = lambda: False
torch.distributed.is_initialized = lambda: False
torch.distributed.get_world_size = lambda group=None: 1 
torch.distributed.get_rank = lambda group=None: 0 

@st.cache_resource
def load_detany3d_model(cfg):
    try:
        model = WrapModel(cfg)
        if not os.path.isfile(cfg.resume):
            st.warning(f"DetAny3D Checkpoint not found at: {cfg.resume}")
            return None
            
        state_dict = torch.load(cfg.resume, map_location='cpu')['state_dict']
        new_model_dict = model.state_dict()
        for k,v in new_model_dict.items():
            if k in state_dict and state_dict[k].size() == new_model_dict[k].size():
                new_model_dict[k] = state_dict[k].detach()
        model.load_state_dict(new_model_dict)
        model.to(device)
        model.setup()
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading DetAny3D model: {e}")
        return None

@st.cache_resource
def load_dino_model():
    try:
        # NOTE: Ensure GroundingDINO files are present at the assumed paths
        model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "checkpoints/groundingdino_weights/groundingdino_swint_ogc.pth")
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading GroundingDINO model. Check weights path. Error: {e}")
        return None

try:
    sam_model = load_detany3d_model(cfg)
    dino_model = load_dino_model()
except Exception as e:
    st.error(f"Model initialization error: {e}")
    sam_model = None
    dino_model = None
    
sam_trans = ResizeLongestSide(cfg.model.pad) 

# --- Utility Functions ---

def generate_video_token(text_prompt: str) -> str:
    timestamp = int(time.time())
    hash_input = f"{text_prompt}-{timestamp}".encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()

@st.cache_data
def load_nuscenes_data(pkl_path):
    if not os.path.exists(pkl_path):
        st.warning(f"Warning: nuScenes pkl not found at {pkl_path}.")
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_color_for_label(label):
    if 'color_map' not in st.session_state: st.session_state.color_map = {}
    if 'color_palette' not in st.session_state:
        palette = []
        for i in range(50):
            hue = i / 50
            rgb_float = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            palette.append([int(c * 255) for c in rgb_float])
        st.session_state.color_palette = palette
    if label not in st.session_state.color_map:
        color_index = len(st.session_state.color_map) % len(st.session_state.color_palette)
        st.session_state.color_map[label] = st.session_state.color_palette[color_index]
    return st.session_state.color_map[label]

# --- Image Preprocessing/Drawing (Needed by predict) ---

import groundingdino.datasets.transforms as T
def convert_image(img, device):
    transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_source = Image.fromarray(img)
    image_transformed, _ = transform(image_source, None)
    return np.asarray(image_source), image_transformed

def crop_hw(img):
    if img.dim() == 4: img = img.squeeze(0)
    h, w = img.shape[1:3]
    # assert max(h, w) % 112 == 0, "target_size must be divisible by 112"
    new_h, new_w = (h // 14) * 14, (w // 14) * 14
    center_h, center_w = h // 2, w // 2
    start_h, start_w = center_h - new_h // 2, center_w - new_w // 2
    return img[:, start_h:start_h + new_h, start_w:start_w + new_w].unsqueeze(0)

def preprocess(x, cfg):
    sam_pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1).to(x.device)
    sam_pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1).to(x.device)
    x = (x - sam_pixel_mean) / sam_pixel_std
    h, w = x.shape[-2:]
    padh, padw = cfg.model.pad - h, cfg.model.pad - w
    return F.pad(x, (0, padw, 0, padh))

def preprocess_dino(x):
    """Normalize pixel values for DINO (ImageNet mean/std)."""
    x = x / 255.0
    IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(x.device)
    IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(x.device)
    return (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD

def draw_text(im, text, pos, scale=0.4, color='auto', font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255), blend=0.33, lineType=1):
    text = str(text)
    pos = [int(pos[0]), int(pos[1])]
    if color == 'auto': color = (0, 0, 0) if ((bg_color[0] + bg_color[1] + bg_color[2])/3) > 127.5 else (255, 255, 255)
    
    # --- Background blending logic ---
    if bg_color is not None:
        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(x_s + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        # Ensure indices are valid slices
        y_s_clip = max(0, y_s)
        y_e_clip = min(im.shape[0], y_e + 1)
        x_s_clip = max(0, x_s)
        x_e_clip = min(im.shape[1], x_e + 1)

        im[y_s_clip:y_e_clip, x_s_clip:x_e_clip, 0] = im[y_s_clip:y_e_clip, x_s_clip:x_e_clip, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s_clip:y_e_clip, x_s_clip:x_e_clip, 1] = im[y_s_clip:y_e_clip, x_s_clip:x_e_clip, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s_clip:y_e_clip, x_s_clip:x_e_clip, 2] = im[y_s_clip:y_e_clip, x_s_clip:x_e_clip, 2]*blend + bg_color[2] * (1 - blend)
        
        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)

# --- NOA Metadata (Required for NOA page) ---
def get_noa_metadata(image_path):
    """Retrieves 3D annotations and camera intrinsics by parsing the NOA path."""
    # NOTE: Placeholder logic, assumes a standard NOA structure
    try:
        parts = image_path.split(os.path.sep)
        images_index = parts.index('images')
        scene_path = os.path.sep.join(parts[:images_index+1]) 
        sensor_name = os.path.splitext(parts[-1])[0]
        
        scene_folder_path = os.path.dirname(os.path.dirname(scene_path))
        
        try:
            base_filename_stem = sensor_name[:sensor_name.rfind('_')]
            ann_filename = base_filename_stem + '_FT.json'
        except:
            ann_filename = 'annotations.json'

        ANN_PATH = os.path.join(scene_folder_path, 'annotations', 'bb3d', ann_filename)
        CALIB_DIR = os.path.join(scene_folder_path, 'calibrations') 
        CALIB_PATH = os.path.join(CALIB_DIR, f"{sensor_name}.json") 

        gt_boxes = None
        if os.path.exists(ANN_PATH):
            with open(ANN_PATH, 'r') as f:
                annotations = json.load(f)
            gt_boxes = annotations.get("objects", annotations)

        calibration_data = {"fx": 1000, "fy": 1000, "cx": 500, "cy": 500}
        if os.path.exists(CALIB_PATH):
             # Placeholder: implement logic to read 'K', 'P_rect', or 'camera_intrinsic'
             pass

        return gt_boxes, calibration_data
    except Exception:
        return None, {"fx": 1000, "fy": 1000, "cx": 500, "cy": 500}


# --- Core Prediction Function (WITH PATCHES) ---

def predict(img, text, box_threshold, text_threshold, iou_threshold, return_boxes=False):
    """Core prediction logic using GroundingDINO and DetAny3D."""
    
    if sam_model is None or dino_model is None:
        st.warning("Using dummy prediction logic. Models failed to load.")
        if img is None: return None, None
        dummy_img = np.array(img, dtype=np.float32) / 255.0 if img is not None else np.zeros((500, 500, 3), dtype=np.float32)
        return dummy_img, None

    with torch.no_grad():
        image_token = hashlib.sha256(io.BytesIO(Image.fromarray(img)._repr_png_()).getvalue()).hexdigest()

        # 1. Grounding DINO 2D Detection
        image_source_dino, image_dino = convert_image(img, device)
        boxes, logits, phrases = dino_predict(model=dino_model, image=image_dino, caption=text, box_threshold=box_threshold, text_threshold=text_threshold, remove_combined=False)
        h, w, _ = image_source_dino.shape
        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_2d_list = [box.to(torch.int).cpu().numpy().tolist() for box in xyxy]
        label_list = phrases
        
        if len(bbox_2d_list) == 0: return np.array(img) / 255.0, None

        # 2. SAM/DetAny3D 3D Prediction Pipeline
        original_size = tuple(img.shape[:-1])
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = sam_trans.apply_image_torch(img_tensor)
        img_tensor = crop_hw(img_tensor)
        before_pad_size = tuple(img_tensor.shape[2:])

        # Preprocessing
        img_for_sam = preprocess(img_tensor, cfg).to(device)
        img_for_dino = preprocess_dino(img_tensor).to(device) # PATCH: DINO preprocessing

        # Calculate vit_pad_size (PATCH: Necessary for WrapModel)
        patch_size = cfg.model.image_encoder.patch_size
        if cfg.model.image_encoder.get('vit_pad_mask', False): # Use .get for robustness
            vit_pad_size = (before_pad_size[0] // patch_size, before_pad_size[1] // patch_size)
        else:
            vit_pad_size = (cfg.model.pad // patch_size, cfg.model.pad // patch_size)

        bbox_2d_tensor = torch.tensor(bbox_2d_list)
        bbox_2d_tensor = sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to(device)

        input_dict = {
            "images": img_for_sam,
            'vit_pad_size': torch.tensor(vit_pad_size).to(device).unsqueeze(0), # PATCH: Added
            "images_shape": torch.Tensor(before_pad_size).to(device).unsqueeze(0),
            "image_for_dino": img_for_dino, # PATCH: Added
            "boxes_coords": bbox_2d_tensor,
        }
        
        # Model Forward
        ret_dict = sam_model(input_dict)

        # 3. Decode BBoxes and Visualization
        K = ret_dict['pred_K']
        decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K) # Assumes decode_bboxes in train_utils.py
        rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])
        pred_box_ious = ret_dict.get('pred_box_ious', None)
        
        # De-normalize image for drawing (approximation using SAM's stats)
        # Assuming cfg.dataset.pixel_mean and std are ImageNet-like [123.675, 116.28, 103.53] and [58.395, 57.12, 57.375]
        mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
        std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
        origin_img = (std.cpu() * img_for_sam[0, :, :before_pad_size[0], :before_pad_size[1]].squeeze(0).detach().cpu()) + mean.cpu()
        
        # Convert to BGR for OpenCV
        todo = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        K = K.detach().cpu().numpy()
        
        final_boxes_3d = []
        for i in range(len(decoded_bboxes_pred_2d)):
            iou_score = pred_box_ious[i][torch.argmax(pred_box_ious[i])].item() if pred_box_ious is not None and pred_box_ious.dim() > 1 else 1.0

            if iou_score < iou_threshold:
                continue
            
            x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
            rot_mat_i = rot_mat[i].detach().cpu().numpy()
            
            # 3D Projection
            vertices_3d, _ = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
            vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            color = get_color_for_label(label_list[i])
            
            # Draw
            draw_bbox_2d(todo, vertices_2d, color=(int(color[0]), int(color[1]), int(color[2])), thickness=2)
            draw_text(todo, f"{label_list[i]}", box_cxcywh_to_xyxy(decoded_bboxes_pred_2d[i]).detach().cpu().numpy().tolist(), scale=0.50*todo.shape[0]/500, bg_color=color)
            final_boxes_3d.append(decoded_bboxes_pred_3d[i])

        # Final cleanup and conversion
        todo = np.clip(todo, 0, 255)
        # If draw_text promotes to float64, this conversion is necessary.
        if todo.dtype != np.float32:
            todo = todo.astype(np.float32)

        # 3. Scale down the 0-255 image data to 0-1 range (for Streamlit/final output)
        todo = todo / 255.0
        rgb_image = cv2.cvtColor(todo, cv2.COLOR_BGR2RGB)
        
    return rgb_image, final_boxes_3d if return_boxes else None

# --- Dataset-Specific Loaders & Search Functions ---

def load_random_nuscenes_image():
    NUSCENES_PKL_PATH = './data/DA3D_pkls/omni3d23daw/nuScenes_val.pkl'
    nuscenes_data = load_nuscenes_data(NUSCENES_PKL_PATH)
    path_replace_rule = ('./data/nuscenes', '/data/nuscenes')

    if not nuscenes_data: return None
    
    random_item = random.choice(nuscenes_data)
    absolute_path = random_item['img_path'].replace(path_replace_rule[0], path_replace_rule[1])

    if not os.path.exists(absolute_path):
        st.error(f"Image path from pkl does not exist: {absolute_path}")
        return None
    return Image.open(absolute_path).convert("RGB")

# --- NOA Helpers (All Pinhole Logic) ---

@st.cache_data
def get_all_noa_pinhole_images():
    """Traverses the NOA root directory and collects paths to ALL pinhole images."""
    NOA_ROOT_DIR = './Traces_NOA_structured'
    ALL_PINHOLE_IMAGES = []
    
    if not os.path.isdir(NOA_ROOT_DIR): return ALL_PINHOLE_IMAGES
        
    for root, dirs, files in os.walk(NOA_ROOT_DIR):
        if 'pinhole' in root:
            image_files = [f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            for img_file in image_files:
                ALL_PINHOLE_IMAGES.append(os.path.join(root, img_file))
    return ALL_PINHOLE_IMAGES

def load_random_noa_image():
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

def find_and_predict_noa(text_prompt, box_threshold, text_threshold, iou_threshold, progress_bar):
    """Generates a video by searching for a scene in NOA containing the target object and processing its frames."""
    NOA_ROOT_DIR = './Traces_NOA_structured' 
    if not os.path.isdir(NOA_ROOT_DIR):
        st.error(f"❌ NOA root directory not found at {NOA_ROOT_DIR}.")
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
                    with open(ann_path, 'r') as f: annotations = json.load(f)
                    if isinstance(annotations, dict) and "objects" in annotations: annotations = annotations["objects"]
                    
                    if any(search_category in (obj.get("geometry", {}).get("class") or obj.get("label", "")).lower() for obj in annotations):
                        found_scene_dir = scene_path
                        break
                except: continue
        if found_scene_dir: break
            
    if not found_scene_dir:
        st.error(f"❌ No scenes found containing '{search_category}'.")
        return None

    # --- Image Collection ---
    img_pinhole_dir = os.path.join(found_scene_dir, 'images/pinhole')
    relevant_images = []
    if os.path.isdir(img_pinhole_dir):
        all_img_files = sorted([f for f in os.listdir(img_pinhole_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
        relevant_images = [os.path.join(img_pinhole_dir, f) for f in all_img_files]
    
    if not relevant_images:
        st.error(f"❌ Scene found, but no images in {img_pinhole_dir}.")
        return None
        
    MAX_FRAMES = 10
    if len(relevant_images) > MAX_FRAMES: relevant_images = relevant_images[:MAX_FRAMES]

    st.info(f"📸 Processing scene {os.path.basename(found_scene_dir)} with {len(relevant_images)} frames...")

    # --- Prepare video output ---
    output_video_dir = './exps/videos'
    os.makedirs(output_video_dir, exist_ok=True)
    video_token = generate_video_token(text_prompt)
    output_video_path = os.path.join(output_video_dir, f'noa_{video_token}.avi')

    first_img = Image.open(relevant_images[0]).convert("RGB")
    width, height = first_img.size
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 5, (width, height), True)
    
    if not video_writer.isOpened():
        output_video_path = os.path.join(output_video_dir, f'noa_{video_token}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 5, (width, height), True)

    if not video_writer.isOpened():
        st.error("❌ Could not initialize video writer with any known codec. Video generation failed.")
        return None
    
    valid_frames = 0
    for i, img_path in enumerate(relevant_images):
        progress_bar.progress(10 + int(90 * (i + 1) / len(relevant_images)), text=f"Processing frame {i+1}/{len(relevant_images)}: {os.path.basename(img_path)}")
        try:
            _, calibration_data = get_noa_metadata(img_path) 
            st.sidebar.caption(f"Frame {i}: Calib fx={calibration_data['fx']:.0f}, fy={calibration_data['fy']:.0f}")

            img = Image.open(img_path).convert("RGB")
            result_img, _ = predict(np.array(img), text_prompt, box_threshold, text_threshold, iou_threshold)

            if result_img is None: continue

            frame = np.clip(result_img * 255, 0, 255).astype(np.uint8)
            if frame.ndim == 3 and frame.shape[2] == 4: frame = frame[:, :, :3]
            if (frame.shape[1], frame.shape[0]) != (width, height): frame = cv2.resize(frame, (width, height))

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            valid_frames += 1

        except Exception as e:
            st.warning(f"⚠️ Skipping {img_path} due to error: {e}")
            continue

    video_writer.release()
    return output_video_path if valid_frames > 0 else None

# --- NuScenes Helpers ---

def find_and_predict(img_path_list, text_prompt, box_threshold, text_threshold, iou_threshold, progress_bar):
    """Finds a single image in nuScenes based on the prompt and runs prediction."""
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

    if not nuscenes_data: st.error("nuScenes data not loaded. Cannot search.") ; return None

    search_category = text_prompt.strip().split('.')[0].split(' ')[-1].lower()
    if not search_category: st.error("Please enter a valid text prompt.") ; return None

    matched_category = None
    for name in category_to_images.keys():
        if search_category in name.lower() or name.lower() in search_category: 
              matched_category = name ; break
            
    if matched_category is None or not category_to_images[matched_category]:
        st.error(f"No images found for category matching '{search_category}' in nuScenes.")
        return None

    random_path = random.choice(category_to_images[matched_category])
    
    st.info(f"Found image with '{matched_category}' at: {os.path.basename(random_path)}. Running prediction...")
    progress_bar.progress(50, text="Image found, running prediction...")

    img = Image.open(random_path).convert("RGB")
    
    return predict(np.array(img), text_prompt, box_threshold, text_threshold, iou_threshold)[0]

def find_and_predict_nuscenes_video(text_prompt, box_threshold, text_threshold, iou_threshold, progress_bar):
    """Finds an image sequence in nuScenes, runs prediction, and creates a video."""
    NUSCENES_PKL_PATH = './data/DA3D_pkls/omni3d23daw/nuScenes_val.pkl'
    nuscenes_data = load_nuscenes_data(NUSCENES_PKL_PATH)

    if not nuscenes_data: st.error("nuScenes data not loaded.") ; return None

    with open('./data/category_meta.json', 'r') as f:
        category_meta = json.load(f)
    category_id_to_name = {i: name for i, name in enumerate(category_meta['thing_classes'])}

    search_category = text_prompt.strip().split('.')[0].split(' ')[-1].lower()
    if not search_category: st.error("Please enter a valid text prompt.") ; return None

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
        if target_scene_items: break

    if not target_scene_items: st.error(f"No sequence found for '{search_category}'.") ; return None

    relevant_items = target_scene_items[:50] 
    st.info(f"Generating video for the sequence with {len(relevant_items)} frames...")

    output_video_dir = './exps/deploy/videos'
    os.makedirs(output_video_dir, exist_ok=True)
    video_token = generate_video_token(text_prompt)
    output_video_path = os.path.join(output_video_dir, f'nuscenes_{video_token}.avi')

    first_img = Image.open(relevant_items[0]['img_path'].replace(path_replace_rule[0], path_replace_rule[1]))
    width, height = first_img.size
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 5, (width, height))
    
    if not video_writer.isOpened(): st.error("❌ Could not initialize video writer.") ; return None

    for i, item in enumerate(relevant_items):
        progress_bar.progress(10 + int(90 * (i + 1) / len(relevant_items)), text=f"Processing image {i+1}/{len(relevant_items)}")
        img_path = item['img_path'].replace(path_replace_rule[0], path_replace_rule[1])
        img = Image.open(img_path).convert("RGB")
        result_img, _ = predict(np.array(img), text_prompt, box_threshold, text_threshold, iou_threshold)
        if result_img is not None:
            frame = (result_img * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

    video_writer.release()
    st.info(f"Video saved to {output_video_path}")
    return output_video_path