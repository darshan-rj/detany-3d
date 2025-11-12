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

# --- get_noa_metadata (Unchanged) ---
def get_noa_metadata(image_path):
    """
    Retrieves 3D annotations (GT boxes) and camera intrinsics (calibrations) 
    by parsing the scene path from the image_path.
    """
    
    # 1. Parse Path to find the scene root
    try:
        # Find the index of 'images' and work backwards
        parts = image_path.split(os.path.sep)
        images_index = parts.index('images')
        
        # The scene directory is the parent of 'images'
        scene_path = os.path.sep.join(parts[:images_index+1]) 
        sensor_name = os.path.splitext(parts[-1])[0]
    except ValueError:
         # Fallback if the standard structure is not found
         return None, {"fx": 1000, "fy": 1000, "cx": 500, "cy": 500}
    except Exception as e:
        st.error(f"Error parsing NOA path structure: {e}")
        return None, {"fx": 1000, "fy": 1000, "cx": 500, "cy": 500}
        
    # 2. Setup Annotation and Calibration Paths
    # We assume 'annotations' and 'calibrations' are siblings to 'images'
    scene_folder_path = os.path.dirname(os.path.dirname(scene_path))
    
    ANN_PATH = os.path.join(scene_folder_path, 'annotations/bb2d/FT.json')
    CALIB_DIR = os.path.join(scene_folder_path, 'all_frames', 'calibrations') 
    CALIB_PATH = os.path.join(CALIB_DIR, f"{sensor_name}.json") 

    gt_boxes = None
    calibration_data = None 

    # --- Load Annotations (GT Boxes) from the single BEV.json per scene ---
    if os.path.exists(ANN_PATH):
        try:
            with open(ANN_PATH, 'r') as f:
                annotations = json.load(f)
            
            # Placeholder for filtering
            if isinstance(annotations, dict) and "objects" in annotations:
                gt_boxes = annotations["objects"]
            elif isinstance(annotations, list):
                gt_boxes = annotations
            
        except Exception as e:
            st.warning(f"Could not load GT data from {ANN_PATH}: {e}")


    # --- Load Calibration Data (Intrinsics) ---
    print(f"Calibration path: {CALIB_PATH}")
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH, 'r') as f:
                calib_json = json.load(f)
                
            if 'K' in calib_json and isinstance(calib_json['K'], list) and len(calib_json['K']) >= 9:
                K = np.array(calib_json['K']).reshape(3, 3)
                calibration_data = {
                    "fx": K[0, 0], "fy": K[1, 1], 
                    "cx": K[0, 2], "cy": K[1, 2]
                }
            elif 'P_rect' in calib_json and isinstance(calib_json['P_rect'], list) and len(calib_json['P_rect']) >= 12:
                P = np.array(calib_json['P_rect']).reshape(3, 4)
                calibration_data = {
                    "fx": P[0, 0], "fy": P[1, 1], 
                    "cx": P[0, 2], "cy": P[1, 2]
                }
            # Added check for 'camera_intrinsic' key (correct for NOA)
            elif 'camera_intrinsic' in calib_json and isinstance(calib_json['camera_intrinsic'], list) and len(calib_json['camera_intrinsic']) >= 9:
                K = np.array(calib_json['camera_intrinsic']).reshape(3, 3)
                calibration_data = {
                    "fx": K[0, 0], "fy": K[1, 1],
                    "cx": K[0, 2], "cy": K[1, 2]
                }
            # Added check for nested 'intrinsics' -> 'intrinsic_matrix' (correct for your file)
            elif 'intrinsics' in calib_json and isinstance(calib_json.get('intrinsics'), dict) and 'intrinsic_matrix' in calib_json['intrinsics']:
                K = np.array(calib_json['intrinsics']['intrinsic_matrix']).reshape(3, 3)
                calibration_data = {
                    "fx": K[0, 0], "fy": K[1, 1],
                    "cx": K[0, 2], "cy": K[1, 2]
                }
            # Added check for separate f, c parameters (common format)
            elif 'f' in calib_json and 'c' in calib_json and isinstance(calib_json['f'], list) and isinstance(calib_json['c'], list):
                calibration_data = {
                    "fx": calib_json['f'][0], "fy": calib_json['f'][1], 
                    "cx": calib_json['c'][0], "cy": calib_json['c'][1]
                }
            else:
                calibration_data = {"fx": 1000, "fy": 1000, "cx": 500, "cy": 500}
                st.warning(f"Calibration format unexpected in {CALIB_PATH}. Using placeholder.")

        except Exception as e:
            st.error(f"Error loading calibration data from {CALIB_PATH}: {e}")
            calibration_data = {"fx": 1000, "fy": 1000, "cx": 500, "cy": 500}
    else:
        st.warning(f"Calibration file not found at {CALIB_PATH}. Using placeholder.")
        calibration_data = {"fx": 1000, "fy": 1000, "cx": 500, "cy": 500}

    return gt_boxes, calibration_data

# ----------------------------------------------------------------------
# 1. ARGPARSE SETUP TO HANDLE STREAMLIT/LAUNCH.JSON ARGUMENTS
# ----------------------------------------------------------------------

def calculate_iou_3d(box1, box2):
    """Calculate 3D IoU between two bounding boxes (simplified AABB)."""
    # Convert to x1,y1,z1,x2,y2,z2
    b1_x1, b1_y1, b1_z1 = box1[0] - box1[5]/2, box1[1] - box1[4]/2, box1[2] - box1[3]/2
    b1_x2, b1_y2, b1_z2 = box1[0] + box1[5]/2, box1[1] + box1[4]/2, box1[2] + box1[3]/2
    b2_x1, b2_y1, b2_z1 = box2[0] - box2[5]/2, box2[1] - box2[4]/2, box2[2] - box2[3]/2
    b2_x2, b2_y2, b2_z2 = box2[0] + box2[5]/2, box2[1] + box2[4]/2, box2[2] + box2[3]/2

    inter_vol = max(0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * max(0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1)) * max(0, min(b1_z2, b2_z2) - max(b1_z1, b2_z1))
    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]
    return inter_vol / (vol1 + vol2 - inter_vol + 1e-6)

def parse_args():
    parser = argparse.ArgumentParser(description='DetAny3D Streamlit Demo')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to run the demo on (e.g., "cuda:0", "cuda:1", "cpu")')
    args = parser.parse_args()
    return args

device = parse_args().device

st.set_page_config(page_title="DetAny3D Streamlit Demo", layout="wide")

@st.cache_resource
def load_config():
    with open('./detect_anything/configs/demo.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return Box(cfg)

cfg = load_config()

def generate_image_token(image: Image.Image) -> str:
    """Generate a unique token (SHA-256 hash) based on a PIL.Image"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image) 
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG') 
    return hashlib.sha256(img_bytes.getvalue()).hexdigest()

def generate_video_token(text_prompt: str) -> str:
    """Generate a unique token for the video based on prompt and timestamp."""
    timestamp = int(time.time())
    hash_input = f"{text_prompt}-{timestamp}".encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()

@st.cache_data
def load_nuscenes_data(pkl_path):
    """Load full annotation data from the nuScenes pickle file."""
    if not os.path.exists(pkl_path):
        st.warning(f"Warning: nuScenes pkl not found at {pkl_path}. Search/load features will be disabled.")
        return None
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_color_for_label(label):
    """Generates a consistent color for a given label."""
    if 'color_map' not in st.session_state:
        st.session_state.color_map = {}
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

# Disable distributed initialization to run on Streamlit/single machine
torch.distributed.is_available = lambda: False
torch.distributed.is_initialized = lambda: False
torch.distributed.get_world_size = lambda group=None: 1 
torch.distributed.get_rank = lambda group=None: 0 

@st.cache_resource
def load_detany3d_model(cfg):
    model = WrapModel(cfg)
    if not os.path.isfile(cfg.resume):
        raise FileNotFoundError(f"Checkpoint file for demo not found at: {cfg.resume}")
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

@st.cache_resource
def load_dino_model():
    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "checkpoints/groundingdino_weights/groundingdino_swint_ogc.pth")
    model.eval()
    model.to(device)
    return model

try:
    sam_model = load_detany3d_model(cfg)
    dino_model = load_dino_model()
except Exception as e:
    st.error(f"Error loading models. Please ensure checkpoint files and GroundingDINO weights are correctly placed. Device used: {device}. Error: {e}")
    st.stop()
    
sam_trans = ResizeLongestSide(cfg.model.pad) 

BOX_TRESHOLD = 0.37
TEXT_TRESHOLD = 0.25

import groundingdino.datasets.transforms as T
def convert_image(img, device):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(img)
    image_transformed, _ = transform(image_source, None)
    return np.asarray(image_source), image_transformed

def crop_hw(img):
    if img.dim() == 4:
        img = img.squeeze(0)
    h, w = img.shape[1:3]
    assert max(h, w) % 112 == 0, "target_size must be divisible by 112"

    new_h = (h // 14) * 14
    new_w = (w // 14) * 14
    center_h, center_w = h // 2, w // 2
    start_h = center_h - new_h // 2
    start_w = center_w - new_w // 2

    img_cropped = img[:, start_h:start_h + new_h, start_w:start_w + new_w]
    return img_cropped.unsqueeze(0)

def preprocess(x, cfg):
    """Normalize pixel values and pad to a square input."""
    sam_pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
    sam_pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
    x = (x - sam_pixel_mean) / sam_pixel_std

    h, w = x.shape[-2:]
    padh = cfg.model.pad - h
    padw = cfg.model.pad - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def preprocess_dino(x):
    """Normalize pixel values and pad to a square input."""
    x = x / 255
    IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    x = (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD
    return x

def draw_text(im, text, pos, scale=0.4, color='auto', font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1):
    """Utility to draw text on an image with a blended background."""
    text = str(text)
    pos = [int(pos[0]), int(pos[1])]

    if color == 'auto':
        color = (0, 0, 0) if ((bg_color[0] + bg_color[1] + bg_color[2])/3) > 127.5 else (255, 255, 255)
    
    if bg_color is not None:
        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(x_s + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)
        
        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)

def predict(img, text, box_threshold, text_threshold, iou_threshold, return_boxes=False):
    with torch.no_grad():
        image_token = generate_image_token(img)

        if img is None:
            st.error("No image received")
            return None, None
        
        label_list = []
        bbox_2d_list = []
        
        image_source_dino, image_dino = convert_image(img, device)
            
        boxes, logits, phrases = dino_predict(
            model=dino_model,
            image=image_dino,
            caption=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            remove_combined=False,
        )
        h, w, _ = image_source_dino.shape
        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        for i, box in enumerate(xyxy):
            bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
            label_list.append(phrases[i])

        if len(bbox_2d_list) == 0:
            return None, None
            
        original_size = tuple(img.shape[:-1])
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        img = img.unsqueeze(0)

        img = sam_trans.apply_image_torch(img)
        img = crop_hw(img)
        before_pad_size = tuple(img.shape[2:])
        
        img_for_sam = preprocess(img, cfg).to(device)
        img_for_dino = preprocess_dino(img).to(device)

        resize_ratio = max(img_for_sam.shape) / max(original_size)
        image_h, image_w = int(before_pad_size[0]), int(before_pad_size[1])

        if cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // cfg.model.image_encoder.patch_size, before_pad_size[1] // cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (cfg.model.pad // cfg.model.image_encoder.patch_size, cfg.model.pad // cfg.model.image_encoder.patch_size)

        bbox_2d_tensor = torch.tensor(bbox_2d_list)
        bbox_2d_tensor = sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to(device)
        input_dict = {
            "images": img_for_sam,
            'vit_pad_size': torch.tensor(vit_pad_size).to(device).unsqueeze(0),
            "images_shape": torch.Tensor(before_pad_size).to(device).unsqueeze(0),
            "image_for_dino": img_for_dino,
            "boxes_coords": bbox_2d_tensor,
        }

        ret_dict = sam_model(input_dict)
        
        K = ret_dict['pred_K']
        decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K)
        rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])
        pred_box_ious = ret_dict.get('pred_box_ious', None)

        origin_img = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_for_sam[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        todo = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        K = K.detach().cpu().numpy()
        
        for i in range(len(decoded_bboxes_pred_2d)):
            x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
            rot_mat_i = rot_mat[i].detach().cpu().numpy()
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
            vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
            color = get_color_for_label(label_list[i])
            
            iou_score = pred_box_ious[i][torch.argmax(pred_box_ious[i])].item() if pred_box_ious is not None else 1.0

            if iou_score < iou_threshold:
                continue

            draw_bbox_2d(todo, vertices_2d, color=(int(color[0]), int(color[1]), int(color[2])), thickness=2)
            if label_list[i] is not None:
                # draw_text(todo, f"{label_list[i]} {[round(c, 2) for c in decoded_bboxes_pred_3d[i][3:6].detach().cpu().numpy().tolist()]}", box_cxcywh_to_xyxy(decoded_bboxes_pred_2d[i]).detach().cpu().numpy().tolist(), scale=0.50*todo.shape[0]/500, bg_color=color)
                draw_text(todo, f"{label_list[i]}", box_cxcywh_to_xyxy(decoded_bboxes_pred_2d[i]).detach().cpu().numpy().tolist(), scale=0.50*todo.shape[0]/500, bg_color=color)
        
        os.makedirs('./exps/deploy', exist_ok=True)
        cv2.imwrite(f'./exps/deploy/{image_token}.jpg', todo)
        todo = np.clip(todo, 0, 255) 
        todo = todo / 255
        rgb_image = cv2.cvtColor(todo, cv2.COLOR_BGR2RGB)
        
    if return_boxes:
        return rgb_image, decoded_bboxes_pred_3d
    return rgb_image, None

def find_and_predict(text_prompt, box_threshold, text_threshold, iou_threshold, progress_bar):
    """Finds an image in nuScenes based on the prompt and runs prediction."""
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

# --- REVISED: find_and_predict_noa for Dynamic Scene Traversal & AVI Fixes ---
def find_and_predict_noa(text_prompt, box_threshold, text_threshold, iou_threshold, progress_bar):
    """
    Finds a single scene (at any depth) in the NOA dataset containing the search category, 
    collects all pinhole images from that scene, and generates a video in AVI format.
    """
    NOA_ROOT_DIR = './Traces_NOA' 
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
    
    # Use os.walk to search for the annotation marker (BEV.json) dynamically
    for root, dirs, files in os.walk(NOA_ROOT_DIR):
        if 'BEV.json' in files and 'bb3d' in root:
            ann_path = os.path.join(root, 'BEV.json')
            
            # The Scene Directory is the parent of 'annotations' (e.g., '000455')
            scene_path = os.path.dirname(os.path.dirname(root)) 
            
            if os.path.exists(ann_path):
                try:
                    with open(ann_path, 'r') as f:
                        annotations = json.load(f)

                    if isinstance(annotations, dict) and "objects" in annotations:
                        annotations = annotations["objects"]
                    
                    # Check if the target category is in this scene's annotations
                    if any(search_category in (obj.get("geometry", {}).get("class") or obj.get("label", "")).lower() for obj in annotations):
                        found_scene_dir = scene_path
                        break
                except:
                    # Skip scene if JSON is corrupt or unreadable
                    continue
        if found_scene_dir:
            break
            
    if not found_scene_dir:
        st.error(f"❌ No scenes found containing '{search_category}'.")
        return None

    # --- Collect all pinhole images from the found scene ---
    img_pinhole_dir = os.path.join(found_scene_dir, 'images/pinhole')
    if os.path.isdir(img_pinhole_dir):
        # We collect all frames from all pinhole sensors (FC, FL, FR, etc.)
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

    # --- Prepare video output ---
    output_video_dir = './exps/videos'
    os.makedirs(output_video_dir, exist_ok=True)
    video_token = generate_video_token(text_prompt)
    output_video_path = os.path.join(output_video_dir, f'noa_{video_token}.mp4') # CHANGED to .avi

    # --- Initialize safe VideoWriter ---
    first_img = Image.open(relevant_images[0]).convert("RGB")
    width, height = first_img.size
    
    
    # Try a few codecs for maximum compatibility
    codecs_to_try = ['mp4v', 'avc1', 'H264']
    video_writer = None
    for codec in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 1, (width, height))
        if video_writer.isOpened():
            video_writer = video_writer
            st.info(f"🎞️ Using codec: {codec}")
            break

    if video_writer is None:
        st.error("❌ Could not initialize video writer. Try changing codec or output format.")
        return None
    
    valid_frames = 0
    for i, img_path in enumerate(relevant_images):
        print(f"Image Path: {img_path}")
        progress_bar.progress(10 + int(90 * (i + 1) / len(relevant_images)),
                              text=f"Processing frame {i+1}/{len(relevant_images)}: {os.path.basename(img_path)}")
        try:
            # --- CALIBRATION DATA RETRIEVAL (using dynamic path parsing) ---
            _, calibration_data = get_noa_metadata(img_path) 
            st.sidebar.caption(f"Frame {i}: Calib fx={calibration_data['fx']:.0f}, fy={calibration_data['fy']:.0f}")

            img = Image.open(img_path).convert("RGB")
            result_img, _ = predict(np.array(img), text_prompt, box_threshold, text_threshold, iou_threshold)

            if result_img is None:
                continue

            # Convert to uint8
            frame = np.clip(result_img * 255, 0, 255).astype(np.uint8)

            # Drop alpha channel if present
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # Resize to match first frame size (safety)
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height))

            # Convert RGB to BGR for OpenCV
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

# --- REVISED: load_random_noa_image (Dynamic Traversal) ---
def load_random_noa_image():
    """Function to load a random image from the NOA dataset (updated to search scene structure dynamically)."""
    NOA_ROOT_DIR = './Traces_NOA'
    ALL_PINHOLE_IMAGES = []
    
    if not os.path.isdir(NOA_ROOT_DIR):
        st.error(f"NOA root directory not found at: {NOA_ROOT_DIR}")
        return None, None
        
    # Find all pinhole image files using os.walk
    for root, dirs, files in os.walk(NOA_ROOT_DIR):
        if 'pinhole' in root:
            image_files = [f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            for img_file in image_files:
                ALL_PINHOLE_IMAGES.append(os.path.join(root, img_file))

    if not ALL_PINHOLE_IMAGES:
        st.error("No pinhole image files found in NOA structure.")
        return None, None

    try:
        image_path = random.choice(ALL_PINHOLE_IMAGES)
        return Image.open(image_path).convert("RGB"), image_path
    except Exception as e:
        st.error(f"Error loading random NOA image: {e}")
        return None, None


def load_random_nuscenes_image():
    """Function to be called by the button to load a random image."""
    NUSCENES_PKL_PATH = './data/DA3D_pkls/omni3d23daw/nuScenes_val.pkl'
    nuscenes_data = load_nuscenes_data(NUSCENES_PKL_PATH)
    path_replace_rule = ('./data/nuscenes', '/data/nuscenes')

    if not nuscenes_data:
        st.error("nuScenes data not loaded. Check pkl path.")
        return None
    
    random_item = random.choice(nuscenes_data)
    absolute_path = random_item['img_path'].replace(path_replace_rule[0], path_replace_rule[1])

    if not os.path.exists(absolute_path):
        st.error(f"Image path from pkl does not exist: {absolute_path}")
        st.session_state.current_image_path = None
        return None
    return Image.open(absolute_path).convert("RGB")


def find_and_predict_nuscenes_video(text_prompt, box_threshold, text_threshold, iou_threshold, progress_bar):
    """Finds an image sequence in nuScenes based on prompt, runs prediction, and creates a video in AVI format."""
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
    output_video_path = os.path.join(output_video_dir, f'nuscenes_{video_token}.avi') # CHANGED to .avi

    first_img = Image.open(relevant_items[0]['img_path'].replace(path_replace_rule[0], path_replace_rule[1]))
    width, height = first_img.size
    
    # 💥 FIX: Use DIVX codec for reliable AVI writing
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 1, (width, height))

    for i, item in enumerate(relevant_items):
        progress_text = f"Processing image {i+1}/{len(relevant_items)}"
        progress_bar.progress(5 + int(90 * (i + 1) / len(relevant_items)), text=progress_text)
        img_path = item['img_path'].replace(path_replace_rule[0], path_replace_rule[1])
        img = Image.open(img_path).convert("RGB")
        result_img, _ = predict(np.array(img), text_prompt, box_threshold, text_threshold, iou_threshold)
        if result_img is not None:
            frame = (result_img * 255).astype(np.uint8)
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()

    
    st.info(f"Video saved to {output_video_path}")
    return output_video_path


st.title("DetAny3D: Detect Anything 3D")
st.markdown(f"**Running on Device:** `{device}`") 

col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    st.markdown("#### 1. Provide an Image")
    uploaded_file = st.file_uploader("Upload your own image", type=["png", "jpg", "jpeg"])
    
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None

    if uploaded_file is not None:
        st.session_state.image = Image.open(uploaded_file).convert("RGB")
        st.session_state.current_image_path = None
    
    st.markdown("...or load a random one from a dataset:")
    b_col1, b_col2 = st.columns(2)
    if b_col1.button("Load from nuScenes", use_container_width=True):
        st.session_state.image = load_random_nuscenes_image()
        st.session_state.current_image_path = None
    if b_col2.button("Load from NOA", use_container_width=True):
        image, path = load_random_noa_image()
        st.session_state.image = image
        st.session_state.current_image_path = path

    if st.session_state.image:
        st.image(st.session_state.image, caption="Uploaded Image",use_container_width=True)

    st.markdown("#### 2. Enter a Text Prompt")
    text_input = st.text_input("Text Prompt", "a car . a person", help="Separate different objects with ' . '")

    with st.expander("Detection Thresholds"):
        box_threshold = st.slider("Box Threshold (2D)", 0.0, 1.0, BOX_TRESHOLD, 0.01)
        text_threshold = st.slider("Text Threshold (2D)", 0.0, 1.0, TEXT_TRESHOLD, 0.01)
        iou_threshold = st.slider("3D IoU Threshold", 0.0, 1.0, 0.2, 0.01, help="Filters final 3D bounding boxes based on their predicted Intersection over Union (IoU) score.")

    st.markdown("#### 3. Run Detection")
    submit_btn = st.button("Run Detection on Current Image", type="primary", use_container_width=True)
    
    st.markdown("---")
    with st.expander("Advanced: Search Datasets & Generate Video"):
        st.info("These options will search the dataset for an image/scene matching your prompt and then run detection. Output is `.avi`.")
        search_nuscenes_btn = st.button("Find & Predict from nuScenes (Single Image)", use_container_width=True)
        search_nuscenes_video_btn = st.button("Find & Predict from nuScenes (Video)", use_container_width=True)
        search_noa_video_btn = st.button("Find & Predict from NOA (Video)", use_container_width=True)

with col2:
    st.header("3D Detection Result")
    output_container = st.empty()
    output_container.info("Output will be displayed here. The first run may take a moment to load models.", icon="🖼️")
    
    
if submit_btn:
    if st.session_state.image is None:
        st.error("Please upload an image first.")
    elif not text_input:
        st.error("Please enter a text prompt.")
    else:
        with st.spinner("Running prediction..."):
            result_image, pred_boxes_3d = predict(np.array(st.session_state.image), text_input, box_threshold, text_threshold, iou_threshold, return_boxes=True)
            if result_image is not None:
                output_container.image(result_image, caption="Detection Result", use_container_width=True)
                
                if st.session_state.current_image_path and './Traces_NOA' in st.session_state.current_image_path:
                    _, calib_data = get_noa_metadata(st.session_state.current_image_path)
                    st.sidebar.success(f"Loaded Calibration/Annotations for frame. Example Calib: fx={calib_data['fx']:.0f}")

if search_nuscenes_btn:
    if not text_input:
        st.error("Please enter a text prompt to search.")
    else:
        progress_bar = output_container.progress(0, text="Starting nuScenes search...")
        result_image = find_and_predict(text_input, box_threshold, text_threshold, iou_threshold, progress_bar)
        if result_image is not None:
            progress_bar.progress(100, text="Done!")
            output_container.image(result_image, caption="Detection Result from nuScenes", use_container_width=True)
        else:
            progress_bar.empty()

if search_noa_video_btn:
    if not text_input:
        st.error("Please enter a text prompt to search.")
    else:
        progress_bar = output_container.progress(0, text="Starting NOA video search and generation...")
        video_path = find_and_predict_noa(text_input, box_threshold, text_threshold, iou_threshold, progress_bar)
        if video_path:
            progress_bar.progress(100, text="Video generation complete! Playing AVI...")
            # Streamlit will use browser capabilities to play the AVI. If it fails, 
            # it means the browser itself lacks the DIVX/AVI decoding plug-in.
            output_container.video(video_path) 
            
            st.download_button(
                label="Download Inference Video (AVI)",
                data=open(video_path, 'rb').read(),
                file_name=os.path.basename(video_path).replace('.avi', '.avi'),
                mime="video/avi" # CHANGED MIME TYPE
            )
        else:
            progress_bar.empty()

if search_nuscenes_video_btn:
    if not text_input:
        st.error("Please enter a text prompt to search.")
    else:
        progress_bar = output_container.progress(0, text="Starting nuScenes video search and generation...")
        video_path = find_and_predict_nuscenes_video(text_input, box_threshold, text_threshold, iou_threshold, progress_bar)
        if video_path:
            progress_bar.progress(100, text="Video generation complete! Playing AVI...")
            output_container.video(video_path) 
            
            st.download_button(
                label="Download nuScenes Inference Video (AVI)",
                data=open(video_path, 'rb').read(),
                file_name=os.path.basename(video_path).replace('.avi', '.avi'),
                mime="video/avi" # CHANGED MIME TYPE
            )
        else:
            progress_bar.empty()