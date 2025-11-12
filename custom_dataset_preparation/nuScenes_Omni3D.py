# nuScenes data preparation for DetAny3D training Conversion to Omni3D format

import os
import json
import pickle
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
from typing import List, Dict
from detectron2.structures import BoxMode # Required for BoxMode.XYXY_ABS = 0

# --- 1. CONFIGURATION AND CONSTANTS ---

# Replace with your nuScenes data root and version
NUSCENES_ROOT = '/data/nuscenes'
NUSCENES_VERSION = 'v1.0-test' # Use 'v1.0-trainval' for full dataset

# Output directory for the DetAny3D/Omni3D format data
OUTPUT_PKL_PATH = './data/DA3D_pkls/nuscenes_omni3d_test.pkl'

# Omni3D-Specific Constants
# The virtual focal length (f_v) for the Virtual Depth calculation (Omni3D convention)
# Placeholder value. The official value is typically around 1070 or 1000.
VIRTUAL_FOCAL_LENGTH_FV = 1070.0 #

# Omni3D Unified Category Mapping (Derived from category_meta.json)
# Maps nuScenes canonical names to Omni3D contiguous IDs (0-97)
OMNI3D_CLASS_MAP = {
    'pedestrian': 0, 
    'car': 1, 
    'truck': 5, 
    'bus': 12, 
    'trailer': 13,
    'motorcycle': 10, 
    'bicycle': 11, 
    'traffic_cone': 8, 
    'barrier': 9,
    # Map 'construction_vehicle' to 'truck' as an educated guess for urban scenes
    'construction_vehicle': 5 
}

# --- 2. TRANSFORMATION UTILITIES ---

def get_3d_box_corners(box: NuScenesBox) -> np.ndarray:
    """
    Computes the 8 corners of the 3D bounding box (3x8 matrix) in the box's local coordinate system.
    This is a standard utility function, which the devkit's Box.corners() method likely performs.
    """
    # Standard corners in the box's local frame (L/2, W/2, H/2)
    l, w, h = box.wlh
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    
    # Rotate and translate to get corners in the parent frame (global, then camera)
    corners = np.vstack((x_corners, y_corners, z_corners))
    return corners # Note: The rotation and translation are applied later using the box's pose.

def transform_box_to_camera_frame(box: NuScenesBox, sample_data_record: Dict, nusc: NuScenes) -> Dict:
    """
    Transforms a nuScenes global annotation box into the specific camera frame 
    and applies Omni3D-specific conversions.
    """
    # 1. Coordinate Transformation: Global to Camera Frame (T_global -> T_camera)
    # The nuScenes devkit Box class has helper methods for this.
    
    # Get the transform from global to ego-vehicle frame
    ego_pose_token = nusc.get('sample_data', sample_data_record['token'])['ego_pose_token']
    ego_pose = nusc.get('ego_pose', ego_pose_token)
    
    # Get the transform from ego-vehicle to camera sensor frame
    calibrated_sensor = nusc.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
    
    # Convert nuScenesBox (in global frame) to the camera frame
    box.translate(-np.array(ego_pose['translation']))
    box.rotate(Quaternion(ego_pose['rotation']).inverse)
    
    box.translate(-np.array(calibrated_sensor['translation']))
    box.rotate(Quaternion(calibrated_sensor['rotation']).inverse)

    # 2. Extract Fields in Camera Frame
    center_cam = box.center.tolist()  # [X, Y, Z] in Camera Frame (Metric Depth Z is center_cam[2])
    pose_R = box.rotation_matrix.tolist() # 3x3 Rotation Matrix (pose)
    dimensions = box.wlh.tolist() # [L, W, H]
    
    # Get the 8 corners in the Camera Frame
    corners_global = box.corners() # 3x8 matrix (X, Y, Z) in Camera Frame
    bbox3D_cam = corners_global.T.tolist() # 8x3 list for the mapper

    # 3. Virtual Depth Calculation
    K = np.array(calibrated_sensor['camera_intrinsic'])
    # Assume horizontal focal length f = K[0, 0]
    f = K[0, 0] 
    metric_depth_z = center_cam[2]
    
    # Virtual Depth: z_v = z * (f_v / f)
    depth_virtual = metric_depth_z * (VIRTUAL_FOCAL_LENGTH_FV / f)
    
    # 4. 2D Bounding Box Calculation (Projection)
    # This is a simplification; in Omni3D, the 2D box is from the original COCO/nuScenes annotations.
    # We project the 3D box and take its min/max for a tight 2D box for robustness.
    corners_2d_proj = K @ corners_global
    corners_2d_proj[:2] /= corners_2d_proj[2] # Normalize by Z
    
    # Calculate 2D bounding box (x1, y1, x2, y2)
    u_min, u_max = int(np.min(corners_2d_proj[0])), int(np.max(corners_2d_proj[0]))
    v_min, v_max = int(np.min(corners_2d_proj[1])), int(np.max(corners_2d_proj[1]))

    # Clamp to image boundaries (for nuScenes: 1600x900)
    W, H = 1600, 900 
    u_min, u_max = np.clip([u_min, u_max], 0, W)
    v_min, v_max = np.clip([v_min, v_max], 0, H)
    
    bbox_2d = [u_min, v_min, u_max, v_max]

    return {
        'bbox': bbox_2d,
        'bbox_mode': BoxMode.XYXY_ABS, # Detectron2 constant for (x1, y1, x2, y2)
        'center_cam': center_cam,
        'dimensions': dimensions,
        'pose': pose_R, # 3x3 rotation matrix R_cam
        'bbox3D_cam': bbox3D_cam,
        'K': K.tolist(), # Required intrinsic matrix
        'depth_virtual': depth_virtual, # THE VIRTUAL DEPTH TARGET
        'focal_length': f,
        'metric_depth': metric_depth_z
    }


# --- 3. MAIN SCRIPT LOGIC ---

def create_omni3d_nuscenes_data(nusc_root: str, version: str, output_path: str):
    print(f"Loading nuScenes dataset version: {version} from {nusc_root}")
    nusc = NuScenes(version=version, dataroot=nusc_root, verbose=True)

    dataset_dicts: List[Dict] = []
    
    # Iterate through all keyframes (samples)
    for sample_idx, sample in enumerate(nusc.sample):
        
        # An Omni3D entry is per camera image/frame
        for cam_name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                         'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
            
            sd_token = sample['data'][cam_name]
            sd_record = nusc.get('sample_data', sd_token)
            
            # --- Per-Frame Info ---
            calibrated_sensor = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            K_matrix = np.array(calibrated_sensor['camera_intrinsic'])

            # This is the single dictionary entry for one image/frame
            frame_dict = {
                "img_path": os.path.join(nusc.dataroot, sd_record['filename']),
                "image_id": sd_token, # Unique ID for this frame
                "dataset_id": 0, # Omni3D ID for nuScenes (must be confirmed)
                "K": K_matrix,
                "height": sd_record['height'],
                "width": sd_record['width'],
                "annotations": []
            }
            
            # --- Annotation Processing ---
            for ann_token in sample['anns']:
                ann_record = nusc.get('sample_annotation', ann_token)
                nu_class_name = ann_record['category_name'].split('.')[-1] # e.g., 'vehicle.car' -> 'car'

                # 1. Category Mapping
                if nu_class_name not in OMNI3D_CLASS_MAP:
                    continue # Skip classes not in the 98 Omni3D target classes

                omni3d_id = OMNI3D_CLASS_MAP[nu_class_name]
                
                # 2. Get 3D Box in Global Frame (NuScenesBox object)
                # This is the standard way to get a Box object from the devkit
                box = NuScenesBox(
                    ann_record['translation'], 
                    ann_record['size'], 
                    Quaternion(ann_record['rotation'])
                )
                
                # 3. Transform and Calculate Metrics
                try:
                    # Transform the GLOBAL box into the LOCAL CAMERA frame
                    ann_data = transform_box_to_camera_frame(box, sd_record, nusc)
                except Exception as e:
                    # Skip if transformation fails (e.g., box behind camera)
                    print(f"Skipping annotation {ann_token} in {sd_token}: {e}")
                    continue
                
                # 4. Final Omni3D Annotation Structure
                ann_dict = {
                    "category_id": omni3d_id,
                    "iscrowd": 0, # nuScenes does not typically have crowd labels
                    "ignore": False, # Use nuScenes' flags if available
                    
                    # Required by DatasetMapper3D (and derived from ann_data)
                    "bbox": ann_data['bbox'],
                    "bbox_mode": ann_data['bbox_mode'],
                    "center_cam": ann_data['center_cam'],
                    "dimensions": ann_data['dimensions'],
                    "pose": ann_data['pose'],
                    "bbox3D_cam": ann_data['bbox3D_cam'],
                    
                    # Omni3D-specific target (virtual depth)
                    "depth_virtual": ann_data['depth_virtual'] 
                }
                
                frame_dict["annotations"].append(ann_dict)
            
            dataset_dicts.append(frame_dict)
    
    # --- 4. Save the Final Data ---
    print(f"\nTotal frames converted: {len(dataset_dicts)}")
    print(f"Saving converted data to {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset_dicts, f)
    
    print("Conversion complete. Data is now in Omni3D structure (DA3D_pkls format).")


if __name__ == '__main__':
    # Ensure the required dependencies are installed:
    # pip install nuscenes-devkit pyquaternion detectron2 numpy
    # Note: Detectron2 is only needed for the BoxMode constant.
    
    # This runs the main data preparation function
    create_omni3d_nuscenes_data(NUSCENES_ROOT, NUSCENES_VERSION, OUTPUT_PKL_PATH)