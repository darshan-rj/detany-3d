import numpy as np
import json
import os
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import torch

def get_noa_annotations(image_path: str):
    """
    Loads 3D bounding box annotations for a given NOA image path.
    Annotation format is assumed to be a list of dicts with 'bbox_3d'.
    'bbox_3d' is [x, y, z, w, h, l, yaw].
    Loads 3D bounding box annotations for a given NOA image path. The file structure
    is assumed to be flattened by `dataset_preparation/noa_prep.py`.

    - Image path format: .../images/pinhole/SCENE_FRAME_CAM.jpeg
    - Annotation path format: .../annotations/bb3d/SCENE_FRAME_bb3d.json
    """
    NOA_ANN_DIR = './Traces_NOA_structured/annotations/bb3d'
    if 'NOA' not in image_path:
        return None

    try:
        # Construct annotation path from image path
        # e.g., .../images/pinhole/scene_frame_cam.jpeg -> .../annotations/bb3d/scene_frame_cam.json
        # The file structure is flattened by `dataset_preparation/noa_prep.py`.
        # Image:  <output_dir>/images/pinhole/SCENE_FRAME_CAM.jpeg
        # Anno:   <output_dir>/annotations/bb3d/SCENE_FRAME_bb3d.json
        img_basename = os.path.basename(image_path)
        # The annotation filename is the same as the image filename, but with 'bb3d.json' instead of '.jpeg'
        # The annotation filename does not contain camera identifiers like _FL, _FC, etc.
        # The annotation filename does not contain camera identifiers like _FR, _FL, _FC, etc.
        # We need to remove it from the image name before creating the annotation name.
        # e.g., LBVS730_..._000293_FL.jpeg -> LBVS730_..._000293_FT.json
        base, _ = os.path.splitext(img_basename)
        ann_filename = base.rsplit('_', 1)[0] + '_FT.json' # Corrected based on user feedback
        print(f"Annotation FileName: {ann_filename}")
        ann_path = os.path.join(NOA_ANN_DIR, ann_filename)

        print(f"Annotation File Path: {ann_path}")
        if not os.path.exists(ann_path):
            print(f"Annotation file not found: {ann_path}")
            return None

        with open(ann_path, 'r') as f:
            annotations = json.load(f)

        # Handle formats where annotations are inside an "objects" key
        if isinstance(annotations, dict) and "objects" in annotations:
            annotations = annotations["objects"]

        if not isinstance(annotations, list):
            print(f"Annotations for {ann_path} are not in a list format.")
            return None

        gt_boxes = []
        for obj in annotations:
            box_data = None
            if isinstance(obj, dict):
                # New format: {"geometry": {"box": [...]}}
                if 'geometry' in obj and isinstance(obj['geometry'], dict) and 'box' in obj['geometry']:
                    box_data = obj['geometry']['box']
                # Original format: {"box": [...]}}
                elif 'box' in obj:
                    box_data = obj['box']

            if box_data and isinstance(box_data, list) and len(box_data) == 7:
                gt_boxes.append(box_data)
        
        print(f"GT boxes: {gt_boxes}")
        return gt_boxes if gt_boxes else None

    except (json.JSONDecodeError, TypeError, OSError, KeyError) as e:
        print(f"Error loading or parsing NOA annotation for {image_path}: {e}")
        return None

def calculate_iou_3d_aabb(box1, box2):
    """
    Calculate 3D IoU between two axis-aligned bounding boxes.
    box format: [x, y, z, w, h, l, yaw] - yaw is ignored for AABB.
    """
    # Convert to x1,y1,z1,x2,y2,z2
    b1_x1, b1_y1, b1_z1 = box1[0] - box1[3]/2, box1[1] - box1[4]/2, box1[2] - box1[5]/2 # x-w/2, y-h/2, z-l/2
    b1_x2, b1_y2, b1_z2 = box1[0] + box1[3]/2, box1[1] + box1[4]/2, box1[2] + box1[5]/2 # x+w/2, y+h/2, z+l/2
    b2_x1, b2_y1, b2_z1 = box2[0] - box2[3]/2, box2[1] - box2[4]/2, box2[2] - box2[5]/2 # x-w/2, y-h/2, z-l/2
    b2_x2, b2_y2, b2_z2 = box2[0] + box2[3]/2, box2[1] + box2[4]/2, box2[2] + box2[5]/2 # x+w/2, y+h/2, z+l/2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_z1 = max(b1_z1, b2_z1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    inter_z2 = min(b1_z2, b2_z2)

    inter_vol = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1) * max(0, inter_z2 - inter_z1)
    
    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]
    
    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-6)
    return iou

def calculate_metrics(pred_boxes_3d, gt_boxes, iou_threshold=0.25):
    """
    Calculates precision, recall, and average IoU for 3D bounding boxes.
    """
    # Convert pred_boxes_3d tensor to list for consistency in return type
    # Convert pred_boxes_3d tensor to a list of lists
    if isinstance(pred_boxes_3d, torch.Tensor):
        pred_boxes_list = pred_boxes_3d.cpu().numpy().tolist()
    else:
        pred_boxes_list = pred_boxes_3d

    if not gt_boxes or not pred_boxes_list:
        return {"precision": 0, "recall": 0, "avg_iou": 0, "true_positives": 0, "false_positives": 0, "false_negatives": len(gt_boxes or [])}

    num_pred = len(pred_boxes_list)
    num_gt = len(gt_boxes)

    if num_gt == 0 and num_pred == 0:
        return {"precision": 1.0, "recall": 1.0, "avg_iou": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0}
    if not gt_boxes or not pred_boxes_list:
        return {"precision": 0, "recall": 0, "avg_iou": 0, "true_positives": 0, "false_positives": num_pred, "false_negatives": num_gt}

    # Create IoU matrix
    iou_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = calculate_iou_3d_aabb(pred_boxes_list[i], gt_boxes[j])

    # Use Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    true_positives = 0
    matched_iou_sum = 0
    
    # Find matches above the IoU threshold
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            true_positives += 1
            matched_iou_sum += iou_matrix[r, c]

    false_positives = num_pred - true_positives
    false_negatives = num_gt - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    avg_iou = matched_iou_sum / true_positives if true_positives > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "avg_iou": avg_iou,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }