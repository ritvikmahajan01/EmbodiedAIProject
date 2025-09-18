"""
3D Object Detection Utilities

This module contains functions for processing OwlV2 outputs and merging detections.
"""

import numpy as np
from typing import Dict, List, Tuple
from PIL import Image
import torch
from .geometry_utils_3d import compute_3d_bbox_iou, extract_depth_region, create_3d_bbox_corners


def process_owlv2_outputs(outputs, processor, target_sizes, text_queries, threshold=0.1):
    """Process raw OwlV2 model outputs into detection format."""
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        detections.append({
            'bbox': box.tolist(),
            'score': float(score),
            'label': text_queries[label],
            'label_id': int(label)
        })
    
    return detections


def detect_objects_in_frame(image_path: str, 
                           text_queries: List[str],
                           processor, 
                           model, 
                           device: str,
                           threshold: float = 0.1) -> Tuple[List[Dict], Image.Image]:
    """Run OwlV2 object detection."""
    
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([[orig_height, orig_width]]).to(device)
    
    detections = process_owlv2_outputs(outputs, processor, target_sizes, text_queries, threshold)
    
    # Clip to image bounds
    for detection in detections:
        bbox = detection['bbox']
        bbox[0] = max(0, min(bbox[0], orig_width))   # x1
        bbox[1] = max(0, min(bbox[1], orig_height))  # y1
        bbox[2] = max(0, min(bbox[2], orig_width))   # x2
        bbox[3] = max(0, min(bbox[3], orig_height))  # y2
        detection['bbox'] = bbox
    
    return detections, image


def project_pixel_to_3d(center_pixel: List[float], depth: float, camera_intrinsics: np.ndarray) -> List[float]:
    """Project a single pixel + depth to 3D coordinates."""
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    center_u, center_v = center_pixel
    
    x_3d = (center_u - cx) * depth / fx
    y_3d = (center_v - cy) * depth / fy
    z_3d = depth
    
    return [float(x_3d), float(y_3d), float(z_3d)]


def generate_3d_detections(detections_2d: List[Dict],
                          depth_image: np.ndarray,
                          camera_intrinsics: np.ndarray,
                          camera_pose: np.ndarray) -> List[Dict]:
    """Generate 3D detections from 2D detections."""
    detections_3d = []
    
    for detection in detections_2d:
        valid_depths, depth_stats = extract_depth_region(
            detection['bbox'], 
            depth_image
        )
        
        if valid_depths is None or depth_stats['valid_pixels'] < 10:
            continue
        if depth_stats['mean'] < 0.2 or depth_stats['mean'] > 10.0:
            continue
        
        x1, y1, x2, y2 = detection['bbox']
        center_pixel = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        
        depth_center = np.median(valid_depths)
        
        center_3d_camera = project_pixel_to_3d(center_pixel, depth_center, camera_intrinsics)
        
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        width_3d = (x2 - x1) * depth_center / fx
        height_3d = (y2 - y1) * depth_center / fy
        depth_range = depth_stats['max'] - depth_stats['min']
        depth_3d = max(0.2, depth_range)
        
        bbox_3d_camera = create_3d_bbox_corners(center_3d_camera, width_3d, height_3d, depth_3d)
        
        camera_pose_inv = np.linalg.inv(camera_pose)
        
        center_3d_cam_hom = np.array(center_3d_camera + [1])
        center_3d_world = (camera_pose_inv @ center_3d_cam_hom)[:3].tolist()
        
        bbox_3d_cam = np.array(bbox_3d_camera)
        bbox_3d_cam_hom = np.concatenate([bbox_3d_cam, np.ones((8, 1))], axis=1)
        bbox_3d_world_hom = (camera_pose_inv @ bbox_3d_cam_hom.T).T
        bbox_3d_world = bbox_3d_world_hom[:, :3].tolist()
        
        detection_3d = {
            'label': detection['label'],
            'score': detection['score'],
            'bbox_2d': detection['bbox'],
            'center_3d_world': center_3d_world,
            'bbox_3d_world': bbox_3d_world,
            'depth_stats': depth_stats
        }
        
        detections_3d.append(detection_3d)
    
    return detections_3d


def find_overlapping_detections(detections_3d: List[Dict], class_name: str, iou_threshold: float = 0.4) -> List[List[int]]:
    """Find which detections of same class overlap significantly."""
    class_detections = [i for i, d in enumerate(detections_3d) if d['label'] == class_name]
    
    overlapping_pairs = []
    
    for i in range(len(class_detections)):
        for j in range(i + 1, len(class_detections)):
            idx1, idx2 = class_detections[i], class_detections[j]
            
            iou = compute_3d_bbox_iou(
                detections_3d[idx1]['bbox_3d_world'],
                detections_3d[idx2]['bbox_3d_world']
            )
            
            if iou >= iou_threshold:
                overlapping_pairs.append([idx1, idx2])
    
    return overlapping_pairs


def merge_detection_cluster(detection_indices: List[int], all_detections: List[Dict]) -> Dict:
    """Merge a cluster of overlapping detections into one."""
    cluster_detections = [all_detections[i] for i in detection_indices]
    
    all_corners = [det['bbox_3d_world'] for det in cluster_detections]
    all_scores = [det['score'] for det in cluster_detections]
    
    merged_corners = np.mean(all_corners, axis=0)
    avg_score = np.mean(all_scores)
    
    class_name = cluster_detections[0]['label']
    
    corners_array = np.array(merged_corners)
    center = np.mean(corners_array, axis=0)
    
    return {
        'label': class_name,
        'score': float(avg_score),
        'bbox_3d_world': merged_corners.tolist(),
        'center_3d_world': center.tolist(),
        'merge_info': {
            'num_detections_merged': len(detection_indices),
            'original_confidences': all_scores
        }
    }


def merge_overlapping_detections(detections_3d: List[Dict], 
                                iou_threshold: float = 0.4,
                                min_detections_for_merge: int = 2) -> List[Dict]:
    """Merge overlapping 3D detections."""
    if not detections_3d:
        return []
    
    print("Merging overlapping detections...")
    
    detections_by_class = {}
    for det in detections_3d:
        class_name = det['label']
        if class_name not in detections_by_class:
            detections_by_class[class_name] = []
        detections_by_class[class_name].append(det)
    
    merged_detections = []
    
    for class_name, class_detections in detections_by_class.items():
        if len(class_detections) < min_detections_for_merge:
            continue
        
        overlapping_pairs = find_overlapping_detections(detections_3d, class_name, iou_threshold)
        
        if not overlapping_pairs:
            continue
        
        processed = set()
        for pair in overlapping_pairs:
            idx1, idx2 = pair
            if idx1 not in processed and idx2 not in processed:
                cluster = set([idx1, idx2])
                
                expanded = True
                while expanded:
                    expanded = False
                    for other_pair in overlapping_pairs:
                        other_idx1, other_idx2 = other_pair
                        if other_idx1 in cluster and other_idx2 not in cluster:
                            cluster.add(other_idx2)
                            expanded = True
                        elif other_idx2 in cluster and other_idx1 not in cluster:
                            cluster.add(other_idx1)
                            expanded = True
                
                if len(cluster) >= min_detections_for_merge:
                    merged_detection = merge_detection_cluster(list(cluster), detections_3d)
                    merged_detections.append(merged_detection)
                    processed.update(cluster)
    
    print(f"Merged {len(detections_3d)} detections into {len(merged_detections)} objects")
    return merged_detections