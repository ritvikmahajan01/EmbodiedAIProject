"""
3D Geometry Utility Functions

This module contains pure 3D geometry and depth processing functions
that are used across multiple lab levels.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def extract_depth_region(bbox_2d: List[float], depth_image: np.ndarray, depth_scale: float = 1000.0) -> tuple:
    """Extract and validate depth values within bounding box."""
    x1, y1, x2, y2 = [int(coord) for coord in bbox_2d]
    h, w = depth_image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    
    depth_region = depth_image[y1:y2, x1:x2] / depth_scale
    valid_mask = (depth_region > 0.1) & (depth_region < 5.0)
    valid_depths = depth_region[valid_mask]
    
    if len(valid_depths) == 0:
        return None, {'min': 0, 'max': 0, 'mean': 0, 'valid_pixels': 0}
    
    return valid_depths, {
        'min': float(np.min(valid_depths)),
        'max': float(np.max(valid_depths)), 
        'mean': float(np.mean(valid_depths)),
        'valid_pixels': len(valid_depths)
    }


def create_3d_bbox_corners(center_3d: List[float], width_3d: float, height_3d: float, depth_3d: float) -> List[List[float]]:
    """Generate 8 corners of 3D bounding box around center."""
    half_w, half_h, half_d = width_3d/2, height_3d/2, depth_3d/2
    
    corners_relative = np.array([
        [-half_w, -half_h, -half_d],
        [+half_w, -half_h, -half_d],
        [-half_w, +half_h, -half_d],
        [+half_w, +half_h, -half_d],
        [-half_w, -half_h, +half_d],
        [+half_w, -half_h, +half_d],
        [-half_w, +half_h, +half_d],
        [+half_w, +half_h, +half_d],
    ])
    
    bbox_3d = corners_relative + np.array(center_3d)
    return bbox_3d.tolist()


def compute_3d_bbox_iou(bbox1: List[List[float]], bbox2: List[List[float]]) -> float:
    """Compute 3D IoU between two bounding boxes defined by 8 corners."""
    try:
        corners1 = np.array(bbox1)
        corners2 = np.array(bbox2)
        
        min1, max1 = np.min(corners1, axis=0), np.max(corners1, axis=0)
        min2, max2 = np.min(corners2, axis=0), np.max(corners2, axis=0)
        
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        
        if np.any(intersection_min >= intersection_max):
            return 0.0
        
        intersection_dims = intersection_max - intersection_min
        intersection_volume = np.prod(intersection_dims)
        
        volume1 = np.prod(max1 - min1)
        volume2 = np.prod(max2 - min2)
        union_volume = volume1 + volume2 - intersection_volume
        
        if union_volume <= 0:
            return 0.0
            
        return float(intersection_volume / union_volume)
        
    except Exception:
        return 0.0