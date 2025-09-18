"""
Ground truth loading utilities for 3D Semantic Object Mapping Lab.

This module contains functions for loading and processing ground truth annotations
and meshes from ARKitScenes dataset, including 3D bounding box computation.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
import open3d as o3d


def compute_box_3d_gt(scale, transform, rotation):
    """Calculate the 8 corners of a 3D bounding box."""
    scales = [i / 2.0 for i in scale]
    l, h, w = scales
    center = np.reshape(transform, (3,))
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d_local = np.vstack([x_corners, y_corners, z_corners])

    corners_3d_rotated = np.dot(rotation.T, corners_3d_local)
    corners_3d = corners_3d_rotated.T + center
    return corners_3d


def load_ground_truth_annotations(annotation_file: str,
                                 allowed_classes: Optional[List[str]] = None) -> List[Dict]:
    """Load ground truth annotations from ARKitScenes annotation file."""
    print(f"Loading ground truth annotations from: {annotation_file}")

    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return []

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    processed_annotations = []

    for label_info in data.get("data", []):
        try:
            label = label_info.get('label', 'unknown')

            if allowed_classes is not None and label not in allowed_classes:
                continue

            obb_data = label_info["segments"]["obbAligned"]
            scale = np.array(obb_data["axesLengths"])
            centroid = np.array(obb_data["centroid"])
            rotation = np.array(obb_data["normalizedAxes"]).reshape(3, 3)

            box_corners = compute_box_3d_gt(scale.tolist(), centroid, rotation)

            processed_annotations.append({
                'label': label,
                'corners': box_corners.tolist(),
                'centroid': centroid.tolist(),
                'scale': scale.tolist(),
                'rotation': rotation.tolist()
            })

        except Exception:
            continue

    print(f"Successfully processed {len(processed_annotations)} ground truth annotations")
    return processed_annotations


def load_ground_truth_mesh(mesh_file: str,
                          downsample_points: int = 100000) -> Optional[o3d.geometry.PointCloud]:
    """Load ground truth mesh and convert to point cloud."""
    print(f"Loading ground truth mesh from: {mesh_file}")

    if not os.path.exists(mesh_file):
        print(f"Mesh file not found: {mesh_file}")
        return None

    try:
        mesh = o3d.io.read_triangle_mesh(mesh_file)

        if len(mesh.vertices) == 0:
            print("Mesh has no vertices")
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors

        if len(pcd.points) > downsample_points:
            sampling_ratio = downsample_points / len(pcd.points)
            pcd = pcd.random_down_sample(sampling_ratio=sampling_ratio)

        print(f"Loaded mesh with {len(pcd.points)} points")
        return pcd

    except Exception:
        print("Error loading mesh")
        return None


def load_ground_truth_data(scene_id: str,
                          base_path: str,
                          config: dict) -> tuple:
    """Complete function to load ground truth data for a scene."""
    mesh_file = os.path.join(base_path, f"../{scene_id}_3dod_mesh.ply")
    annotation_file = os.path.join(base_path, f"../{scene_id}_3dod_annotation.json")

    gt_annotations = []
    gt_mesh = None

    if config.get('show_annotations', True):
        gt_annotations = load_ground_truth_annotations(
            annotation_file,
            config.get('allowed_classes')
        )

    if config.get('show_mesh', True):
        gt_mesh = load_ground_truth_mesh(
            mesh_file,
            config.get('mesh_downsample_points', 75000)
        )

    return gt_annotations, gt_mesh