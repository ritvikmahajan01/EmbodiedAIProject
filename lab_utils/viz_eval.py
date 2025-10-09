"""
Visualization and evaluation utilities for 3D Semantic Object Mapping Lab.

This module contains functions for 3D scene visualization using Rerun and
evaluation metrics for comparing detections with ground truth.
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from typing import Dict, List, Optional
import open3d as o3d


def visualize_3d_scene_bbox_results(point_cloud: o3d.geometry.PointCloud = None,
                               detections_3d: List[Dict] = None,
                               raw_detections_3d: List[Dict] = None,  # NEW: Raw detections before merging
                               gt_annotations: List[Dict] = None,
                               gt_mesh: o3d.geometry.PointCloud = None,
                               show_ground_truth: bool = True,
                               show_gt_mesh: bool = False,
                               show_object_pointclouds: bool = False,
                               show_raw_detections: bool = False,  # NEW: Default False for compatibility
                               title: str = "3D Scene Visualization",
                               config = None) -> None:
    """
    Unified visualization function for all levels (GT, E, C, A).
    Enhanced to show both raw and merged detections for Level E.
    """
    # Use config dimensions if provided, otherwise use defaults
    width = config.RERUN_WIDTH if config else 1600
    height = config.RERUN_HEIGHT if config else 800

    # Initialize Rerun with consistent recording ID
    rr.init("3d_scene_analysis")
    print(f"Visualizing: {title}")

    # Log TSDF point cloud (for Level E, C, A)
    if point_cloud is not None and len(point_cloud.points) > 0:
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None

        if colors is not None and len(colors) > 0:
            colors_uint8 = (colors * 255).astype(np.uint8)
            rr.log("world/environment/tsdf_pointcloud",
                   rr.Points3D(points, colors=colors_uint8, radii=0.008))
        else:
            
            # Color by height if no RGB data
            z_values = points[:, 2]
            z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min())

            height_colors = np.zeros((len(points), 3), dtype=np.uint8)
            height_colors[:, 0] = (z_normalized * 255).astype(np.uint8)
            height_colors[:, 1] = ((1 - np.abs(z_normalized - 0.5) * 2) * 255).astype(np.uint8)
            height_colors[:, 2] = ((1 - z_normalized) * 255).astype(np.uint8)

            rr.log("world/environment/tsdf_pointcloud",
                   rr.Points3D(points, colors=height_colors, radii=0.008))

    # Log RAW 3D object detections (before merging) - NEW SECTION
    if show_raw_detections and raw_detections_3d and len(raw_detections_3d) > 0:
        raw_detection_colors = {
            'chair': [255, 100, 100],    # Light red
            'table': [100, 255, 100],    # Light green
            'sofa': [100, 100, 255],     # Light blue
            'bed': [255, 200, 100],      # Light orange
            'shelf': [255, 150, 200],    # Light pink
            'pillow': [200, 100, 255],   # Light purple
            'window': [100, 255, 255],   # Light cyan
            'basketball': [255, 255, 100] # Light yellow
        }

        raw_centers = []
        raw_sizes = []
        raw_colors = []
        raw_labels = []

        for i, detection in enumerate(raw_detections_3d):
            bbox_3d = np.array(detection['bbox_3d_world'])
            label = detection['label']
            score = detection['score']

            # Use lighter colors to distinguish from merged detections
            color = raw_detection_colors.get(label, [200, 200, 200])

            # Calculate bounding box center and size
            min_bounds = np.min(bbox_3d, axis=0)
            max_bounds = np.max(bbox_3d, axis=0)
            box_center = (min_bounds + max_bounds) / 2
            box_size = max_bounds - min_bounds

            raw_centers.append(box_center)
            raw_sizes.append(box_size)
            raw_colors.append(color)
            raw_labels.append(f"Raw_{label}_{i} ({score:.2f})")

        # Log all raw bounding boxes
        if raw_centers:
            rr.log(
                "world/detections/raw_bboxes",
                rr.Boxes3D(
                    centers=raw_centers,
                    sizes=raw_sizes,
                    colors=raw_colors,
                    labels=raw_labels
                )
            )

    # Log MERGED 3D object detections (updated with new path)
    if detections_3d and len(detections_3d) > 0:
        detection_class_colors = {
            'chair': [255, 0, 0],      # Bright red (merged)
            'table': [0, 255, 0],      # Bright green (merged)
            'sofa': [0, 0, 255],       # Bright blue (merged)
            'bed': [255, 165, 0],      # Bright orange (merged)
            'stool': [128, 0, 128],
            'cabinet': [165, 42, 42],
            'shelf': [255, 192, 203],  # Bright pink (merged)
            'pillow': [128, 0, 128],   # Purple (merged)
            'window': [0, 255, 255],   # Cyan (merged)
            'basketball': [255, 105, 180] # Hot pink (merged)
        }

        detection_centers = []
        detection_sizes = []
        detection_colors = []
        detection_labels = []

        for i, detection in enumerate(detections_3d):
            bbox_3d = np.array(detection['bbox_3d_world'])
            label = detection['label']
            score = detection['score']

            # Use bright colors for merged detections
            color = detection_class_colors.get(label, [255, 255, 255])

            # Calculate bounding box center and size
            min_bounds = np.min(bbox_3d, axis=0)
            max_bounds = np.max(bbox_3d, axis=0)
            box_center = (min_bounds + max_bounds) / 2
            box_size = max_bounds - min_bounds

            detection_centers.append(box_center)
            detection_sizes.append(box_size)
            detection_colors.append(color)

            # Enhanced label for merged detections
            label_text = f"Merged_{label} ({score:.2f})"
            if 'merge_info' in detection:
                merge_count = detection['merge_info']['num_detections_merged']
                label_text += f" [x{merge_count}]"
            elif 'num_observations' in detection:
                label_text += f" [{detection['num_observations']} obs]"

            detection_labels.append(label_text)

            # Log object point clouds if requested (Level C feature)
            if show_object_pointclouds and 'pointcloud' in detection:
                pc_points = detection['pointcloud']
                pc_colors = detection.get('colors')

                if pc_colors is not None:
                    if pc_colors.dtype != np.uint8:
                        pc_colors = (pc_colors * 255).astype(np.uint8)
                else:
                    pc_colors = np.array([color] * len(pc_points), dtype=np.uint8)

                rr.log(
                    f"world/detections/object_{label}_{i}/pointcloud",
                    rr.Points3D(pc_points, colors=pc_colors, radii=0.01)
                )

        # Log merged bounding boxes
        if detection_centers:
            rr.log(
                "world/detections/merged_bboxes",
                rr.Boxes3D(
                    centers=detection_centers,
                    sizes=detection_sizes,
                    colors=detection_colors,
                    labels=detection_labels
                )
            )

    # Log ground truth data
    if show_ground_truth and (gt_annotations or gt_mesh):
        gt_class_colors = {
            'chair': [255, 100, 100],
            'table': [100, 255, 100],
            'sofa': [100, 100, 255],
            'bed': [255, 200, 100],
            'cabinet': [200, 150, 100],
            'shelf': [255, 150, 200],
            'door': [150, 150, 150],
        }

        # Log ground truth mesh (only if explicitly requested, e.g., in GT mode)
        if show_gt_mesh and gt_mesh is not None and len(gt_mesh.points) > 0:
            points = np.asarray(gt_mesh.points)

            if gt_mesh.has_colors():
                colors = np.asarray(gt_mesh.colors)
                colors_uint8 = (colors * 255).astype(np.uint8)
                rr.log("ground_truth/mesh", rr.Points3D(points, colors=colors_uint8, radii=0.005))
            else:
                gt_mesh_color = np.full((len(points), 3), [255, 215, 0], dtype=np.uint8)
                rr.log("ground_truth/mesh", rr.Points3D(points, colors=gt_mesh_color, radii=0.005))

        # Log ground truth bounding boxes
        if gt_annotations:
            annotations_by_class = {}
            for ann in gt_annotations:
                class_name = ann['label']
                if class_name not in annotations_by_class:
                    annotations_by_class[class_name] = []
                annotations_by_class[class_name].append(ann)

            for class_name, class_annotations in annotations_by_class.items():
                class_centers = []
                class_sizes = []
                class_color = gt_class_colors.get(class_name, [200, 200, 200])
                class_colors = []
                class_labels = []

                for i, ann in enumerate(class_annotations):
                    corners = np.array(ann['corners'])

                    min_bounds = np.min(corners, axis=0)
                    max_bounds = np.max(corners, axis=0)
                    box_center = (min_bounds + max_bounds) / 2
                    box_size = max_bounds - min_bounds

                    class_centers.append(box_center)
                    class_sizes.append(box_size)
                    class_colors.append(class_color)
                    class_labels.append(f"GT_{class_name}_{i}")

                if class_centers:
                    rr.log(
                        f"ground_truth/bboxes/{class_name}",
                        rr.Boxes3D(
                            centers=class_centers,
                            sizes=class_sizes,
                            colors=class_colors,
                            labels=class_labels
                        )
                    )

    # Add coordinate frame for reference
    frame_size = 1.0
    rr.log(
        "world/coordinate_frame",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[frame_size, 0, 0], [0, frame_size, 0], [0, 0, frame_size]],
            colors=[[255, 100, 100], [100, 255, 100], [100, 100, 255]],
            labels=["X (1m)", "Y (1m)", "Z (1m)"]
        )
    )

    # --- Blueprint to collapse right and bottom panels ---
    # This blueprint defines the layout of the viewer. Here, we specify a 3D view
    # and explicitly set the state of the Selection and Time panels to "collapsed".
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(origin="/"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
    rr.send_blueprint(blueprint)

    # Show the embedded viewer with configurable dimensions
    rr.notebook_show(width=width, height=height)
    print("3D visualization complete")


def evaluate_simple_iou(detections: List[Dict],
                        ground_truth: List[Dict],
                        confidence_threshold: float = 0.3) -> Dict:
    """
    Simple evaluation: compute mean IoU between detections and GT.
    """
    # Filter by confidence
    detections = [d for d in detections if d['score'] >= confidence_threshold]

    if not detections or not ground_truth:
        return {
            'num_detections': len(detections),
            'num_ground_truth': len(ground_truth),
            'mean_iou': 0.0
        }

    # For each GT box, find best matching detection
    best_ious = []

    for gt in ground_truth:
        gt_corners = np.array(gt['corners'])
        gt_min, gt_max = np.min(gt_corners, axis=0), np.max(gt_corners, axis=0)

        best_iou = 0.0
        for det in detections:
            # Only consider same class
            if det['label'] != gt['label']:
                continue

            det_corners = np.array(det['bbox_3d_world'])
            det_min, det_max = np.min(det_corners, axis=0), np.max(det_corners, axis=0)

            # Compute IoU
            inter_min = np.maximum(gt_min, det_min)
            inter_max = np.minimum(gt_max, det_max)

            if np.all(inter_min < inter_max):
                inter_vol = np.prod(inter_max - inter_min)
                gt_vol = np.prod(gt_max - gt_min)
                det_vol = np.prod(det_max - det_min)
                union_vol = gt_vol + det_vol - inter_vol
                iou = inter_vol / union_vol if union_vol > 0 else 0
                best_iou = max(best_iou, iou)

        best_ious.append(best_iou)

    return {
        'num_detections': len(detections),
        'num_ground_truth': len(ground_truth),
        'mean_iou': np.mean(best_ious),
        'median_iou': np.median(best_ious),
        'matched_gt': sum(1 for iou in best_ious if iou > 0.1)
    }


def evaluate_level_results(detections: List[Dict],
                           gt_annotations: List[Dict],
                           level_name: str,
                           confidence_threshold: float = 0.3,
                           required_miou: float = 0.14) -> Dict:
    """
    Unified evaluation function for all levels.
    """
    print("\n" + "="*50)
    print(f"{level_name.upper()} EVALUATION")
    print("="*50)

    if not detections or not gt_annotations:
        print("Cannot perform evaluation - missing detections or ground truth")
        return {
            'level': level_name,
            'num_detections': len(detections) if detections else 0,
            'num_ground_truth': len(gt_annotations) if gt_annotations else 0,
            'mean_iou': 0.0,
            'passed': False,
            'error': 'Missing data'
        }

    eval_results = evaluate_simple_iou(detections, gt_annotations, confidence_threshold)

    # Display results
    print(f"Detections: {eval_results['num_detections']}")
    print(f"Ground Truth: {eval_results['num_ground_truth']}")
    print(f"Mean IoU: {eval_results['mean_iou']:.3f}")
    print(f"Median IoU: {eval_results['median_iou']:.3f}")
    print(f"GT Matched (IoU>0.1): {eval_results['matched_gt']}/{eval_results['num_ground_truth']}")

    # Check pass/fail
    passed = eval_results['mean_iou'] > required_miou

    if passed:
        print(f"✓ PASSED: mIoU {eval_results['mean_iou']:.3f} > {required_miou} requirement")
    else:
        print(f"✗ NOT PASSED: mIoU {eval_results['mean_iou']:.3f} < {required_miou} requirement")

    eval_results.update({
        'level': level_name,
        'passed': passed,
        'required_miou': required_miou,
        'confidence_threshold': confidence_threshold
    })

    return eval_results