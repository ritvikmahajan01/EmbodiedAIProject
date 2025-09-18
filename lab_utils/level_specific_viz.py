"""
Level-Specific Visualization Functions

This module contains complex visualization functions that are specific to each lab level.
These are moved here to reduce notebook clutter while maintaining full functionality.
"""

import os
from typing import Dict, List, Optional, Tuple
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
import torch
from tqdm import tqdm

from .data_utils import get_frame_list, load_camera_poses, validate_and_align_frame_data
from .tsdf_utils import build_tsdf_point_cloud
from .model_loaders import load_sam_model, load_clip_model, load_owlv2_model


def visualize_level_a_example(config, frame_index: int = 29) -> Dict:
    """Level A visualization: SAM proposals with CLIP semantic similarity overlays."""
    print("=" * 50)
    print("LEVEL A: SEMANTIC SEGMENT ANALYSIS")
    print("=" * 50)
    print(f"Processing frame {frame_index} with text queries: 'pillow' vs 'sofa'")
    
    # Load models
    print("Loading models...")
    sam_model, sam_processor, device = load_sam_model(model_size='base')
    clip_model, clip_processor, _ = load_clip_model(device=device)
    
    # Get frame data
    frames_metadata = get_frame_list(config.RGB_PATH, config.LEVEL_A_CONFIG['frame_skip'])
    if frame_index >= len(frames_metadata):
        frame_index = len(frames_metadata) // 2
        print(f"Adjusted frame index to {frame_index}")
    
    frame_name = frames_metadata[frame_index]['filename']
    rgb_path = os.path.join(config.RGB_PATH, frame_name)
    
    if not os.path.exists(rgb_path):
        print(f"Frame not found: {rgb_path}")
        return {}
    
    image = Image.open(rgb_path).convert("RGB")
    print(f"Frame: {frame_name}, size: {image.size}")
    
    # Import the required functions from the notebook context
    try:
        from __main__ import generate_sam_proposals, extract_clip_features_from_segment
    except ImportError:
        print("ERROR: Could not import required functions from notebook.")
        print("Please ensure generate_sam_proposals and extract_clip_features_from_segment are defined in the notebook.")
        return {}
    
    # Generate SAM proposals
    # print(f"Generating SAM proposals with {config.LEVEL_A_CONFIG['grid_size']}x{config.LEVEL_A_CONFIG['grid_size']} grid...")
    proposals = generate_sam_proposals(
        image,
        sam_model,
        sam_processor,
        device,
        grid_size=config.LEVEL_A_CONFIG['grid_size'],
        confidence_threshold=config.LEVEL_A_CONFIG['sam_confidence_threshold']
    )
    
    print(f"Generated {len(proposals)} proposals above confidence threshold")
    
    if not proposals:
        print("No proposals generated - try lowering sam_confidence_threshold")
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"No SAM Proposals - {frame_name}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return {'frame_name': frame_name, 'proposals': [], 'has_proposals': False}
    
    # Extract CLIP features for all proposals
    print("Extracting CLIP features...")
    proposal_features = []
    for proposal in proposals:
        features = extract_clip_features_from_segment(
            image,
            proposal['mask'],
            clip_model,
            clip_processor,
            device,
            padding_ratio=config.LEVEL_A_CONFIG['padding_ratio_image_crops'] 
        )
        if features is not None:
            proposal_features.append({
                'proposal': proposal,
                'features': features
            })
    
    print(f"Successfully extracted features for {len(proposal_features)}/{len(proposals)} proposals")
    
    if not proposal_features:
        print("No CLIP features extracted - check segment quality")
        return {'frame_name': frame_name, 'proposals': proposals, 'proposal_features': []}
    
    # Define text queries
    text_queries = ['pillow', 'sofa']
    
    # Compute text embeddings
    print("Computing text embeddings...")
    text_embeddings = {}
    for query in text_queries:
        inputs = clip_processor(text=[query], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        text_embeddings[query] = text_features.cpu().numpy().squeeze()
    
    # Compute similarities
    similarities = {}
    for query in text_queries:
        query_similarities = []
        for pf in proposal_features:
            similarity = np.dot(text_embeddings[query], pf['features'])
            query_similarities.append(similarity)
        similarities[query] = np.array(query_similarities)
    
    # Visualization with color legend
    image_gray = image.convert('L')
    image_gray_rgb = np.stack([np.array(image_gray)] * 3, axis=-1)
    
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], hspace=0.3)
    
    # Main visualization subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    viridis_cmap = cm.get_cmap('viridis')
    
    for idx, query in enumerate(text_queries):
        ax = ax1 if idx == 0 else ax2
        overlay = image_gray_rgb.astype(np.float32).copy()
        
        query_sims = similarities[query]
        if len(query_sims) > 0:
            sim_min, sim_max = query_sims.min(), query_sims.max()
            if sim_max > sim_min:
                sim_normalized = (query_sims - sim_min) / (sim_max - sim_min)
            else:
                sim_normalized = np.ones_like(query_sims) * 0.5
            
            for i, pf in enumerate(proposal_features):
                mask = pf['proposal']['mask']
                similarity_score = sim_normalized[i]
                color_rgba = viridis_cmap(similarity_score)
                color_rgb = np.array(color_rgba[:3]) * 255
                alpha = 0.7
                overlay[mask] = overlay[mask] * (1 - alpha) + color_rgb * alpha
        
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"Semantic Response: '{query}'\n"
                    f"Similarity range: [{query_sims.min():.3f}, {query_sims.max():.3f}]", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Add color legend spanning both columns
    cax = fig.add_subplot(gs[1, :])
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Semantic Similarity (Normalized)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add descriptive text labels on the colorbar
    cbar.ax.text(0.1, -0.8, 'Low\n(Dark Purple)', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=9)
    cbar.ax.text(0.9, -0.8, 'High\n(Yellow)', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nSemantic Similarity Analysis:")
    for query in text_queries:
        sims = similarities[query]
        print(f"  '{query}': mean={sims.mean():.3f}, max={sims.max():.3f}, std={sims.std():.3f}")
    
    return {
        'frame_name': frame_name,
        'frame_index': frame_index,
        'proposals': proposals,
        'proposal_features': proposal_features,
        'similarities': similarities,
        'text_queries': text_queries,
        'stats': {
            'total_proposals': len(proposals),
            'successful_features': len(proposal_features),
            'queries_tested': len(text_queries)
        }
    }


def query_and_visualize_semantic_grid(level_a_results: Dict,
                                     text_query: str,
                                     environment_pcd: o3d.geometry.PointCloud = None,
                                     config = None) -> None:
    """Query the semantic voxel grid with text and visualize results."""
    print(f"\nQuerying semantic grid with: '{text_query}'")
    
    voxel_grid = level_a_results['voxel_grid']
    clip_model = level_a_results['clip_model']
    clip_processor = level_a_results['clip_processor']
    device = level_a_results['device']
    
    # Query the grid
    voxel_centers, similarities = voxel_grid.query_text(
        text_query,
        clip_model,
        clip_processor,
        device
    )
    
    if len(voxel_centers) == 0:
        print("No voxels to visualize!")
        return
    
    print(f"Query results:")
    print(f"  Voxels with data: {len(voxel_centers)}")
    print(f"  Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
    print(f"  Mean similarity: {similarities.mean():.3f}")
    
    # Normalize similarities for visualization
    sim_normalized = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-6)
    
    # Initialize Rerun
    rr.init("level_a_semantic_mapping")
    
    width = config.RERUN_WIDTH if config else 1600
    height = config.RERUN_HEIGHT if config else 800
    
    # Clear and redraw
    rr.log("world", rr.Clear(recursive=True))
    
    # Log environment if available
    if environment_pcd is not None and len(environment_pcd.points) > 0:
        points = np.asarray(environment_pcd.points)
        colors = np.full((len(points), 3), [200, 200, 200], dtype=np.uint8)
        rr.log("world/environment", 
               rr.Points3D(points, colors=colors, radii=0.005))
    
    # Create viridis colormap for semantic voxels
    colormap = cm.get_cmap('viridis')
    
    # Prepare voxel visualization
    voxel_colors = []
    voxel_radii = []
    
    for sim in sim_normalized:
        color_rgba = colormap(sim)
        color_rgb = (np.array(color_rgba[:3]) * 255).astype(np.uint8)
        voxel_colors.append(color_rgb)
        radius = 0.02 + sim * 0.03  # 2cm to 5cm based on similarity
        voxel_radii.append(radius)
    
    voxel_colors = np.array(voxel_colors)
    voxel_radii = np.array(voxel_radii)
    
    # Log semantic voxels
    rr.log("world/semantic_voxels",
           rr.Points3D(voxel_centers, colors=voxel_colors, radii=voxel_radii))
    
    # Add annotations
    rr.log("world/query_text",
           rr.TextDocument(f"Query: '{text_query}'\n"
                          f"Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]\n"
                          f"Occupied voxels: {len(voxel_centers)}"))
    
    # Add coordinate frame
    rr.log("world/coordinate_frame",
           rr.Arrows3D(
               origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
               labels=["X", "Y", "Z"]
           ))
    
    print("\nViridis colormap:")
    print("  Dark purple/blue: Low similarity")
    print("  Green: Medium similarity")  
    print("  Yellow: High similarity")
    
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(origin="/"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
    rr.send_blueprint(blueprint)
    
    # Show embedded viewer
    rr.notebook_show(width=width, height=height)


def visualize_level_c_example(config, frame_index: int = 65) -> Dict:
    """Educational visualization for Level C: Show OwlV2 + SAM on a single frame."""
    print("=" * 60)
    print("LEVEL C EXAMPLE VISUALIZATION")
    print("=" * 60)
    print(f"Demonstrating OwlV2 + SAM segmentation on frame {frame_index}")
    print(f"Target classes: {config.OBJECT_CLASSES}")
    
    # Load models
    print("\n1. Loading models...")
    owl_processor, owl_model, device = load_owlv2_model()
    sam_model, sam_processor, _ = load_sam_model(
        model_size=config.LEVEL_C_CONFIG['sam_model_size'], 
        device=device
    )
    
    # Get frame data
    print("\n2. Loading frame data...")
    camera_poses = load_camera_poses(config.TRAJ_FILE_PATH)
    frames_metadata = get_frame_list(config.RGB_PATH, config.LEVEL_C_CONFIG['frame_skip'])
    
    if frame_index >= len(frames_metadata):
        frame_index = len(frames_metadata) // 2
        print(f"Adjusted frame index to {frame_index}")
    
    # Align frames to get camera data
    aligned_frames = validate_and_align_frame_data(
        frames_metadata,
        camera_poses,
        config.RGB_PATH,
        config.DEPTH_PATH,
        config.INTRINSICS_PATH
    )
    
    if not aligned_frames or frame_index >= len(aligned_frames):
        print("No valid frames found!")
        return {}
    
    frame_data = aligned_frames[frame_index]
    frame_name = frame_data['frame_name']
    
    print(f"Selected frame: {frame_name}")
    
    # Import the detect_objects_in_frame function from the notebook context
    from __main__ import detect_objects_in_frame, segment_with_sam_bbox
    
    # Run OwlV2 Detection
    print("\n3. Running OwlV2 object detection...")
    detections_2d, image = detect_objects_in_frame(
        frame_data['rgb_path'],
        config.OBJECT_CLASSES,
        owl_processor,
        owl_model,
        device,
        threshold=config.LEVEL_C_CONFIG['detection_threshold']
    )
    
    print(f"Found {len(detections_2d)} detections above threshold {config.LEVEL_C_CONFIG['detection_threshold']}")
    
    if not detections_2d:
        print("No detections found. Try lowering detection_threshold or different frame_index")
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"No Detections Found - Frame {frame_name}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return {'frame_name': frame_name, 'detections_2d': [], 'segmentation_results': []}
    
    # Run SAM segmentation
    print("\n4. Running SAM segmentation...")
    segmentation_results = []
    
    for detection in detections_2d:
        mask = segment_with_sam_bbox(
            image,
            detection['bbox'],
            sam_model,
            sam_processor,
            device,
            confidence_threshold=config.LEVEL_C_CONFIG['sam_confidence_threshold']
        )
        
        if mask is not None:
            segmentation_results.append({
                'detection': detection,
                'mask': mask,
                'mask_area': np.sum(mask)
            })
    
    print(f"SAM segmentation: {len(segmentation_results)}/{len(detections_2d)} successful")
    
    # Visualize side-by-side
    print("\n5. Visualizing results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: OwlV2 detections
    ax1.imshow(image)
    ax1.set_title("OwlV2 Object Detections", fontsize=14, fontweight='bold')
    
    for det in detections_2d:
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=2, edgecolor='blue', facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f"{det['label']}: {det['score']:.2f}",
                bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7),
                fontsize=10, color='white', weight='bold')
    ax1.axis('off')
    
    # Right: SAM segmentations
    if segmentation_results:
        overlay = np.array(image).copy()
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
        
        for i, result in enumerate(segmentation_results):
            mask = result['mask']
            color = colors[i % len(colors)]
            overlay[mask] = overlay[mask] * 0.6 + np.array(color) * 0.4
        
        ax2.imshow(overlay)
        ax2.set_title("SAM Segmentation Masks", fontsize=14, fontweight='bold')
    else:
        ax2.imshow(image)
        ax2.set_title("No SAM Segmentations", fontsize=14, fontweight='bold')
    
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Summary
    results = {
        'frame_name': frame_name,
        'frame_index': frame_index,
        'detections_2d': detections_2d,
        'segmentation_results': segmentation_results,
        'stats': {
            'owl_detections': len(detections_2d),
            'successful_segmentations': len(segmentation_results),
            'success_rate': len(segmentation_results) / len(detections_2d) if detections_2d else 0
        }
    }
    
    print(f"\n" + "="*40)
    print("EXAMPLE VISUALIZATION SUMMARY")
    print("="*40)
    print(f"Frame: {frame_name}")
    print(f"OwlV2 detections: {results['stats']['owl_detections']}")
    print(f"SAM segmentations: {results['stats']['successful_segmentations']}")
    print(f"Success rate: {results['stats']['success_rate']:.1%}")
    
    return results


def visualize_2d_detections(image: Image.Image, 
                           detections: List[Dict], 
                           save_path: str = None,
                           show_plot: bool = True) -> None:
    """Visualize 2D detection results on the image."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Set proper axis limits
    ax.set_xlim(0, image.size[0])  # width
    ax.set_ylim(image.size[1], 0)  # height (inverted for image coordinates)
    
    unique_labels = list(set(d['label'] for d in detections))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), w, h, 
            linewidth=2,
            edgecolor=color_map[det['label']],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(
            x1, y1-5, 
            f"{det['label']} ({det['score']:.2f})",
            bbox=dict(boxstyle="round,pad=0.3", 
                     facecolor=color_map[det['label']], 
                     alpha=0.7),
            fontsize=10, color='white', weight='bold'
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_level_e_example(config, 
                              frame_index: int = 50,
                              show_depth_analysis: bool = True) -> Dict:
    """
    Educational visualization for Level E: Show OwlV2 detection on a single frame.
    """
    print("=" * 60)
    print("LEVEL E EXAMPLE VISUALIZATION")
    print("=" * 60)
    print(f"Demonstrating OwlV2 object detection on frame {frame_index}")
    print(f"Target classes: {config.OBJECT_CLASSES}")
    
    # Load models
    print("\n1. Loading OwlV2 model...")
    processor, model, device = load_owlv2_model()
    
    # Get frame data
    print("\n2. Loading frame data...")
    camera_poses = load_camera_poses(config.TRAJ_FILE_PATH)
    frames_metadata = get_frame_list(config.RGB_PATH, config.LEVEL_E_CONFIG['frame_skip'])
    
    if frame_index >= len(frames_metadata):
        frame_index = len(frames_metadata) // 2
        print(f"Adjusted frame index to {frame_index}")
    
    # Align frames to get camera data
    aligned_frames = validate_and_align_frame_data(
        frames_metadata,
        camera_poses,
        config.RGB_PATH,
        config.DEPTH_PATH,
        config.INTRINSICS_PATH
    )
    
    if not aligned_frames or frame_index >= len(aligned_frames):
        print("No valid frames found!")
        return {}
    
    frame_data = aligned_frames[frame_index]
    frame_name = frame_data['frame_name']
    
    print(f"Selected frame: {frame_name}")
    
    # Import the detect_objects_in_frame function from the notebook context
    from __main__ import detect_objects_in_frame
    
    # Run OwlV2 Detection
    print("\n3. Running OwlV2 object detection...")
    detections_2d, image = detect_objects_in_frame(
        frame_data['rgb_path'],
        config.OBJECT_CLASSES,
        processor,
        model,
        device,
        threshold=config.LEVEL_E_CONFIG['detection_threshold']
    )
    
    print(f"Found {len(detections_2d)} detections above threshold {config.LEVEL_E_CONFIG['detection_threshold']}")
    
    # Display detection details
    for i, det in enumerate(detections_2d):
        bbox = det['bbox']
        print(f"  {i+1}. {det['label']}: {det['score']:.3f} at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    
    if not detections_2d:
        print("No detections found. Try:")
        print("- Lowering DETECTION_THRESHOLD")
        print("- Trying a different frame_index")
        
        # Still show the image
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"No Detections Found - Frame {frame_name}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        return {
            'frame_name': frame_name,
            'detections_2d': [],
            'has_detections': False
        }
    
    # Visualize 2D Detections
    print("\n4. Visualizing 2D detection results...")
    visualize_2d_detections(image, detections_2d, show_plot=True)
    
    # Summary
    results = {
        'frame_name': frame_name,
        'frame_index': frame_index,
        'detections_2d': detections_2d,
        'has_detections': len(detections_2d) > 0,
        'image_size': image.size,
        'detection_classes': list(set(d['label'] for d in detections_2d))
    }
    
    print(f"\n" + "="*40)
    print("EXAMPLE VISUALIZATION SUMMARY")
    print("="*40)
    print(f"Frame: {frame_name}")
    print(f"Detections: {len(detections_2d)}")
    print(f"Classes found: {results['detection_classes']}")
    
    return results