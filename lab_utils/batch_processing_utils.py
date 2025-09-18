"""
Batch Processing Utility Functions

This module contains infrastructure code for batch processing frames
across multiple lab levels. These functions handle memory management
and batch processing logic.
"""

import gc
from typing import Dict, List, Tuple
import cv2
import torch
from tqdm import tqdm

from .model_loaders import load_owlv2_model, load_sam_model


def process_frames_with_sam(frames_for_detection: List[Dict],
                           config,
                           owl_processor, owl_model,
                           sam_processor, sam_model,
                           device: str) -> tuple:
    """Process frames with OwlV2 detection and SAM segmentation."""
    all_pointclouds = []
    frame_results = {}
    
    # Initialize statistics
    stats = {
        'frames_processed': 0,
        'frames_attempted': len(frames_for_detection),
        'total_2d_detections': 0,
        'total_sam_segments': 0,
        'total_pointclouds': 0,
        'detection_classes': set(),
        'processing_errors': 0
    }
    
    # Process frames in batches
    batch_size = config.LEVEL_C_CONFIG['sam_batch_size']
    total_batches = (len(frames_for_detection) + batch_size - 1) // batch_size
    
    print(f"Processing {len(frames_for_detection)} frames in {total_batches} batches of {batch_size}...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(frames_for_detection))
        batch_frames = frames_for_detection[start_idx:end_idx]
        
        for frame_data in tqdm(batch_frames, desc=f"Batch {batch_idx+1}/{total_batches}", leave=False):
            try:
                # Import the process_frame_with_sam function from the notebook context
                from __main__ import process_frame_with_sam
                
                # Process single frame
                frame_result = process_frame_with_sam(
                    frame_data, owl_processor, owl_model,
                    sam_processor, sam_model, device, config
                )
                
                if frame_result['pointclouds']:
                    frame_results[frame_result['frame_name']] = frame_result
                    all_pointclouds.extend(frame_result['pointclouds'])
                    
                    # Update statistics
                    stats['frames_processed'] += 1
                    stats['total_2d_detections'] += len(frame_result['detections_2d'])
                    stats['total_sam_segments'] += len(frame_result['segments'])
                    stats['total_pointclouds'] += len(frame_result['pointclouds'])
                    
                    for pc in frame_result['pointclouds']:
                        stats['detection_classes'].add(pc['label'])
                    
            except Exception as e:
                stats['processing_errors'] += 1
                print(f"Warning: Error processing frame {frame_data.get('frame_name', 'unknown')}: {e}")
                continue
        
        # Memory cleanup after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Frame processing complete: {stats['frames_processed']}/{stats['frames_attempted']} frames successful")
    if stats['processing_errors'] > 0:
        print(f"Processing errors: {stats['processing_errors']}")
    
    return all_pointclouds, frame_results, stats


def process_frames_in_batches(frames_for_detection: List[Dict],
                             config,
                             processor, model, device: str) -> tuple:
    """Process object detection frames in batches with consolidated statistics."""
    all_detections_3d_raw = []
    frame_results = {}
    
    # Initialize consolidated statistics
    stats = {
        'frames_processed': 0,
        'frames_attempted': len(frames_for_detection),
        'total_2d_detections': 0,
        'total_3d_detections_raw': 0,
        'detection_classes': set(),
        'processing_errors': 0
    }
    
    # Process frames in batches
    batch_size = config.LEVEL_E_CONFIG['owl_batch_size']
    total_batches = (len(frames_for_detection) + batch_size - 1) // batch_size
    
    print(f"Processing {len(frames_for_detection)} frames in {total_batches} batches of {batch_size}...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(frames_for_detection))
        batch_frames = frames_for_detection[start_idx:end_idx]
        
        for frame_data in tqdm(batch_frames, desc=f"Batch {batch_idx+1}/{total_batches}", leave=False):
            try:
                # Import functions from the notebook context
                from __main__ import detect_objects_in_frame, generate_3d_detections
                
                # Extract frame data
                frame_name = frame_data['frame_name']
                rgb_path = frame_data['rgb_path']
                depth_path = frame_data['depth_path']
                camera_pose = frame_data['camera_pose']
                camera_intrinsics = frame_data['camera_intrinsics']
                
                # Run 2D object detection
                detections_2d, image = detect_objects_in_frame(
                    rgb_path, 
                    config.OBJECT_CLASSES,
                    processor, 
                    model, 
                    device,
                    threshold=config.LEVEL_E_CONFIG['detection_threshold']
                )
                
                # Skip frames with no detections
                if not detections_2d:
                    continue
                
                # Load depth image
                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    stats['processing_errors'] += 1
                    continue
                
                # Generate 3D detections
                detections_3d_frame = generate_3d_detections(
                    detections_2d,
                    depth_image,
                    camera_intrinsics,
                    camera_pose,
                )
                
                # Store results
                frame_results[frame_name] = {
                    'detections_2d': detections_2d,
                    'detections_3d': detections_3d_frame,
                    'camera_pose': camera_pose.tolist(),
                    'camera_intrinsics': camera_intrinsics.tolist(),
                    'timestamp': frame_data['timestamp']
                }
                
                # Update consolidated statistics
                all_detections_3d_raw.extend(detections_3d_frame)
                stats['frames_processed'] += 1
                stats['total_2d_detections'] += len(detections_2d)
                stats['total_3d_detections_raw'] += len(detections_3d_frame)
                
                # Collect unique detection classes
                for det in detections_3d_frame:
                    stats['detection_classes'].add(det['label'])
                    
            except Exception as e:
                stats['processing_errors'] += 1
                print(f"Warning: Error processing frame {frame_data.get('frame_name', 'unknown')}: {e}")
                continue
        
        # Memory cleanup after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final statistics summary
    print(f"Batch processing complete: {stats['frames_processed']}/{stats['frames_attempted']} frames successful")
    if stats['processing_errors'] > 0:
        print(f"Processing errors: {stats['processing_errors']}")
    
    return all_detections_3d_raw, frame_results, stats