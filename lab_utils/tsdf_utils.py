"""
TSDF (Truncated Signed Distance Function) utilities for 3D Semantic Object Mapping Lab.

This module contains functions for building TSDF point clouds from RGB-D frames
using Open3D's volume integration pipeline.
"""

import gc
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from typing import Optional

# Import from other lab utilities
from .data_utils import load_camera_poses, get_frame_list, validate_and_align_frame_data


def build_tsdf_point_cloud(config,
                           max_frames_for_mapping: int = 1000,
                           use_cached: bool = True) -> Optional[o3d.geometry.PointCloud]:
    """
    Build TSDF point cloud from RGB-D frames with integrated volume processing.
    """
    # Check if we already have a TSDF point cloud cached
    if use_cached and 'tsdf_point_cloud' in globals() and globals()['tsdf_point_cloud'] is not None:
        print("Using cached TSDF point cloud")
        return globals()['tsdf_point_cloud']

    print("Building TSDF point cloud...")

    try:
        # Load and validate frame data
        camera_poses = load_camera_poses(config.TRAJ_FILE_PATH)
        frames_metadata = get_frame_list(config.RGB_PATH, config.TSDF_CONFIG['frame_skip'])

        aligned_frames = validate_and_align_frame_data(
            frames_metadata, camera_poses,
            config.RGB_PATH, config.DEPTH_PATH, config.INTRINSICS_PATH,
            timestamp_tolerance=0.1
        )

        if not aligned_frames:
            print("No aligned frames found for TSDF generation!")
            return None

        # Prepare frames for TSDF
        frames_for_mapping = aligned_frames[:max_frames_for_mapping]
        total_frames = len(frames_for_mapping)
        target_frames = min(config.TSDF_CONFIG['max_frames'], total_frames)
        mapping_stride = max(1, total_frames // target_frames)
        frames_for_tsdf = frames_for_mapping[::mapping_stride]

        print(f"TSDF integration: using {len(frames_for_tsdf)} frames")
        print(f"  Volume: {config.TSDF_CONFIG['volume_length']}m³")
        print(f"  Resolution: {config.TSDF_CONFIG['resolution']}³ voxels")
        print(f"  Batch size: {config.TSDF_CONFIG['batch_size']} frames")

        # Clear memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Create TSDF volume
        volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=config.TSDF_CONFIG['volume_length'],
            resolution=config.TSDF_CONFIG['resolution'],
            sdf_trunc=0.08,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=[-config.TSDF_CONFIG['volume_length']/2, -config.TSDF_CONFIG['volume_length']/2, -config.TSDF_CONFIG['volume_length']/2]
        )

        # Process frames in batches
        total_batches = (len(frames_for_tsdf) + config.TSDF_CONFIG['batch_size'] - 1) // config.TSDF_CONFIG['batch_size']
        print(f"Processing {len(frames_for_tsdf)} frames in {total_batches} batches...")

        for batch_idx in range(total_batches):
            start_idx = batch_idx * config.TSDF_CONFIG['batch_size']
            end_idx = min(start_idx + config.TSDF_CONFIG['batch_size'], len(frames_for_tsdf))
            batch_frames = frames_for_tsdf[start_idx:end_idx]

            print(f"  Batch {batch_idx + 1}/{total_batches}: frames {start_idx}-{end_idx-1}")

            for i, frame_data in enumerate(tqdm(batch_frames, desc=f"Batch {batch_idx+1}", leave=False)):
                try:
                    rgb_raw = o3d.io.read_image(frame_data['rgb_path'])
                    depth_raw = o3d.io.read_image(frame_data['depth_path'])

                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        rgb_raw, depth_raw,
                        depth_scale=config.TSDF_CONFIG['depth_scale'],
                        depth_trunc=config.TSDF_CONFIG['depth_trunc'],
                        convert_rgb_to_intensity=False
                    )

                    volume.integrate(rgbd, frame_data['intrinsics_o3d'], frame_data['camera_pose'])

                except Exception as e:
                    print(f"Warning: Could not integrate frame {start_idx + i}: {e}")
                    continue

            # Memory cleanup after each batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Extract and process mesh
        print("Extracting mesh from TSDF volume...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        print("Converting mesh to point cloud...")
        pcd = mesh.sample_points_uniformly(number_of_points=200000)

        # Downsample for efficient visualization
        print(f"Downsampling from {len(pcd.points)} points...")
        pcd_voxel = pcd.voxel_down_sample(voxel_size=config.TSDF_CONFIG['voxel_size'])

        target_points = 100000
        if len(pcd_voxel.points) > target_points:
            point_cloud = pcd_voxel.farthest_point_down_sample(target_points)
        else:
            point_cloud = pcd_voxel

        print(f"Final TSDF point cloud: {len(point_cloud.points)} points")

        # Cache it globally for reuse
        globals()['tsdf_point_cloud'] = point_cloud

        return point_cloud

    except Exception as e:
        print(f"Error building TSDF point cloud: {e}")
        import traceback
        traceback.print_exc()
        return None