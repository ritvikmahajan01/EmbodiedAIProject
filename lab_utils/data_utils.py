"""
Data loading and processing utilities for 3D Semantic Object Mapping Lab.

This module contains functions for loading and validating RGB-D frame data,
camera intrinsics, camera poses, and aligning temporal data from ARKitScenes dataset.
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation
import open3d as o3d


def get_frame_list(config, frame_skip=1, max_frames=25, start_frame_name=None):
    files = sorted(os.listdir(config.RGB_PATH))
    frames = []

    # If start_frame_name is provided, find its index
    if start_frame_name is not None:
        try:
            start_idx = next(i for i, f in enumerate(files) if f == start_frame_name)
            files = files[start_idx:]  # start from that frame
        except StopIteration:
            raise ValueError(f"Start frame {start_frame_name} not found in {config.RGB_PATH}")

    for i, f in enumerate(files[::frame_skip]):
        if i >= max_frames:
            break

        timestamp = os.path.splitext(f)[0]

        # Load intrinsics for this frame
        try:
            K, image_size = load_camera_intrinsics(config.INTRINSICS_PATH, f)
        except Exception as e:
            print(f"Skipping {f}, failed to load intrinsics: {e}")
            continue

        # Load camera pose
        camera_poses = load_camera_poses(config.TRAJ_FILE_PATH)
        camera_pose = camera_poses.get(timestamp)
        if camera_pose is None:
            # Try to find the closest timestamp
            timestamp_float = float(str(timestamp).split("_")[-1])
            pose_timestamps = [(t, abs(timestamp_float - float(t))) for t in camera_poses.keys()]
            pose_timestamps.sort(key=lambda x: x[1])
            if pose_timestamps and pose_timestamps[0][1] < 0.5:  # or whatever your tolerance is
                closest_timestamp = pose_timestamps[0][0]
                camera_pose = camera_poses[closest_timestamp]
            else:
                print(f"No close camera pose found for {timestamp}, using identity.")
                camera_pose = np.eye(4)

        frames.append({
            "frame_name": f,
            "filename": f,
            "timestamp": timestamp,
            "rgb_path": os.path.join(config.RGB_PATH, f),
            "depth_path": os.path.join(config.DEPTH_PATH, f.replace(".jpg", ".png")),
            "camera_intrinsics": K,
            "intrinsics_size": image_size,
            "camera_pose": camera_pose
        })

    return frames



def load_camera_intrinsics(intrinsics_path: str, frame_name: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load camera intrinsics for a specific frame."""
    intrinsics_name = frame_name.replace('.png', '.pincam')
    intrinsic_file = os.path.join(intrinsics_path, intrinsics_name)

    with open(intrinsic_file, 'r') as f:
        params = f.readlines()[0].strip().split()

    width, height = int(params[0]), int(params[1])
    fx, fy = float(params[2]), float(params[3])
    cx, cy = float(params[4]), float(params[5])

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K, (width, height)


def load_camera_poses(traj_file: str) -> Dict[str, np.ndarray]:
    """Load camera poses from trajectory file."""
    pose_dict = {}

    with open(traj_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 7:
                continue

            timestamp = parts[0]
            rot_vec = np.array(parts[1:4], dtype=float)
            translation = np.array(parts[4:7], dtype=float)

            pose_matrix = np.identity(4)
            pose_matrix[:3, :3] = Rotation.from_rotvec(rot_vec).as_matrix()
            pose_matrix[:3, 3] = translation

            pose_dict[timestamp] = pose_matrix

    return pose_dict


def validate_and_align_frame_data(frames_list: List[Dict],
                                camera_poses: Dict[str, np.ndarray],
                                rgb_path: str,
                                depth_path: str,
                                intrinsics_path: str,
                                timestamp_tolerance: float = 0.1) -> List[Dict]:
    """Validate and align RGB frames, depth images, and camera poses to ensure consistency."""
    print(f"Validating and aligning {len(frames_list)} frames...")

    aligned_frames = []
    skipped_count = 0

    for frame_data in frames_list:

        frame_name = frame_data['filename']
        timestamp = frame_data['timestamp']

        print("frame_name: ", frame_name)
        
        try:
            # Check if all required files exist
            rgb_file = os.path.join(rgb_path, frame_name)
            depth_file = os.path.join(depth_path, frame_name)
            intrinsics_file = os.path.join(intrinsics_path, frame_name.replace('.png', '.pincam'))

            files_exist = {
                'rgb': os.path.exists(rgb_file),
                'depth': os.path.exists(depth_file),
                'intrinsics': os.path.exists(intrinsics_file)
            }

            if not all(files_exist.values()):
                skipped_count += 1
                continue

            # Find matching camera poseso
            camera_pose = camera_poses.get(timestamp)
            
            if camera_pose is None:
                timestamp_float = float(timestamp)
                pose_timestamps = [(t, abs(timestamp_float - float(t))) for t in camera_poses.keys()]
                pose_timestamps.sort(key=lambda x: x[1])

                if pose_timestamps and pose_timestamps[0][1] < timestamp_tolerance:
                    closest_timestamp = pose_timestamps[0][0]
                    camera_pose = camera_poses[closest_timestamp]
                else:

                    print(f"Skipped frame {timestamp} because no camera pose within tolerance {timestamp_tolerance}")
                    skipped_count += 1
                    continue
            
            # Load and validate camera intrinsics
            try:
                camera_intrinsics, image_size = load_camera_intrinsics(intrinsics_path, frame_name)
            except Exception:
                skipped_count += 1
                continue

            # Create Open3D intrinsics object
            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                image_size[0], image_size[1],
                camera_intrinsics[0, 0],
                camera_intrinsics[1, 1],
                camera_intrinsics[0, 2],
                camera_intrinsics[1, 2]
            )

            aligned_frame = {
                'frame_name': frame_name,
                'timestamp': timestamp,
                'rgb_path': rgb_file,
                'depth_path': depth_file,
                'camera_pose': camera_pose,
                'camera_intrinsics': camera_intrinsics,
                'intrinsics_o3d': o3d_intrinsics,
                'image_size': image_size,
                'original_index': frame_data.get('original_index', -1)
            }

            aligned_frames.append(aligned_frame)

        except Exception as e:
            print(f"Skipped, exception {e}")
            skipped_count += 1
            continue

    print(f"Aligned {len(aligned_frames)} frames (skipped {skipped_count})")
    return aligned_frames