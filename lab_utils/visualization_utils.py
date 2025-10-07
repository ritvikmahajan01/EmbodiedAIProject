"""
2D Visualization Utilities

This module contains functions for visualizing detection results.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict


def visualize_2d_detections(image: Image.Image, 
                           detections: List[Dict], 
                           save_path: str = None,
                           show_plot: bool = True) -> None:
    """Visualize 2D detection results on the image."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
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

def visualize_sam_proposals_grid(image: Image.Image, 
                                 proposals: List[Dict],
                                 x_points: np.ndarray,
                                 y_points: np.ndarray,
                                 save_path: str = None):
    """
    Visualize SAM proposals with the grid points overlaid.
    
    Args:
        image: Original RGB image
        proposals: List of proposal dicts with 'mask', 'point', 'confidence'
        x_points: X coordinates of grid points
        y_points: Y coordinates of grid points
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    img_array = np.array(image)
    
    # 1. Original image with grid points
    axes[0].imshow(img_array)
    axes[0].set_title(f'Grid Points ({len(x_points)}x{len(y_points)})', fontsize=14)
    
    # Plot all grid points
    for x in x_points:
        for y in y_points:
            axes[0].plot(x, y, 'ro', markersize=8, alpha=0.6)
    
    # Highlight points that generated proposals
    for proposal in proposals:
        px, py = proposal['point']
        axes[0].plot(px, py, 'go', markersize=12, markeredgewidth=2, 
                    markeredgecolor='yellow', alpha=0.8)
    
    axes[0].axis('off')
    axes[0].legend(['All grid points', 'Generated proposals'], loc='upper right')
    
    # 2. Segmentation masks overlay
    axes[1].imshow(img_array)
    axes[1].set_title(f'Segmentation Masks ({len(proposals)} objects)', fontsize=14)
    
    # Create colored overlay
    overlay = np.zeros_like(img_array, dtype=np.float32)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(proposals)))
    
    for idx, proposal in enumerate(proposals):
        mask = proposal['mask']
        color = colors[idx][:3]
        
        # Apply mask with transparency
        for c in range(3):
            overlay[:, :, c][mask] = color[c] * 255
    
    # Blend overlay with original image
    blended = img_array.astype(np.float32) * 0.5 + overlay * 0.5
    axes[1].imshow(blended.astype(np.uint8))
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"SAM PROPOSAL STATISTICS")
    print(f"{'='*60}")
    print(f"Grid size: {len(x_points)} x {len(y_points)} = {len(x_points)*len(y_points)} points")
    print(f"Proposals generated: {len(proposals)}")
    print(f"Success rate: {len(proposals)/(len(x_points)*len(y_points))*100:.1f}%")
    
    if proposals:
        areas = [p['area'] for p in proposals]
        confidences = [p['confidence'] for p in proposals]
        print(f"\nArea range: {min(areas)} - {max(areas)} pixels")
        print(f"Mean area: {np.mean(areas):.0f} pixels")
        print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"Mean confidence: {np.mean(confidences):.3f}")
    print(f"{'='*60}\n")


