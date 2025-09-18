"""
2D Visualization Utilities

This module contains functions for visualizing detection results.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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