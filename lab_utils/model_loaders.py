"""
Model loading utilities for 3D Semantic Object Mapping Lab.

This module contains functions for loading and initializing the various
foundational vision models used in the lab: OwlV2, SAM, and CLIP.
"""

import torch
from typing import Tuple
from transformers import (
    Owlv2ForObjectDetection,
    Owlv2Processor,
    SamModel,
    SamProcessor,
    CLIPModel,
    CLIPProcessor
)


def load_owlv2_model(device: str = None) -> Tuple[object, object, str]:
    """Load the OwlV2 model and processor."""
    print("Loading OWLv2 model...")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    print(f"Model loaded on device: {device}")

    return processor, model, str(device)


def load_sam_model(model_size: str = "base", device: str = None) -> Tuple:
    """
    Load SAM model from Hugging Face using transformers.

    Args:
        model_size: Model size - 'base', 'large', 'huge'
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (model, processor, device)
    """
    print(f"Loading SAM model ({model_size})...")

    # Model configurations from HuggingFace
    model_configs = {
        'base': 'facebook/sam-vit-base',
        'large': 'facebook/sam-vit-large',
        'huge': 'facebook/sam-vit-huge'
    }

    model_id = model_configs.get(model_size, model_configs['base'])

    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    print(f"SAM model loaded on device: {device}")
    return model, processor, str(device)


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32", device: str = None):
    """
    Load CLIP model from HuggingFace.

    Args:
        model_name: CLIP model identifier from HuggingFace
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (model, processor, device)
    """
    print(f"Loading CLIP model: {model_name}...")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    print(f"CLIP model loaded on device: {device}")
    return model, processor, str(device)