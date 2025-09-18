# Object-Based World Representations with Foundation Models

## Idea Description

This project explores **object-based scene representations** for understanding and tracking, inspired by the suggested project *“Explore pure object-based world representations using foundation models”*.  

Instead of building dense volumetric (voxel) maps like in **ConceptFusion**, which are complex and memory-heavy, our approach represents the scene purely through **objects** — each described semantically and tracked over time.  

We will use an **RGB-D dataset** (e.g., ARKit Scenes) to simulate realistic indoor environments. As a starting point, we will build upon **Lab 2, Part E** (excluding voxelization).

---

## Processing Pipeline

At each timestep (image), we will:

1. **Segmentation**  
   - Segment images using a foundation model like **Segment Anything (SAM)**.

2. **Semantic Encoding**  
   - Recognize and encode each segmented object with vector embeddings from a model such as **CLIP** or **DINO**.

3. **3D Projection**  
   - Project the object’s location into 3D using depth + camera intrinsics.  
   - Add this 3D position to the object’s embedding.

4. **Association Across Time**  
   - Match current objects with previously identified objects using **semantic embeddings + 3D positions**.

---

## Exploration Directions

We will experiment with different object representations:
- Pure semantic embeddings
- 3D bounding boxes
- Center-of-mass points

---

## Extensions

- **Depth Estimation from RGB**  
  - Use a monocular depth estimation model with RGB images.  
  - Compare with ground-truth depth data, both qualitatively and quantitatively (using object positions over time).

- **Dynamic Environments**  
  - Handle moving objects.  
  - Associate objects across different times and locations.  
  - Track object **trajectories**.

- **Open Vocabulary Understanding**  
  - Extend detection and tracking to an **open vocabulary** setting.  
  - Use text captions from large vision-language models (LVLMs), following approaches like [1].

---

## References

[1] Example work using LVLMs for text-based open-vocabulary object detection.
