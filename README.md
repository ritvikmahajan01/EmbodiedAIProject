# Object-Based World Representations with Foundation Models

## Description

This project explores **object-based scene representations** for understanding and tracking. Instead of building dense volumetric (voxel) maps like in **ConceptFusion**, which are complex and memory-heavy, our approach represents the scene purely through **objects** — each described semantically and tracked over time.  

We will use two datasets, **RGB-D dataset** (e.g., ARKit Scenes) for static scenes, and **THUD++** for dynamic scenes, to simulate realistic indoor environments. 

---
## Instructions on how to run the code
The project consists of two main Jupyter notebooks:

- full_pipeline_with_3d_vis.ipynb demonstrates static tracking for world representation and includes the 3D visualization and DepthAnything analysis.

- DynamicTracker.ipynb demonstrates static and dynamic tracking query based.
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

## Extensions

- **Depth Estimation from RGB**  
  - Use a monocular depth estimation model with RGB images: DepthAnythingv2.  

- **Dynamic Environments**  
  - Handle moving objects.  
  - Associate objects across different times and locations.  

- **Open Vocabulary Understanding**  
  - Extend detection and tracking to an **open vocabulary** setting.  
---

## References

[1] Example work using LVLMs for text-based open-vocabulary object detection.
