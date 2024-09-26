# Project Name: 3D Hand Pose Estimation

## Part 1: High-Level Solution

The goal of this project is to create a system for 3D hand pose estimation using a single RGB video input. The system will determine the precise 3D positions of key hand joints—such as the knuckles, fingertips, and wrist—based on 2D data from a video stream. 

The challenge is predicting the depth of each hand joint from 2D video frames. While 2D images contain valuable spatial information, they don’t directly offer depth, so this needs to be inferred by the model. The project builds on a simpler version that focused on recognizing hand gestures (which initially what I planned to do but decided to modify it at the last minute), but this one adds complexity by estimating 3D poses.

### Key Objectives:
1. **Segmentation**:
   First, the system detect and segment the hand from the video frame. Since the InterHand2.6M dataset includes diverse hand poses, orientations, and interactions, the segmentation algorithm needs to be good enough to handle different backgrounds, lighting conditions, and hand placements. Therefore, isolating the hand accurately is important for further processing.

2. **Feature Extraction**:
   After segmenting the hand, the next step is to extract features from the image, specifically focusing on identifying the 2D locations of hand joints like the knuckles, fingertips, and wrist. These 2D keypoints will then be used to estimate the depth (z-axis), transforming the 2D input into 3D positions through machine learning models.

3. **3D Pose Estimation**:
   Estimating the 3D hand pose is the main challenge of this project. Deep learning models such as CNNs or Graph CNNs will be trained to predict the 3D coordinates of each key joint. The InterHand2.6M dataset, which includes 3D annotations for hand joints, will be instrumental in training the model to ensure it can predict 3D poses that are consistent across frames.

4. **Real-Time Performance**:
   Real-time performance is essential, so the system needs to process video streams at 15–30 frames per second. This will likely involve optimizing the model for speed without affecting the accuracy.

### Dataset

   The **[InterHand2.6M](https://mks0601.github.io/InterHand2.6M/)** dataset will be used to develop and evaluate the model. Introduced at **ECCV 2020**, this dataset is specifically designed for **3D interacting hand pose estimation** from a single RGB image. The dataset contains over 2.6 million frames with accurate 3D annotations for both single and interacting hand poses.

1. **Training (70%)**
   
2. **Validation (15%)**

3. **Testing (15%)**
   

### Challenges and Learning Goals

There are several challenges:

1. **2D to 3D Transformation**:
   One of the big challenges is estimating the 3D position of each hand joint from 2D video input, as RGB images don’t provide depth information directly. The model will need to learn how to infer the depth (z-axis) for each joint using the ground truth 3D data from the InterHand2.6M dataset.

2. **Occlusion Handling**:
   Hands often occlude themselves during gestures, especially when interacting with objects or other hands. The model must somehow handle these cases by predicting the locations of occluded joints probably based on anatomical knowledge.

3. **Real-Time Processing**:
   Since the system will be processing video in real time, it must be optimized to run efficiently without affecting the accuracy, so the goal here is to achieve a balance between speed and precision.

4. **Generalization**:
   The model should be able to generalize well across different lighting conditions, hand shapes, and skin tones.

### Image Properties and Features

The key features required for this project are the 3D coordinates of important hand joints, including:
- **Knuckles**
- **Fingertips**
- **Wrist**


09/26/24 Update:
Relevant paper:
https://arxiv.org/pdf/2005.04551v1 (using the same dataset)


