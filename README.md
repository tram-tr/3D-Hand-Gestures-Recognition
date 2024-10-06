# Project Name: 3D Hand Gesture Recognition Using MediaPipe Keypoints

## Part 1: High-Level Solution (Revised)

The goal of this project is to create a 3D hand gesture recognition system using keypoints detected by MediaPipe. By utilizing MediaPipe's real-time hand-tracking capabilities, the focus will be shifted from developing a full 3D pose estimation model to building a classifier that works on top of pre-detected keypoints to recognize gestures. 

### Key Objectives (Revised):
1. **Keypoint Detection (MediaPipe):** Instead of building a custom 3D hand pose estimator from scratch, the system will use MediaPipe, which can detect 21 keypoints of the hand in real time from RGB input. This step will serve as the foundation for the project, handling the initial hand segmentation and keypoint extraction.

2. **Feature Extraction:** After obtaining the keypoints from MediaPipe, the next step is to extract features from these keypoints. These features will represent important hand joints such as the knuckles, fingertips, and wrist.

3. **Gesture Classification:** The primary focus will be on building a machine learning classifier to recognize hand gestures based on the extracted keypoints. The model can be trained to identify predefined gestures such as thumbs-up, OK sign, or other hand poses commonly used in human-computer interaction.

4. **Real-time Performance:** Since MediaPipe provides real-time hand tracking, the system should be able to process video streams at 15–30 frames per second (FPS). The classifier for gesture recognition will need to be optimized for speed to ensure real-time performance.

### Challenges and Learning Goals

1. **MediaPipe Keypoints as Features**:
   Instead of extracting raw 3D positions, this project will focus on utilizing the 2D keypoints detected by MediaPipe. The challenge will be to design features or feature transformations that can improve the gesture classification accuracy.
   

### Image Properties and Features

The project will use the 21 keypoints detected by MediaPipe, including:
- **Knuckles**
- **Fingertips**
- **Wrist**



