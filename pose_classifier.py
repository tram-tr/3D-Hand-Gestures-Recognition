import cv2
import mediapipe as mp
from src.feature_extraction import extract_features
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

labels = {
    'ROM01': 'No interaction between 2 hands',
    'ROM02': 'Interaction between 2 Hands',
    'ROM03': 'Left hand (LT) with no occlusions',
    'ROM04': 'Left hand (LT) with occlusion',
    'ROM05': 'Right hand (LT) with no occlusions',
    'ROM06': 'Right hand (RT) with occlusion',
    'ROM07': 'Right-hand finger occlusions',
    'ROM08': 'Left-hand finger occlusions',
    'ROM09': 'Interaction fingers touching'
}

hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
]


class HandPoseClassifier3D:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model, self.label_encoder = pickle.load(f)

    def preprocess_features(self, features):
        distances = features["distances"]
        angles = features["angles"]
        mass_center_distances = features["mass_center_distances"]
        feature_vector = distances + angles + mass_center_distances
        max_features_length = 500 
        padded_vector = feature_vector + [0] * (max_features_length - len(feature_vector))
        return np.array(padded_vector).reshape(1, -1)

    def render_3d(self, keypoints):
        plt.cla() 
        ax = plt.axes(projection='3d')
        keypoints = np.array(keypoints)
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='r', marker='o')
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        plt.pause(0.01)  

    def main(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )

        cap = cv2.VideoCapture(0)
        plt.ion() 
        fig = plt.figure()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (324, 512))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            dynamic_frame = np.zeros_like(frame_bgr)

            keypoints_dict = {"left": [], "right": []}
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    hand_label = handedness.classification[0].label.lower()  # "left" or "right"
                
                    keypoints_dict[hand_label] = keypoints

                    # render 3D Keypoints
                    self.render_3d(keypoints)

                    # dynamic keypoints to the blank canvas
                    for kp1, kp2 in hand_connections:
                        if kp1 < len(keypoints) and kp2 < len(keypoints):  
                            x1, y1 = int(keypoints[kp1][0] * dynamic_frame.shape[1]), int(keypoints[kp1][1] * dynamic_frame.shape[0])
                            x2, y2 = int(keypoints[kp2][0] * dynamic_frame.shape[1]), int(keypoints[kp2][1] * dynamic_frame.shape[0])
                            cv2.line(dynamic_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    for kp in keypoints:
                        x, y = int(kp[0] * dynamic_frame.shape[1]), int(kp[1] * dynamic_frame.shape[0])
                        cv2.circle(dynamic_frame, (x, y), 5, (0, 255, 0), -1)

                    distances, angles, mass_center_distances = extract_features(keypoints, 314, 512)
                    features = {
                        "distances": distances,
                        "angles": angles,
                        "mass_center_distances": mass_center_distances
                    }
                    feature_vector = self.preprocess_features(features)
                    prediction = self.model.predict(feature_vector)
                    predicted_label = self.label_encoder.inverse_transform(prediction)[0]
                    
                    cv2.putText(
                        frame_bgr,
                        f"{labels[predicted_label]}",
                        (10, 30 if hand_label == "left" else 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            combined_frame = np.hstack((frame_bgr, dynamic_frame))
            cv2.imshow('Hand Pose Classifier and Visualization', combined_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

        cap.release()
        cv2.destroyAllWindows()
        plt.ioff() 
        plt.show()

if __name__ == "__main__":
    model_path = "models/best_xgboost.pkl"
    visualizer = HandPoseClassifier3D(model_path)
    visualizer.main()
