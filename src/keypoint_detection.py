# keypoint_detection.py
import cv2
import mediapipe as mp
import json
import os
import argparse

mp_hands = mp.solutions.hands

def detect_keypoints(image_path):
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    keypoints = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    
    hands.close()
    return keypoints

def save_keypoints(image_path, output_path):
    keypoints = detect_keypoints(image_path)
    if keypoints:
        with open(output_path, 'w') as f:
            json.dump({"keypoints": keypoints}, f)
        print(f"keypoints saved to {output_path}")
    else:
        print(" o hand detected in the image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to input image')
    parser.add_argument('--output', type=str, required=True, help='path to save keypoints')
    args = parser.parse_args()

    save_keypoints(args.input, args.output)
