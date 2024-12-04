import os
import cv2
import mediapipe as mp
import json
from tqdm import tqdm

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_keypoints(image_path, max_num_hands=2):
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    keypoints_dict = {"left": [], "right": []}
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            # determine if the hand is left or right and store accordingly
            hand_label = handedness.classification[0].label.lower()  # "left" or "right"
            keypoints_dict[hand_label] = keypoints
    
    hands.close()
    return keypoints_dict

def extract_keypoints_from_dataset(data_dir, output_file):
    keypoints_data = {}

    for image_file in tqdm(os.listdir(data_dir)):
        if image_file.endswith(".jpg"):
            image_id = os.path.splitext(image_file)[0]  
            input_path = os.path.join(data_dir, image_file)
            
            # detect keypoints
            keypoints_dict = detect_keypoints(input_path)
            keypoints_data[image_id] = keypoints_dict
    

    with open(output_file, 'w') as f:
        json.dump(keypoints_data, f, indent=4)
    print(f"keypoints saved to {output_file}")

if __name__ == "__main__":
    splits = ['train', 'val', 'test']
    base_dir = 'processed_data'  
    output_dir = 'annotations' 

    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        data_dir = os.path.join(base_dir, split)
        output_file = os.path.join(output_dir, f"{split}/keypoints.json")
        print(f"processing {split} data...")
        extract_keypoints_from_dataset(data_dir, output_file)
