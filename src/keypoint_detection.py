import cv2
import mediapipe as mp
import json
import argparse

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
            
            # draw landmarks on the image for each detected hand
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    hands.close()
    return keypoints_dict, image

def save_keypoints_and_show(image_path, output_path):
    keypoints_dict, image_with_keypoints = detect_keypoints(image_path)
    
    json_output_path = output_path if output_path.endswith('.json') else output_path + '.json'
    with open(json_output_path, 'w') as f:
        json.dump(keypoints_dict, f)
    print(f"Keypoints saved to {json_output_path}")

    image_output_path = output_path if output_path.endswith('.jpg') else output_path + '.jpg'
    cv2.imwrite(image_output_path, image_with_keypoints)
    
    cv2.imshow("Detected Keypoints", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to input image')
    parser.add_argument('--output', type=str, required=True, help='path to save keypoints')
    args = parser.parse_args()

    save_keypoints_and_show(args.input, args.output)
