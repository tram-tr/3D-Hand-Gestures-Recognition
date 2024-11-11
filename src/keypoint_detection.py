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
    
    keypoints_list = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            keypoints_list.append(keypoints)
            # draw landmarks on the image for each detected hand
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    hands.close()
    return keypoints_list, image

def save_keypoints_and_show(image_path, output_path):
    keypoints, image_with_keypoints = detect_keypoints(image_path)
    # save keypoints to a JSON file
    json_output_path = output_path if output_path.endswith('.json') else output_path + '.json'
    with open(json_output_path, 'w') as f:
        json.dump({"keypoints": keypoints}, f)
    print(f"Keypoints saved to {json_output_path}")

    image_output_path = output_path if output_path.endswith('.jpg') else output_path + '.jpg'
    cv2.imwrite(image_output_path, image_with_keypoints)
    
    # show the image with keypoints
    cv2.imshow("detected keypoints", image_with_keypoints)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to input image')
    parser.add_argument('--output', type=str, required=True, help='path to save keypoints')
    args = parser.parse_args()

    save_keypoints_and_show(args.input, args.output)
