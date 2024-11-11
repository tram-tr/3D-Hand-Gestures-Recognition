import cv2
import mediapipe as mp

class HandKeypointVisualizer:
    def main(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45
        )

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (512, 314))
            
            # convert the image to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # convert the frame back to BGR for OpenCV visualization
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Hand Keypoint Visualizer', frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27:  # press 'ESC' to exit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    visualizer = HandKeypointVisualizer()
    visualizer.main()
