# feature_extraction.py
import json
import numpy as np
import argparse

def extract_features(keypoints):
    keypoints_np = np.array(keypoints)
    distances = []
    for i in range(len(keypoints_np)):
        for j in range(i + 1, len(keypoints_np)):
            distances.append(np.linalg.norm(keypoints_np[i] - keypoints_np[j]))
    return distances

def save_features(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    keypoints = data.get("keypoints", [])
    if keypoints:
        features = extract_features(keypoints)
        with open(output_path, 'w') as f:
            json.dump({"features": features}, f)
        print(f"features saved to {output_path}")
    else:
        print("no keypoints found in the input file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to keypoints JSON file')
    parser.add_argument('--output', type=str, required=True, help='path to save features')
    args = parser.parse_args()

    save_features(args.input, args.output)
