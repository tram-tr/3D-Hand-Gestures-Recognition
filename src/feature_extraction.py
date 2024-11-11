import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

def normalize_keypoints(keypoints, image_width, image_height, center_index=0):
    # flatten the nested keypoints
    if len(keypoints) > 0 and isinstance(keypoints[0], list):
        keypoints = [kp for hand in keypoints for kp in hand]  

    # scale keypoints to the image dimensions
    scaled_keypoints = [(kp[0] * image_width, kp[1] * image_height, kp[2]) for kp in keypoints]

    # center keypoints around the specified index
    center_x, center_y, center_z = scaled_keypoints[center_index]
    normalized_keypoints = [(x - center_x, y - center_y, z - center_z) for x, y, z in scaled_keypoints]

    # normalize distances
    max_distance = max(np.linalg.norm([x, y, z]) for x, y, z in normalized_keypoints)
    normalized_keypoints = [(x / max_distance, y / max_distance, z / max_distance) for x, y, z in normalized_keypoints]

    return normalized_keypoints

def extract_features(keypoints, image_width, image_height):
    # normalize keypoints
    normalized_keypoints = normalize_keypoints(keypoints, image_width, image_height)
    
    # compute pairwise distances as features
    keypoints_np = np.array(normalized_keypoints)
    distances = []
    for i in range(len(keypoints_np)):
        for j in range(i + 1, len(keypoints_np)):
            distances.append(np.linalg.norm(keypoints_np[i] - keypoints_np[j]))
    return distances

def save_features(input_path, output_path, image_width, image_height):
    with open(input_path, 'r') as f:
        data = json.load(f)
    keypoints = data.get("keypoints", [])
    features = extract_features(keypoints, image_width, image_height)

    json_output_path = output_path if output_path.endswith('.json') else output_path + '.json'
    with open(json_output_path, 'w') as f:
        json.dump({"features": features}, f)
    print(f"features saved to {json_output_path}")

    visualize_features(features, output_path)

def visualize_features(features, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(features, marker='o', linestyle='-', color='b')
    plt.title('pairwise distances between keypoints')
    plt.xlabel('feature index')
    plt.ylabel('distance')
    plt.grid(True)

    if not output_path.endswith('.png'):
        output_path += '.png'
    plt.savefig(output_path)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to keypoints JSON file')
    parser.add_argument('--output', type=str, required=True, help='path to save features')
    parser.add_argument('--image_width', type=int, default=512, help='width of the image (default: 512)')
    parser.add_argument('--image_height', type=int, default=314, help='height of the image (default: 314)')
    args = parser.parse_args()

    save_features(args.input, args.output, args.image_width, args.image_height)
