import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

def normalize_keypoints(keypoints, image_width, image_height, center_index=0):
    # scale and normalize keypoints
    scaled_keypoints = [(kp[0] * image_width, kp[1] * image_height, kp[2]) for kp in keypoints]
    center_x, center_y, center_z = scaled_keypoints[center_index]
    normalized_keypoints = [(x - center_x, y - center_y, z - center_z) for x, y, z in scaled_keypoints]
    max_distance = max(np.linalg.norm([x, y, z]) for x, y, z in normalized_keypoints)
    normalized_keypoints = [(x / max_distance, y / max_distance, z / max_distance) for x, y, z in normalized_keypoints]
    return normalized_keypoints

def calculate_mass_center(keypoints):
    # calculate the mass center of the keypoints
    keypoints_np = np.array(keypoints)
    mass_center = keypoints_np.mean(axis=0)  # average x, y, z coordinates
    return mass_center

def extract_features(keypoints, image_width, image_height):
    # normalize keypoints and compute features
    normalized_keypoints = normalize_keypoints(keypoints, image_width, image_height)
    
    # compute pairwise euclidean distances
    keypoints_np = np.array(normalized_keypoints)
    distances = [np.linalg.norm(keypoints_np[i] - keypoints_np[j])
                 for i in range(len(keypoints_np)) for j in range(i + 1, len(keypoints_np))]

    # compute distances from the mass center
    mass_center = calculate_mass_center(keypoints_np)
    mass_center_distances = [np.linalg.norm(kp - mass_center) for kp in keypoints_np]

    # compute angles between consecutive keypoint vectors
    angles = []
    for i in range(1, len(keypoints_np) - 1):
        v1 = keypoints_np[i] - keypoints_np[i - 1]
        v2 = keypoints_np[i + 1] - keypoints_np[i]
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angles.append(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return distances, angles, mass_center_distances

def save_features(input_path, output_path, image_width, image_height):
    # load keypoints data from the input file
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    features = {}
    
    # extract features for left hand if available
    if "left" in data and data["left"]:
        left_keypoints = data["left"]
        distances, angles, mass_center_distances = extract_features(left_keypoints, image_width, image_height)
        features["left"] = {
            "distances": distances,
            "angles": angles,
            "mass_center_distances": mass_center_distances
        }
    
    # extract features for right hand if available
    if "right" in data and data["right"]:
        right_keypoints = data["right"]
        distances, angles, mass_center_distances = extract_features(right_keypoints, image_width, image_height)
        features["right"] = {
            "distances": distances,
            "angles": angles,
            "mass_center_distances": mass_center_distances
        }
    
    # save features to a json file
    json_output_path = output_path if output_path.endswith('.json') else output_path + '.json'
    with open(json_output_path, 'w') as f:
        json.dump(features, f)
    print(f"Features saved to {json_output_path}")

    # visualize features
    visualize_features(features, output_path)

def visualize_features(features, output_path):
    # plot features for the left hand
    if "left" in features:
        plt.figure(figsize=(10, 5))
        plt.plot(features["left"]["distances"], marker='o', linestyle='-', color='b')
        plt.title('Left Hand - Pairwise Distances Between Keypoints')
        plt.xlabel('Feature Index')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.savefig(output_path + '_left_distances.png')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(features["left"]["angles"], marker='o', linestyle='-', color='g')
        plt.title('Left Hand - Angles Between Keypoint Vectors')
        plt.xlabel('Feature Index')
        plt.ylabel('Angle (Radians)')
        plt.grid(True)
        plt.savefig(output_path + '_left_angles.png')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(features["left"]["mass_center_distances"], marker='o', linestyle='-', color='r')
        plt.title('Left Hand - Distances to Mass Center')
        plt.xlabel('Feature Index')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.savefig(output_path + '_left_mass_center_distances.png')
        plt.show()

    # plot features for the right hand
    if "right" in features:
        plt.figure(figsize=(10, 5))
        plt.plot(features["right"]["distances"], marker='o', linestyle='-', color='b')
        plt.title('Right Hand - Pairwise Distances Between Keypoints')
        plt.xlabel('Feature Index')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.savefig(output_path + '_right_distances.png')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(features["right"]["angles"], marker='o', linestyle='-', color='g')
        plt.title('Right Hand - Angles Between Keypoint Vectors')
        plt.xlabel('Feature Index')
        plt.ylabel('Angle (Radians)')
        plt.grid(True)
        plt.savefig(output_path + '_right_angles.png')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(features["right"]["mass_center_distances"], marker='o', linestyle='-', color='r')
        plt.title('Right Hand - Distances to Mass Center')
        plt.xlabel('Feature Index')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.savefig(output_path + '_right_mass_center_distances.png')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to keypoints JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to save features')
    parser.add_argument('--image_width', type=int, default=314, help='Width of the image (default: 512)')
    parser.add_argument('--image_height', type=int, default=512, help='Height of the image (default: 314)')
    args = parser.parse_args()

    save_features(args.input, args.output, args.image_width, args.image_height)
