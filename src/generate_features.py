import json
import os
from tqdm import tqdm  
from feature_extraction import extract_features  

def process_keypoints(keypoints_file, output_file, image_width, image_height):
    with open(keypoints_file, 'r') as f:
        keypoints_data = json.load(f)
    
    features_data = {}
 
    for sample_id, hands_data in tqdm(keypoints_data.items(), desc="extracting features"):
        features = {}
        # extract features for left hand if available
        if "left" in hands_data and hands_data["left"]:
            left_keypoints = hands_data["left"]
            distances, angles, mass_center_distances = extract_features(left_keypoints, image_width, image_height)
            features["left"] = {
                "distances": distances,
                "angles": angles,
                "mass_center_distances": mass_center_distances
            }
        
        # extract features for right hand if available
        if "right" in hands_data and hands_data["right"]:
            right_keypoints = hands_data["right"]
            distances, angles, mass_center_distances = extract_features(right_keypoints, image_width, image_height)
            features["right"] = {
                "distances": distances,
                "angles": angles,
                "mass_center_distances": mass_center_distances
            }
        
        features_data[sample_id] = features
    
    with open(output_file, 'w') as f:
        json.dump(features_data, f, indent=4)
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    dir = 'annotations'
    splits = ['train', 'val', 'test']
    os.makedirs(dir, exist_ok=True)

    for split in splits:
        keypoints_file = os.path.join(dir, f'{split}/keypoints.json')
        output_file = os.path.join(dir, f'{split}/features.json')
        
        print(f'processing {split} dataset...')
        process_keypoints(keypoints_file, output_file, 324, 512)
