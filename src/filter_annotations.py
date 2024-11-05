import os
import json
from pycocotools.coco import COCO

source_dir = '../raw_annotations/test'
split_data_dir = '../data'
output_dir = '../annotations'
os.makedirs(output_dir, exist_ok=True)

# collect `cam_ids`, `image_ids`, and `seq_names` for each split
def collect_ids(split_dir):
    collected_data = {}
    for seq_name in os.listdir(split_dir):
        seq_path = os.path.join(split_dir, seq_name)
        if os.path.isdir(seq_path):
            collected_data[seq_name] = {}
            for cam_folder in os.listdir(seq_path):
                cam_id = cam_folder.replace('cam', '')
                cam_path = os.path.join(seq_path, cam_folder)
                collected_data[seq_name][cam_id] = set()
                for image_file in os.listdir(cam_path):
                    numeric_part = ''.join(filter(str.isdigit, image_file))
                    image_id = int(numeric_part)
                    collected_data[seq_name][cam_id].add(image_id)
    return collected_data

# InterHand2.6M_data.json 
def filter_coco_data(data_file, collected_data):
    coco = COCO(data_file)
    
    filtered_images = []
    for img in coco.dataset['images']:
        seq_name = img['seq_name']
        cam_id = img['camera']
        frame_idx = int(img['frame_idx'])

        if (seq_name in collected_data and 
            cam_id in collected_data[seq_name] and 
            frame_idx in collected_data[seq_name][cam_id] and 
            img.get('capture') == 1):
            
            filtered_images.append(img)
    
    filtered_image_ids = {img['id'] for img in filtered_images}
    
    filtered_annotations = [
        ann for ann in coco.dataset['annotations']
        if ann['image_id'] in filtered_image_ids
    ]
    
    return {
        "images": filtered_images,
        "annotations": filtered_annotations
    }

# InterHand2.6M_camera.json 
def filter_camera_data(camera_data, cam_ids):
    filtered_data = {}
    capture_data = camera_data.get("1", {})
    filtered_attributes = {}
    for attribute_name in ['campos', 'camrot', 'focal', 'princpt']:
        if attribute_name in capture_data:
            filtered_cameras = {cam: details for cam, details in capture_data[attribute_name].items() if cam in cam_ids}
            if filtered_cameras:
                filtered_attributes[attribute_name] = filtered_cameras
    if filtered_attributes:
        filtered_data["1"] = filtered_attributes
    return filtered_data

# InterHand2.6M_joint_3d.json 
def filter_joint_data(joint_data, collected_data):
    filtered_data = {"1": {}}
    frames = joint_data.get("1", {})
    for seq_name, cams in collected_data.items():
        for cam_id, img_ids in cams.items():
            for frame_idx, frame_data in frames.items():
                if int(frame_idx) in img_ids:
                    filtered_data["1"][str(frame_idx)] = frame_data
    if not filtered_data["1"]:
        return {}
    return filtered_data

# InterHand2.6M_MANO_NeuralAnnot.json 
def filter_mano_data(mano_data, collected_data):
    filtered_data = {"1": {}}
    frames = mano_data.get("1", {})
    for seq_name, cams in collected_data.items():
        for cam_id, img_ids in cams.items():
            for frame_idx, frame_data in frames.items():
                if int(frame_idx) in img_ids:
                    filtered_data["1"][str(frame_idx)] = frame_data
    if not filtered_data["1"]:
        return {}
    return filtered_data

for split in ['train', 'val', 'test']:
    split_dir = os.path.join(split_data_dir, split)
    split_annotation_dir = os.path.join(output_dir, split)
    os.makedirs(split_annotation_dir, exist_ok=True)

    collected_data = collect_ids(split_dir)
    
    # filter InterHand2.6M_data.json
    data_file = os.path.join(source_dir, 'InterHand2.6M_test_data.json')
    filtered_data = filter_coco_data(data_file, collected_data)
    with open(os.path.join(split_annotation_dir, 'InterHand2.6M_data.json'), 'w') as f:
        json.dump(filtered_data, f)
    
    # filter InterHand2.6M_camera.json
    with open(os.path.join(source_dir, 'InterHand2.6M_test_camera.json'), 'r') as f:
        camera_data = json.load(f)
    filtered_camera_data = filter_camera_data(camera_data, {cam for seq in collected_data.values() for cam in seq})
    with open(os.path.join(split_annotation_dir, 'InterHand2.6M_camera.json'), 'w') as f:
        json.dump(filtered_camera_data, f)
    
    # filter InterHand2.6M_joint_3d.json
    with open(os.path.join(source_dir, 'InterHand2.6M_test_joint_3d.json'), 'r') as f:
        joint_3d_data = json.load(f)
    filtered_joint_3d_data = filter_joint_data(joint_3d_data, collected_data)
    with open(os.path.join(split_annotation_dir, 'InterHand2.6M_joint_3d.json'), 'w') as f:
        json.dump(filtered_joint_3d_data, f)
    
    # filter InterHand2.6M_MANO_NeuralAnnot.json
    with open(os.path.join(source_dir, 'InterHand2.6M_test_MANO_NeuralAnnot.json'), 'r') as f:
        mano_data = json.load(f)
    filtered_mano_data = filter_mano_data(mano_data, collected_data)
    with open(os.path.join(split_annotation_dir, 'InterHand2.6M_MANO_NeuralAnnot.json'), 'w') as f:
        json.dump(filtered_mano_data, f)
