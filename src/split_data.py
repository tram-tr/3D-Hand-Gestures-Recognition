import os
import shutil
import random

source_dir = '../raw_data'  
output_dir = '../data'
train_ratio, val_ratio = 0.6, 0.2

train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)

# iterate over each main folder (e.g., ROM01_No_Interaction_2_Hand, ROM02_Interaction_2_Hand, etc.)
for main_folder in os.listdir(source_dir):
    main_folder_path = os.path.join(source_dir, main_folder)
    if os.path.isdir(main_folder_path):
        # cam[id] subfolders within the main folder
        cam_folders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
        
        random.shuffle(cam_folders)
        
        num_cam_folders = len(cam_folders)
        train_idx = int(num_cam_folders * train_ratio)
        val_idx = train_idx + int(num_cam_folders * val_ratio)
        
        # split cam folders into train, val, and test sets
        train_cams = cam_folders[:train_idx]
        val_cams = cam_folders[train_idx:val_idx]
        test_cams = cam_folders[val_idx:]
        
        for cam_folder in train_cams:
            src = os.path.join(main_folder_path, cam_folder)
            dst = os.path.join(train_dir, main_folder, cam_folder)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)
        
        for cam_folder in val_cams:
            src = os.path.join(main_folder_path, cam_folder)
            dst = os.path.join(val_dir, main_folder, cam_folder)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)
        
        for cam_folder in test_cams:
            src = os.path.join(main_folder_path, cam_folder)
            dst = os.path.join(test_dir, main_folder, cam_folder)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)
        

