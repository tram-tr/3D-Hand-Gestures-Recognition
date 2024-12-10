import os
import json

def generate_labels(data_dir, output_file):
    labels = {}
    for rom_dir in os.listdir(data_dir):
        rom_path = os.path.join(data_dir, rom_dir)

        if os.path.isdir(rom_path):
            rom_id = rom_dir.split('_')[0]  # use ROM directory name as the label
            if rom_id == 'ROM01':
                rom_label = 'No interaction between 2 hands'
            elif rom_id == 'ROM02':
                rom_label = 'Interaction between 2 Hands'
            elif rom_id == 'ROM03':
                rom_label = 'Left hand (LT) with no occlusions'
            elif rom_id == 'ROM04':
                rom_label = 'Left hand (LT) with occlusion'
            elif rom_id == 'ROM05':
                rom_label = 'Right hand (LT) with no occlusions'
            elif rom_id == 'ROM06':
                rom_label = 'Right hand (RT) with occlusion'
            elif rom_id == 'ROM07':
                rom_label = 'Right-hand finger occlusions'
            elif rom_id == 'ROM08':
                rom_label = 'Left-hand finger occlusions'
            elif rom_id == 'ROM09':
                rom_label = 'Interaction fingers touching'
            else:
                rom_label = ''

            for cam_id in os.listdir(rom_path):
                cam_path = os.path.join(rom_path, cam_id)

                if os.path.isdir(cam_path):
                    for image_file in os.listdir(cam_path):
                        if image_file.endswith(".jpg"):
                            image_id = os.path.splitext(image_file)[0] 
                            key = f"{cam_id.split('cam')[-1]}_{image_id.split('image')[-1]}"
                            labels[key] = [rom_id, rom_label]
    
    # save the labels to a JSON file
    with open(output_file, "w") as f:
        json.dump(labels, f, indent=4)
    print(f"labels saved to {output_file}")

if __name__ == "__main__":
    base_dir = 'data'
    sets = ['train', 'val', 'test']

    for dataset in sets:
        data_path = os.path.join(base_dir, dataset)
        output_path = f"annotations/{dataset}/labels.json"
        generate_labels(data_path, output_path)
