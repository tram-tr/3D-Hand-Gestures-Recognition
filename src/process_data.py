import os
import shutil

def process_images(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        output_split_dir = os.path.join(output_dir, split)

        if not os.path.exists(output_split_dir):
            os.makedirs(output_split_dir)

        for rom_dir in os.listdir(split_dir):
            rom_path = os.path.join(split_dir, rom_dir)

            if os.path.isdir(rom_path):
                for cam_id in os.listdir(rom_path):
                    cam_path = os.path.join(rom_path, cam_id)

                    if os.path.isdir(cam_path):
                        for image_file in os.listdir(cam_path):
                            if image_file.endswith('.jpg'):
                                image_id = os.path.splitext(image_file)[0].replace('image', '')
                                new_name = f"{cam_id.split('cam')[-1]}_{image_id}.jpg"
                                src_path = os.path.join(cam_path, image_file)
                                dst_path = os.path.join(output_split_dir, new_name)

                                shutil.copy(src_path, dst_path)

        print(f'processed {split} images into {output_split_dir}')


if __name__ == "__main__":
    data_dir = "data"  
    output_dir = "processed_data"  

    process_images(data_dir, output_dir)
