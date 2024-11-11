#!/bin/bash

BASE_INPUT_DIR="../sample"
BASE_OUTPUT_DIR="../sample_keypoint"

# list of folders to process
FOLDERS=(
  "ROM01_No_Interaction_2_Hand"
  "ROM02_Interaction_2_Hand"
  "ROM03_LT_No_Occlusion"
  "ROM04_LT_Occlusion"
  "ROM04_RT_Occlusion"
  "ROM05_RT_Wrist_ROM"
  "ROM07_Rt_Finger_Occlusions"
  "ROM08_Lt_Finger_Occlusions"
  "ROM09_Interaction_Fingers_Touching"
)

mkdir -p "$BASE_OUTPUT_DIR"

for FOLDER in "${FOLDERS[@]}"; do
  INPUT_DIR="$BASE_INPUT_DIR/$FOLDER"

  for IMAGE_FILE in "$INPUT_DIR"/*.jpg; do
    BASENAME=$(basename "$IMAGE_FILE" .jpg)
    
    OUTPUT_PATH="$BASE_OUTPUT_DIR/$BASENAME"

    echo "processing $IMAGE_FILE..."
    
    # Run the keypoint detection script
    python keypoint_detection.py --input "$IMAGE_FILE" --output "$OUTPUT_PATH"
  done
done

echo "Keypoint detection completed for all images."
