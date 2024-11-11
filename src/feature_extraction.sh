#!/bin/bash

INPUT_DIR="../sample_keypoint"
OUTPUT_DIR="../sample_feature"
IMAGE_WIDTH=512
IMAGE_HEIGHT=314


mkdir -p "$OUTPUT_DIR"


for FILE in "$INPUT_DIR"/*.json; do
  BASENAME=$(basename "$FILE" .json)
  
  OUTPUT_PATH="$OUTPUT_DIR/$BASENAME"

  echo "processing $FILE..."
  python feature_extraction.py --input "$FILE" --output "$OUTPUT_PATH" --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT
done

echo "feature extraction completed for all JSON files."
