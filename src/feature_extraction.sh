#!/bin/bash

INPUT_DIR="../sample_keypoint"
OUTPUT_DIR="../sample_feature"

mkdir -p "$OUTPUT_DIR"

for FILE in "$INPUT_DIR"/*.json; do
  BASENAME=$(basename "$FILE" .json)
  OUTPUT_PATH="$OUTPUT_DIR/$BASENAME"
  
  echo "Processing $FILE..."

  if python feature_extraction.py --input "$FILE" --output "$OUTPUT_PATH"; then
    echo "Feature extraction successful for $FILE."
  else
    echo "Feature extraction failed for $FILE."
  fi
done

echo "Feature extraction completed for all JSON files."
