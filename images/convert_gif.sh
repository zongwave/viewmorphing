#!/bin/bash

format="jpg"
prefix="minion"
save_output="true"
output_file="out_"${input_file}

interp_count=32

if [ "x$1" != "x" ]; then
    prefix=$1
fi

if [ "x$2" != "x" ]; then
    interp_count=$2
fi

# Create the list of images in the desired order
echo "file '0_out_${prefix}.${format}'" > images.txt
for (( i=1; i<=interp_count+1; i++ )); do
  echo "file '${i}_out_${prefix}.${format}'" >> images.txt
done

for (( i=interp_count; i>=1; i-- )); do
  echo "file '${i}_out_${prefix}.${format}'" >> images.txt
done
echo "file '0_out_${prefix}.${format}'" >> images.txt

# Use ffmpeg to create the GIF from the list of images
ffmpeg -f concat -safe 0 -i images.txt -vf "fps=30" -loop 0 out_${prefix}.gif

# Print a success message
echo "GIF created successfully: out_${prefix}.gif"

