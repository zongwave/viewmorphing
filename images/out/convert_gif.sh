#!/bin/bash

# Check if prefix argument is provided, otherwise use default
prefix=${1:-"minion_out"}
output_file="${prefix}.gif"

# Find all image files with the specified prefix and supported extensions
files=$(ls -1 "${prefix}"*.{jpg,jpeg,png,bmp} 2>/dev/null | sort -V)

if [ -z "$files" ]; then
    echo "No image files found with prefix '${prefix}' and supported extensions (jpg, jpeg, png, bmp)."
    exit 1
fi

# Remove existing images.txt file if it exists
if [ -f images.txt ]; then
    rm images.txt
fi

# Generate images.txt with the list of images
count=$(echo "${files}" | wc -l)

for (( i=0; i<${count}; i++ )); do
    echo "file '${prefix}_${i}.jpg'" >> images.txt
done

for (( i=${count}-1; i>=0; i-- )); do
    echo "file '${prefix}_${i}.jpg'" >> images.txt
done

# Use ffmpeg to create the GIF from the list of images
ffmpeg -f concat -safe 0 -i images.txt -vf "fps=30" -loop 0 "${output_file}"

# Clean up the images.txt file
if [ -f images.txt ]; then
    rm images.txt
fi

# Print a success message
echo "GIF created successfully: ${output_file}"
