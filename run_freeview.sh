#! /bin/sh

input_file_1="minion_1.jpg"
input_file_2="minion_2.jpg"

save_output="true"

interp_count=8

if [ "x$1" != "x" ]; then
    input_file_1=$1
fi

if [ "x$2" != "x" ]; then
    input_file_2=$2
fi

# Extract the base name from input_file_1
base_name="${input_file_1%_*}"
echo "base name: $base_name"

# Define the output file name
output_file="${base_name}_out.jpg"

if [ "x$3" != "x" ]; then
    interp_count=$3
fi

if [ "x$4" != "x" ]; then
    save_output=$4
fi

# Use ffprobe to get the width and height
dimensions=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "./images/${input_file_1}")

# Extract width and height
width=$(echo ${dimensions} | cut -d'x' -f1)
height=$(echo ${dimensions} | cut -d'x' -f2)

# Print the dimensions
echo "Width: ${width}, Height: ${height}"

./test-freeview --interp-count ${interp_count} --input ./images/${input_file_1} --input ./images/${input_file_2} --in-w ${width} --in-h ${height} --output ./images/out/${output_file} --out-w ${width} --out-h ${height} --save ${save_output}

