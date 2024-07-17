#! /bin/sh

input_format="jpg"
input_file="minion"
save_output="true"
output_file="out_"${input_file}

interp_count=8

if [ "x$1" != "x" ]; then
    input_file=$1
fi

if [ "x$2" != "x" ]; then
    input_format=$2
fi

if [ "x$3" != "x" ]; then
    save_output=$3
fi

# Use ffprobe to get the width and height
dimensions=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "./images/${input_file}_1.${input_format}")

# Extract width and height
width=$(echo ${dimensions} | cut -d'x' -f1)
height=$(echo ${dimensions} | cut -d'x' -f2)

# Print the dimensions
echo "Width: ${width}, Height: ${height}"

./test-freeview --interp-count ${interp_count} --input ./images/${input_file}_1.${input_format} --input ./images/${input_file}_2.${input_format} --in-w ${width} --in-h ${height} --output ${output_file}.${input_format} --out-w ${width} --out-h ${height} --save ${save_output}

